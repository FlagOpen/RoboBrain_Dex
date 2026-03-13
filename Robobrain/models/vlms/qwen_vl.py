import torch
import torch.nn as nn
from typing import Callable, Dict, List, Optional, Sequence, Type
from pathlib import Path
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from Robobrain.models.vlms.base_vlm import VLM
from Robobrain.models.backbones.llm.prompting import Qwen2VLPromptBuilder, PromptBuilder
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer
from qwen_vl_utils import process_vision_info 

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
BOA_TOKEN = 151665
BOR_TOKEN = 151666
_DYN_TOKEN_MIN = 151667 
_DYN_TOKEN_MAX = 151682 
_DEX_TOKEN_MIN = 151683
_DEX_TOKEN_MAX = 152194

# =======================================================================
#   1. Qwen2.5-VL Compatible VLM (full-finetune + FSDP-ready)
# =======================================================================
class DummyLLMBackbone(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.identifier = "qwen-llm-backbone"
        self.llm = model
        self.tokenizer = processor.tokenizer
        self.embed_dim = model.config.hidden_size
        self.processor = processor
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def embed_input_ids(self, input_ids):
        return self.llm.model.language_model.embed_tokens(input_ids)
    
    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return Qwen2VLPromptBuilder
    
    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Qwen2_5_VLDecoderLayer

class DummyVisionBackbone(nn.Module):
    def __init__(self, model, processor):
        super().__init__()
        self.identifier = "qwen-vision-backbone"
        self.model = model
        self.embed_dim = model.config.vision_config.hidden_size
        self.image_processor = processor.image_processor
        
        self._default_image_resolution = (3, 224, 224)
        self.num_patches = 256

    def forward(self, pixel_values):
        return self.model.model.vision_tower(pixel_values)
    
    def get_image_transform(self):
        def _transform(images):
            # if isinstance(images, list):
            #     out = self.image_processor(videos=images, return_tensors="pt")
            # else:
            #     out = self.image_processor(images=images, return_tensors="pt")
            out = self.image_processor.preprocess(images, return_tensors="pt")
            pv = out["pixel_values"]  
            if isinstance(pv, List):
                pv = pv[0]
            grid = out["image_grid_thw"][0]
            return {
                "pixel_values": pv,       # [T, 3, H, W] or [B, T, 3, H, W]
                "image_grid_thw": grid,   # [T, 3] or [B, T, 3]
            }
        return _transform
    
    @property
    def default_image_resolution(self):
        return self._default_image_resolution
    
class DummyProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.identifier = "qwen-projector"

    def forward(self, x):
        return x
    
class QwenVLVLM(VLM):
    """
    Fully-compatible VLM wrapper for Qwen2.5-VL.
    - No vision_backbone / llm_backbone separation (set to None)
    - Only supports full-finetune (all params trainable)
    - Supports FSDP by wrapping transformer decoder layers
    - Forward / generate follow HF Qwen2_5_VLForConditionalGeneration
    """

    def __init__(self, model_id: str, device_map="cuda"):
        # Load Qwen-VL
        nn.Module.__init__(self)
        hf_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            cache_dir="Your Qwen Directory",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float32,
            device_map=device_map,
        )
        self.arch_specifier = "DummyProjector"
        self.processor = AutoProcessor.from_pretrained(model_id, cache_dir="Robobrain-Dex Directory")
        self.processor.tokenizer.padding_side = "right"
        llm_backbone = DummyLLMBackbone(hf_model, self.processor)
        vision_backbone = DummyVisionBackbone(hf_model, self.processor)
        super().__init__(
            model_family="qwen-vl",
            model_id=model_id,
            vision_backbone=vision_backbone,
            llm_backbone=llm_backbone,
            enable_mixed_precision_training=True,
        )
        self.model = hf_model
        self.projector = DummyProjector()
        # Required for Base VLM
        self.all_module_keys = ["model"]
        self.trainable_module_keys = ["model"]

    # ===================================================================
    #   2. Required Abstract Methods
    # ===================================================================
    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_family: str,
        model_id: str,
        vision_backbone=None,
        llm_backbone=None,
        **kwargs
    ):
        """Load Qwen-VL from checkpoint (your own format)."""
        vlm = cls(model_id)
        state = torch.load(pretrained_checkpoint, map_location="cpu")
        vlm.model.load_state_dict(state["model"])
        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None):
        """VLA pretrain does not need prompt builder."""
        raise NotImplementedError("PromptBuilder is not used in QwenVLVLM.")

    def freeze_backbones(self, stage: str) -> None:
        """Only support full-finetune."""
        assert stage in {"full-finetune", "vla-full-train"}, \
            "QwenVLVLM only supports full-finetune."
        self.model.requires_grad_(True)
        self.trainable_module_keys = ["model"]

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Simplified checkpoint loading."""
        if pretrained_checkpoint is None:
            pretrained_checkpoint = run_dir / "checkpoints" / "latest-checkpoint.pt"

        state = torch.load(pretrained_checkpoint, map_location="cpu")
        self.model.load_state_dict(state["model"])

    # ===================================================================
    #   3. FSDP Auto-Wrap Policy
    # ===================================================================
    def get_fsdp_wrapping_policy(self) -> Callable:
        """
        For Qwen2.5-VL, wrap only the decoder layers.
        """
        # from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLDecoderLayer

        # return partial(
        #     transformer_auto_wrap_policy,
        #     transformer_layer_cls={Qwen2_5_VLDecoderLayer},
        # )
        def no_wrap_policy(module, recurse, nonwrapped_numel):
            return False
        return no_wrap_policy
        
    # ===================================================================
    #   4. Forward & Generate
    # ===================================================================
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Direct call to HF model forward."""

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            **kwargs,
        )

    def generate(self, **kwargs):
        return self.model.generate(**kwargs)

    # ===================================================================
    #   5. Build Multi-Modal Inputs
    # ===================================================================
    def build_qwenvl_inputs(self, images, instructions, solutions=None, **kwargs):
        """
        Construct and tokenize multimodal chat-style inputs for Qwen2.5-VL (batched).

        Overview:
            For each sample i:
                - Takes a list of PIL images: images[i] = [img_0, img_1, ...]
                - Takes a matching instruction string instructions[i]
                - Optionally formats instruction with a chain-of-thought template (CoT_prompt) if present in config.
                - Builds a single-turn chat message containing:
                      [{"role": "user", "content": [
                          {"type": "image", "image": <PIL.Image>}, ...,
                          {"type": "text", "text": <final_prompt>}
                      ]}]
                - Applies processor.apply_chat_template(..., add_generation_prompt=True)
                - Extracts vision inputs via process_vision_info
                - Calls processor(...) to produce a BatchFeature with token + vision tensors.

        Parameters:
            images (List[List[PIL.Image.Image]]):
                Length B. Each element is a (possibly empty) list of PIL images associated with that instruction.
                Supports multi-image inputs (ordered). For video-as-frames, upstream code should decide packaging.
            instructions (List[str]):
                Length B textual prompts or task instructions.
            **kwargs:
                Reserved for future extensions (e.g., system prompts, style controls, additional metadata).

        Config Dependencies:
            self.config.datasets.vla_data.get("CoT_prompt", str):
                If present, each instruction string is injected into the template by replacing "{instruction}".

        Returns:
            BatchFeature (HF):
                Typical keys (moved to self.model.device):
                    input_ids: LongTensor [B, T]
                    attention_mask: LongTensor/Bool [B, T]
                    pixel_values / image_grid / video specifics (model-dependent)
                    (Possibly) token_type_ids or other processor outputs
                The structure aligns with what Qwen2_5_VLForConditionalGeneration.forward expects.

        Shapes / Notes:
            - Sequence length T varies by number of images (special tokens) + prompt length.
            - pixel_values may have internal batching distinct from B if images are flattened; underlying model maps them.
            - The association between images and textual placeholders is preserved by processor ordering.

        Edge Cases:
            - Empty image list per sample is allowed (pure text prompt).
            - Mismatched lengths of images and instructions raise AssertionError.
            - CoT prompt replacement is naive string replace; ensure template contains "{instruction}" placeholder.

        Performance:
            - This path aims for faster inference vs. more granular per-turn assembly.
            - Minor tokenization differences (e.g., whitespace) can affect highly overfitted benchmarks.

        Does Not:
            - Perform augmentation.
            - Cache processed pixel tensors.
            - Handle streaming input.

        """

        # Create messages: one message per sample
        messages = []
        assert len(images) == len(instructions), "Images and instructions must have the same length"
        for imgs, instruction in zip(images, instructions):

            if "CoT_prompt" in self.config.datasets.vla_data:  # If using a grounding prompt to task
                content = [{"type": "image", "image": img} for img in imgs]
                CoT_prompt = self.config.datasets.vla_data.get("CoT_prompt", "")
                prompt = CoT_prompt.replace("{instruction}", instruction)
                content.append({"type": "text", "text": prompt})
            elif "Wds_prompt" in self.config.datasets.vla_data:
                prefix_prompt = instruction.split(":")[0] + ": "
                if len(imgs) !=3:
                    content = [
                        {"type": "text", "text": "".join(prefix_prompt)}, 
                        {"type": "text", "text": "robot front image"}, 
                        {"type": "image", "image": imgs[0]},# {"type": "image", "image": f"data:image;base64,{agentview_base64}"},
                        {"type": "text", "text": " and robot wrist image"},
                        {"type": "image", "image": imgs[1]}]
                    if len(instruction.split(":")) == 4:
                        all_task = instruction.split(":")[-2].split('.')[0].strip()
                        instr = instruction.split(":")[-1].strip()
                        content.append({"type": "text", "text": f"\nYour overall task is: {all_task.lower()}. Currently, focus on completing the subtask: {instr.lower()}"},)
                    else:
                        instr = instruction.split(":")[-1].strip()
                        content.append({"type": "text", "text": f"\nYour overall task is: {instr.lower()}"})
                    
                else:
                    content = [
                        {"type": "text", "text": "".join(prefix_prompt)}, 
                        {"type": "text", "text": "robot front image"}, 
                        {"type": "image", "image": imgs[0]},# {"type": "image", "image": f"data:image;base64,{agentview_base64}"},
                        {"type": "text", "text": ", right wrist image"},
                        {"type": "image", "image": imgs[1]},# {"type": "image", "image": f"data:image;base64,{wrist_right_base64}"},
                        {"type": "text", "text": " and left wrist image"},
                        {"type": "image", "image": imgs[2]}]
                    if len(instruction.split(":")) == 4:
                        all_task = instruction.split(":")[-2].split('.')[0].strip()
                        instr = instruction.split(":")[-1].strip()
                        content.append({"type": "text", "text": f"\nYour overall task is: {all_task.lower()}. Currently, focus on completing the subtask: {instr.lower()}"},)
                    else:
                        # print(instruction)
                        all_task = instruction.split(":")[-2].split('.')[0].strip()
                        instr = instruction.split(":")[-1].strip()
                        content.append({"type": "text", "text": f"\nYour overall task is: {all_task.lower()}. Currently, focus on completing the subtask: {instr.lower()}"})
            else:
                content = [{"type": "image", "image": img} for img in imgs]
                prompt = instruction
                content.append({"type": "text", "text": prompt})

            msg = [{"role": "user", "content": content}]

            if solutions is not None:
                solution = solutions[len(messages)]
                msg.append({"role": "assistant", "content": [{"type": "text", "text": solution}]})
            messages.append(msg)

        # Prepare text prompts using processor
        # default process is json --> message --> texts --> input_ids
        texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]

        # image_inputs = list of PIL
        image_inputs, video_inputs = process_vision_info(messages)
        batch_input = self.processor(text=texts, images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")


        # if solutions, mask out the solution tokens in labels
        if solutions is not None:
            action_token_min = _ACTION_TOKEN_MIN # how can we know this range? --> we has other way for this, but is slower see qwenhelix branch
            action_token_max = _ACTION_TOKEN_MAX # here only for fast_tokenizer
            labels = batch_input['input_ids'].clone()
            # For each sequence in the batch, find the first occurrence of an action token.
            for i in range(labels.size(0)):
                seq = labels[i]
                # Create a mask for tokens within the action token range.
                mask_seq = (seq >= action_token_min) & (seq <= action_token_max)
                nonzero_indices = torch.nonzero(mask_seq, as_tuple=False)
                if nonzero_indices.numel() > 0:
                    first_action_index = nonzero_indices[0].item()
                    # Mask out all tokens before the first action token.
                    seq[:first_action_index] = IGNORE_INDEX
                else:
                    # If no action token is found, mask the entire sequence.
                    seq[:] = IGNORE_INDEX
                    
            
            labels[labels == self.processor.tokenizer.pad_token_id] = -100 ## mask out pad tokens as well
            batch_input['labels'] = labels

        return batch_input.to(self.model.device)