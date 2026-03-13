import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import draccus
import timm
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from timm.models.vision_transformer import LayerScale
from transformers import AutoTokenizer

from Robobrain.conf import ModelConfig
from Robobrain.extern.hf.configuration_robobrain import OpenVLAConfig, DexVLAConfig
from Robobrain.extern.hf.modeling_robobrain import OpenVLAForActionPrediction, DexVLAForActionPrediction
from Robobrain.extern.hf.processing_robobrain import PrismaticImageProcessor, PrismaticProcessor
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    GenerationMixin,
    AutoConfig,
)

@dataclass
class HFConvertConfig:
    # fmt: off
    dexvla_model_path_or_id: Union[str, Path] = (                       # Path to Pretrained VLA (on disk or HF Hub)
        "Pretrained VLA Directory"
    )
    ckpt_name: Path = Path('step-040000-epoch-00-loss=0.7969.pt')       # The specific checkpoint to be converted (modify accordingly)
    output_hf_model_local_path: Path = Path(                            # Path to Local Path to save HF model
        "Output Directory"
    )

    # HF Hub Credentials (required for Gated Models like LLaMa-2)
    hf_token: Union[str, Path] = ''                                     # Environment variable or Path to HF Token
    lamtype: str = "RQ_HandLatentActionModel"
    codebook_size: int = 16                                             # Latent action codebook size                                             
    motion_codebook_size: int = 512
    use_diffusion_head: bool = False
    window_size: int = 32
    input_dim: int = 48
    def __post_init__(self) -> None:
        self.hf_token = self.hf_token.read_text().strip() if isinstance(self.hf_token, Path) else self.hf_token

    # fmt: on


# HF Transformers overwrites parameters with names containing `gamma`; we're going to patch VisionBackbone.LayerScale.
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109
#   =>> Transformers :: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L3960
def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


# === Conversion Constants ===
PROJECTOR_KEY_MAPPING = {
    "projector.0.weight": "projector.fc1.weight",
    "projector.0.bias": "projector.fc1.bias",
    "projector.2.weight": "projector.fc2.weight",
    "projector.2.bias": "projector.fc2.bias",
    "projector.4.weight": "projector.fc3.weight",
    "projector.4.bias": "projector.fc3.bias",
}


def remap_state_dicts_for_hf(
    # prismatic_vision_backbone_state_dict: List[Dict[str, torch.Tensor]],
    # projector_state_dict: Dict[str, torch.Tensor],
    llm_backbone_state_dict: Dict[str, torch.Tensor],
    action_decoder_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Iterate through Prismatic component state dictionaries and unify / fix key mapping for HF conversion."""
    hf_state_dict = {}
    # print("prismatic_vision_backbone_state_dict:", prismatic_vision_backbone_state_dict.keys())
    # print("projector_state_dict:", projector_state_dict.keys())
    print("llm_backbone_state_dict:", llm_backbone_state_dict.keys())
    print("action_decoder_dict:", action_decoder_dict.keys())

    # Iterate through LLM Backbone =>> replace `llm.` with `language_model.`
    for key, value in llm_backbone_state_dict.items():
        hf_state_dict[key.replace("llm.", "")] = value

    for key, value in action_decoder_dict.items():
        prefix = "action_decoder"
        hf_state_dict[f"{prefix}.{key}"] = value
    
    return hf_state_dict


@draccus.wrap()
def convert_dexvla_weights_to_hf(cfg: HFConvertConfig) -> None:
    print(f"[*] Converting DexVLA Model `{cfg.dexvla_model_path_or_id}` to HF Transformers Format")
    torch.set_default_dtype(torch.bfloat16)

    # Get `config.json`, 'dataset_statistics.json' and `checkpoint_pt` -- mirrors logic in `prismatic.models.load.py`
    if os.path.isdir(cfg.dexvla_model_path_or_id):
        print(f"[*] Loading from Local Path `{(run_dir := Path(cfg.dexvla_model_path_or_id))}`")
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / cfg.ckpt_name
        dataset_statistics_json = run_dir / "dataset_statistics.json"

        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"
    else:
        raise ValueError(
            f"Local path `{cfg.dexvla_model_path_or_id}` does not exist. "
            "Please set `dexvla_model_path_or_id` to a valid local checkpoint directory."
        )

    # Load "Native" Config JSON =>> Create LLM Config & Instantiate Tokenizer
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        prismatic_config = ModelConfig.get_choice_class(vla_cfg["base_vlm"])().__dict__

    # Load Normalization Statistics
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)


    hf_config = DexVLAConfig.from_pretrained("Pretrained VLA Directory",
                                            llm_backbone_id=prismatic_config["llm_backbone_id"],
                                            arch_specifier=prismatic_config["arch_specifier"],
                                            image_resize_strategy=prismatic_config["image_resize_strategy"],
                                            llm_max_length=prismatic_config["llm_max_length"],
                                            torch_dtype=torch.bfloat16,
                                            norm_stats=norm_stats,
                                            use_diffusion_head=cfg.use_diffusion_head,
                                            window_size=cfg.window_size,
                                            input_dim=cfg.input_dim,)

    # Instantiate & Add Pad to Tokenizer =>> following `prismatic.models.materialize.get_llm_backbone_and_tokenizer`
    #   TODO :: Implement batched generation -- in which case this should set `padding_side = "left"`!
    print("[*] Instantiating and Patching Tokenizer, LLM Config")
    qwen_processor = AutoProcessor.from_pretrained("Pretrained VLA Directory")
    qwen_processor.tokenizer.padding_side = "right"
    tokenizer = qwen_processor.tokenizer
   
    if cfg.lamtype == "UncontrolledHandVideo_LatentActionModel":
        special_tokens_dict = {'additional_special_tokens': ['<BOA>', '<BOR>'] + [f'<VIDEO_{i}>' for i in range(cfg.codebook_size)] + [f'<DEX_{i}>' for i in range(cfg.codebook_size)]}
    elif cfg.lamtype == "RQ_HandLatentActionModel":
        special_tokens_dict = {'additional_special_tokens': ['<BOA>', '<BOR>'] + [f'<DYN_{i}>' for i in range(cfg.codebook_size)] + [f'<DEX_{i}>' for i in range(cfg.motion_codebook_size)]}
    else:
        special_tokens_dict = {'additional_special_tokens': ['<BOA>', '<BOR>'] + [f'<ACT_{i}>' for i in range(cfg.codebook_size)]}
    tokenizer.add_special_tokens(special_tokens_dict)

    tokenizer.init_kwargs.pop("add_prefix_space", None)  # Pop to prevent unnecessary warning on reload...
    print("hf_config.pad_token_id:", hf_config.pad_token_id)
    assert tokenizer.pad_token_id == hf_config.pad_token_id, "Incorrect Pad Token ID!"
    assert len(tokenizer) > hf_config.text_config.vocab_size, "Tokenizer vocabulary must be larger than LLM vocabulary!"

    # Patch LLM Config in `hf_config` with vocab_size (+ `hf_config.pad_to_multiple_of`), pad_token_id + validate
    # hf_config.text_config.vocab_size += hf_config.pad_to_multiple_of
    def _pad_to_multiple(n, m):
        return (n + m - 1) // m * m

    vocab_target = _pad_to_multiple(len(tokenizer), hf_config.pad_to_multiple_of)
    hf_config.text_config.vocab_size = vocab_target
    hf_config.text_config.pad_token_id = hf_config.pad_token_id
    hf_config.text_config.torch_dtype = torch.bfloat16
    assert hf_config.text_config.use_cache, "LLM config `use_cache` should be True for inference (set default)!"

    hf_image_processor = qwen_processor.image_processor
    print("[*] Creating PrismaticProcessor Instance from Tokenizer and PrismaticImageProcessor")
    # hf_processor = PrismathicProcessor(
    #     image_processor=hf_image_processor, 
    #     tokenizer=qwen_processor.tokenizer
    # )
    
    # Load Prismatic Model State Dictionary (in preparation for conversion)
    print("[*] Loading Prismatic VLM State Dictionary from Checkpoint")
    model_state_dict = torch.load(checkpoint_pt, map_location="cpu")["model"]
    assert ("downsampler" not in model_state_dict) or (len(model_state_dict["downsampler"]) == 0), "Downsampler?"
    # assert all([k in model_state_dict for k in ["vlm.vision_backbone", "vlm.projector", "vlm.llm_backbone", "action_decoder"]]), "Missing keys!"
    assert all([k in model_state_dict for k in ["vlm.llm_backbone", "action_decoder"]]), "Missing keys!"

    # Convert
    print("[*] Running Conversion")
    converted_state_dict = remap_state_dicts_for_hf(
        # model_state_dict["vlm.vision_backbone"],
        # model_state_dict["vlm.projector"],
        model_state_dict["vlm.llm_backbone"],
        model_state_dict["action_decoder"],
    )

    # Create PrismaticForConditionalGeneration =>> Note that we can't initialize on `meta` device because TIMM
    print("[*] Building (Randomly Initialized) Model =>> DexVLAForActionPrediction")
    hf_model = DexVLAForActionPrediction(hf_config)

    ### With tokenizer not padded to the multiple of 64 ( 32064 -> 32033 )
    hf_model.language_model.resize_token_embeddings(len(tokenizer), hf_config.pad_to_multiple_of)
    print("len(tokenizer) =", len(tokenizer))
    print("config vocab_size =", hf_config.text_config.vocab_size)
    print("checkpoint vocab_size =", converted_state_dict["model.language_model.embed_tokens.weight"].shape[0])
    hf_model.load_state_dict(converted_state_dict, strict=True, assign=True)

    # Cast Model to BF16 before Saving
    hf_model.to(torch.bfloat16)

    # Save Pretrained Versions to Local Path
    print("[*] Saving Model & Processor to Local Path")
    hf_model.save_pretrained(cfg.output_hf_model_local_path, max_shard_size="7GB")
    hf_image_processor.save_pretrained(cfg.output_hf_model_local_path)
    qwen_processor.save_pretrained(cfg.output_hf_model_local_path)

    # Copy `dataset_statistics.json` File to Converted Checkpoint Directory
    output_dataset_statistics_json = cfg.output_hf_model_local_path / "dataset_statistics.json"
    shutil.copyfile(dataset_statistics_json, output_dataset_statistics_json)

    print(f"[*] Saving Complete! Saved converted checkpoint to: {cfg.output_hf_model_local_path}")


if __name__ == "__main__":
    convert_dexvla_weights_to_hf()
