"""
deploy.py

Starts VLA server which the client can query to get robot actions.
"""

import os.path

# ruff: noqa: E402
import json_numpy

json_numpy.patch()
import json
import logging
import numpy as np
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import draccus
import torch
import torchvision
import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from PIL import Image
import time

from real_world_deployment import DexVLAInference
from Robobrain.extern.hf.modeling_robobrain import ActionDecoder


# === Server Interface ===
class DexVLAServer:
    def __init__(self, cfg) -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given observation + instruction.
        """
        self.cfg = cfg

        self.policy = DexVLAInference(saved_model_path=cfg.pretrained_checkpoint, pred_action_horizon=32)
        self.policy.vla.action_decoder = ActionDecoder(window_size=32, input_dim=48, vis_dim=2048, hidden_dim=512).to("cuda:0") #4096 ---> 2048
        
        decoder_path = os.path.join(cfg.pretrained_checkpoint, "action_decoder.pt")
        checkpoint = torch.load(decoder_path, map_location='cpu')
        updated_checkpoint = {}
        for key, value in checkpoint.items():
            # Strip LoRA wrapper prefix to get the base model key
            if key.startswith('modules_to_save.default.'):
                new_key = key.replace('modules_to_save.default.', '')
                updated_checkpoint[new_key] = value

            # Full checkpoint loading (for non-LoRA checkpoints)
            # new_key = key.replace('modules_to_save.default.', '')
            # updated_checkpoint[new_key] = value
        self.policy.vla.action_decoder.to(torch.float)
        self.policy.vla.action_decoder.load_state_dict(updated_checkpoint)


    def get_server_action(self, payload: Dict[str, Any]) -> str:
        try:
            if double_encode := "encoded" in payload:
                # Support cases where `json_numpy` is hard to install, and numpy arrays are "double-encoded" as strings
                assert len(payload.keys()) == 1, "Only uses encoded payload!"
                payload = json.loads(payload["encoded"])

            observation = payload
            instruction = observation["instruction"]
            img = observation["image"]
            proprio = torch.tensor(observation["state"], dtype=torch.float32).to("cuda:0")

            unnorm_key = "pour_the_drink" 
            action_norm_stats = self.policy.vla.get_action_stats(unnorm_key)
            mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
            mask = torch.tensor(mask, dtype=torch.bool)
            action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
            action_high = torch.from_numpy(np.array(action_norm_stats["max"]))
            action_low = torch.from_numpy(np.array(action_norm_stats["min"]))
            proprio_norm_stats = self.policy.vla.get_proprio_stats(unnorm_key)
            proprio_high, proprio_low = np.array(proprio_norm_stats["max"]), np.array(proprio_norm_stats["min"])
            proprio_high = torch.from_numpy(np.array(proprio_norm_stats["max"]))
            proprio_low = torch.from_numpy(np.array(proprio_norm_stats["min"]))
            action_high = action_high.to(device=proprio.device, dtype=proprio.dtype)
            action_low = action_low.to(device=proprio.device, dtype=proprio.dtype)
            proprio_high = proprio_high.to(device=proprio.device, dtype=proprio.dtype)
            proprio_low = proprio_low.to(device=proprio.device, dtype=proprio.dtype)
            mask = mask.to(device=proprio.device)
            
            proprio = torch.where(
                mask,
                torch.clamp(2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1, -1, 1),
                proprio,
            )
            img = torch.from_numpy(np.array(img))      # (H, W, C), uint8
            img = img.float().div(255.0)                         # [0,1]
            resized_curr_image = img.to("cuda:0",dtype=torch.float)   # (3, 224, 224)
            all_actions = self.policy.step(resized_curr_image, instruction, proprio)
            all_actions = all_actions[0]
            
            for i in range(all_actions.shape[0]):
                all_actions[i] = torch.where(
                    mask,
                    0.5 * (all_actions[i] + 1) * (action_high - action_low) + action_low,
                    all_actions[i],
                )
            all_actions = all_actions.cpu().numpy()
            if double_encode:
                return JSONResponse(json_numpy.dumps(all_actions))
            else:
                return JSONResponse(all_actions)
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'observation': dict, 'instruction': str}\n"
            )
            return "error"

    def run(self, host: str = "0.0.0.0", port: int = 8777) -> None:
        self.app = FastAPI()
        self.app.post("/act")(self.get_server_action)
        uvicorn.run(self.app, host=host, port=port)


@dataclass
class DeployConfig:
    # fmt: off

    # Server Configuration
    host: str = "0.0.0.0"                            # Host IP Address
    port: int = 7998                                 # Host Port

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    model_family: str = "Robobrain-Dex"              # Model family
 
    pretrained_checkpoint: Union[str, Path] = ""     # checkpoint path
    
    use_l1_regression: bool = True                   # If True, uses continuous action head with L1 regression objective
    use_diffusion: bool = False                      # If True, uses continuous action head with diffusion modeling objective (DDIM)
    num_diffusion_steps_train: int = 50              # (When `diffusion==True`) Number of diffusion steps used for training
    num_diffusion_steps_inference: int = 50          # (When `diffusio`n==True`) Number of diffusion steps used for inference
    use_film: bool = False                           # If True, uses FiLM to infuse language inputs into visual features
    num_images_in_input: int = 1                     # Number of images in the VLA input (default: 3)
    use_proprio: bool = True                         # Whether to include proprio state in input

    center_crop: bool = True                         # Center crop? (if trained w/ random crop image aug)

    lora_rank: int = 32                              # Rank of LoRA weight matrix (MAKE SURE THIS MATCHES TRAINING!)

    unnorm_key: Union[str, Path] = "pour_the_drink"  # pour_the_drinks_into_the_glass put_the_cola_into_the_basket         # Action un-normalization key
    use_relative_actions: bool = False               # Whether to use relative actions (delta joint angles)

    load_in_8bit: bool = False                       # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False                       # (For OpenVLA only) Load with 4-bit quantization

    #################################################################################################################
    # Utils
    #################################################################################################################
    seed: int = 7                                    # Random Seed (for reproducibility)
    # fmt: on


@draccus.wrap()
def deploy(cfg: DeployConfig) -> None:
    server = DexVLAServer(cfg)
    server.run(cfg.host, port=cfg.port)


if __name__ == "__main__":
    deploy()
