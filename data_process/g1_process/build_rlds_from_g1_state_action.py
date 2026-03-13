#!/usr/bin/env python3
"""
Convert episodes stored as
  /<root>/<task_name>/episode_XXXXXX/
      data.json
      colors/
          000000_color_0.jpg, 000001_color_0.jpg, ...
into an RLDS (TFDS) dataset.

This script mirrors the structure of `build_rlds_from_parquet_mp4.py`, but reads
per-frame RGB images from the `colors` folder and per-frame state from
`data.json`.

Key assumptions (tweak if your data differs):
- `data.json` schema:
  {
    "data": [
      {
        "idx": 0,
        "colors": {"color_0": "colors/000000_color_0.jpg", ...},
        "depths": {...},
        "states": {
          "left_arm": {"qpos": [...], "qvel": [...], "torque": [...]},
          "right_arm": {"qpos": [...], "qvel": [...], "torque": [...]}
        }
      }, ...
    ]
  }
- Images are named `<frame:06d>_color_0.jpg` etc. and relative paths in
  `colors` match the files on disk inside the episode dir.
- We build a compact `state` as the concatenation of both arms' `qpos` only.
  (`full_state` uses qpos+qvel+torque for both arms if present.)
- The `action` at step *i* is the next state's vector at *i+1* (behavior cloning style),
  so an episode with N frames yields N-1 steps.

If your state layout changes, adjust `_pack_state()` accordingly.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_datasets.rlds import rlds_base
from tqdm import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R

EP_DIR_RE = re.compile(r"^episode_(\d{4,})$")
FINGER_TIP_INDECES = [12, 13, 14, 27, 28, 29, 42, 43, 44, 57, 58, 59, 72, 73, 74]

# ------------------------- Utilities -------------------------
def euler_apply(points_xyz, euler_xyz):
    rot = R.from_euler('xyz', euler_xyz, degrees=True)
    return rot.apply(points_xyz)

def _read_json(p: Path) -> Dict[str, Any]:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_rgb(image_path: Path, height: Optional[int] = None, width: Optional[int] = None) -> np.ndarray:
    img = Image.open(image_path).convert("RGB")
    if height and width:
        img = img.resize((width, height), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


@dataclass
class FrameRec:
    idx: int
    image_rel: str  # e.g. "colors/000000_color_0.jpg"
    action_left_wrist: np.ndarray
    action_right_wrist: np.ndarray
    action_left_hand: np.ndarray
    action_right_hand: np.ndarray
    state_left_wrist: np.ndarray
    state_right_wrist: np.ndarray
    state_left_hand: np.ndarray
    state_right_hand: np.ndarray


def _as_np(a: Any, dtype=np.float32) -> np.ndarray:
    if a is None:
        return np.zeros((0,), dtype=dtype)
    arr = np.asarray(a, dtype=dtype).reshape(-1)
    return arr

def quat_apply(points_xyz, quat_xyzw):
    rot = R.from_quat(quat_xyzw)
    return rot.apply(points_xyz)

global _R_W_B, _p_B_W, _R_W_C, _p_C_W, Matrix_C_B
_p_B_W = np.asarray([-4.2, -3.7, 0.76], dtype=np.float64).reshape(3)
_R_W_B = R.from_quat(np.asarray([(0, 0, -0.7071, 0.7071)], dtype=np.float64).reshape(4)).as_matrix()
_p_C_W = np.asarray([-4.182464599609375, -3.7537224292755127, 1.2338616847991943], dtype=np.float64).reshape(3)
# R_C_W = R.from_quat(np.asarray([ 0.646973,  0.28535231,  0.646973, -0.28535231], dtype=np.float64).reshape(4)).as_matrix()
R_C_W = R.from_quat(np.asarray([ 0.,    0.93232531, -0.36162068,  0.        ], dtype=np.float64).reshape(4)).as_matrix()
_R_W_C = R_C_W.T  # world->camera

def _wrist_base_to_cam(wrist7_B: np.ndarray) -> np.ndarray:
    """
    Convert [px,py,pz,qx,qy,qz,qw] from BASE frame to CAMERA frame.
    Returns same format in camera frame.
    """
    if wrist7_B is None or wrist7_B.size != 7 or _R_W_B is None:
        raise ValueError(f"[wrist_base_to_cam] {wrist7_B}")
    p_B = wrist7_B[:3].astype(np.float64)
    q_B = wrist7_B[3:7].astype(np.float64)

    # Base -> World
    p_W = _R_W_B @ p_B + _p_B_W
    R_w_B = R.from_quat(q_B).as_matrix()
    R_w_W = _R_W_B @ R_w_B
    # World -> Camera
    p_C = _R_W_C @ (p_W - _p_C_W)
    R_w_C = _R_W_C @ R_w_W
    q_C = R.from_matrix(R_w_C).as_quat()

    return np.concatenate([p_C.astype(np.float32), q_C.astype(np.float32)], dtype=np.float32)

def mat4_to_pq(T) -> np.ndarray:
    """4x4 matrix -> [px,py,pz,qx,qy,qz,qw] (float32)"""
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    t = T[:3, 3]
    q = R.from_matrix(T[:3, :3]).as_quat()  # (x,y,z,w)
    return np.concatenate([t.astype(np.float32), q.astype(np.float32)], dtype=np.float32)

def _parse_data_json(ep_dir: Path, color_key: str = "color_0") -> List[FrameRec]:
    j = _read_json(ep_dir / "data.json")
    data = j.get("data", [])
    frames: List[FrameRec] = []
    for row in data:
        idx = int(row.get("idx", len(frames)))
        colors = row.get("colors", {})
        # choose the requested color key (e.g., color_0)
        image_rel = colors.get(color_key) or next(iter(colors.values()))
        states = row.get("states", {})
        actions = row.get("actions", {})
        state_wrist_L = states.get("left_hand_pos_state", {})
        state_wrist_R = states.get("right_hand_pos_state", {})
        state_hand_L = _as_np(states.get("left_fingertip_pos_state", {}).get("xyz")).reshape(5, 3)
        state_hand_R = _as_np(states.get("right_fingertip_pos_state", {}).get("xyz")).reshape(5, 3)
        action_wrist_L = actions.get("left_hand_pos_action", {})
        action_wrist_R = actions.get("right_hand_pos_action", {})
        action_hand_L = _as_np(actions.get("left_fingertip_pos_action", {}).get("xyz")).reshape(5, 3)
        action_hand_R = _as_np(actions.get("right_fingertip_pos_action", {}).get("xyz")).reshape(5, 3)
        # hand_L = _as_np(states.get("left_hand_state_pose_75", {}).get("qpos"))[FINGER_TIP_INDECES].reshape(5, 3)
        # hand_R = _as_np(states.get("right_hand_state_pose_75", {}).get("qpos"))[FINGER_TIP_INDECES].reshape(5, 3)
        action_left_hand = np.empty_like(action_hand_L)
        action_right_hand = np.empty_like(action_hand_R)
        state_left_hand = np.empty_like(state_hand_L)
        state_right_hand = np.empty_like(state_hand_R)
        # Trans_L = np.array([0, 0, 0.7071, 0.7071])
        # Trans_R = np.array([-0.7071, 0.7071, 0, 0])
        Trans_L = np.array([0.7071, 0, 0.7071, 0])
        Trans_R = np.array([0, 0.7071, 0, 0.7071])
        for i in range(5):
            # left hand
            action_lw = quat_apply(action_hand_L[i], Trans_L)
            action_left_hand[i] = action_lw
            action_rw = quat_apply(action_hand_R[i], Trans_R)
            action_right_hand[i] = action_rw
            state_lw = quat_apply(state_hand_L[i], Trans_L)
            state_left_hand[i] = state_lw
            state_rw = quat_apply(state_hand_R[i], Trans_R)
            state_right_hand[i] = state_rw
            action_left_hand[i] = euler_apply(action_left_hand[i], [0, 0, 0])
            action_right_hand[i] = euler_apply(action_right_hand[i], [0, 0, 180])
            state_left_hand[i] = euler_apply(state_left_hand[i], [0, 0, 0])
            state_right_hand[i] = euler_apply(state_right_hand[i], [0, 0, 180])
        action_left_wrist_B  = mat4_to_pq(_as_np(action_wrist_L.get("pose")))
        action_right_wrist_B = mat4_to_pq(_as_np(action_wrist_R.get("pose")))
        state_left_wrist_B  = mat4_to_pq(_as_np(state_wrist_L.get("pose")))
        state_right_wrist_B = mat4_to_pq(_as_np(state_wrist_R.get("pose")))
        action_left_wrist_C  = _wrist_base_to_cam(action_left_wrist_B)
        action_right_wrist_C = _wrist_base_to_cam(action_right_wrist_B)
        state_left_wrist_C  = _wrist_base_to_cam(state_left_wrist_B)
        state_right_wrist_C = _wrist_base_to_cam(state_right_wrist_B)
        action_left_wrist_matrix = R.from_quat(action_left_wrist_C[3:]).as_matrix() @ R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
        action_left_wrist_C[3:] = R.from_matrix(action_left_wrist_matrix).as_quat()
        action_right_wrist_matrix = R.from_quat(action_right_wrist_C[3:]).as_matrix() @ R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix() 
        action_right_wrist_C[3:] =R.from_matrix(action_right_wrist_matrix).as_quat()
        state_left_wrist_matrix = R.from_quat(state_left_wrist_C[3:]).as_matrix() @ R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
        state_left_wrist_C[3:] = R.from_matrix(state_left_wrist_matrix).as_quat()
        state_right_wrist_matrix = R.from_quat(state_right_wrist_C[3:]).as_matrix() @ R.from_euler('xyz', [0, 0, 180], degrees=True).as_matrix()
        state_right_wrist_C[3:] = R.from_matrix(state_right_wrist_matrix).as_quat()

        frames.append(
            FrameRec(
                idx=idx,
                image_rel=image_rel,
                action_left_wrist=action_left_wrist_C,
                action_right_wrist=action_right_wrist_C,
                action_left_hand=action_left_hand.reshape(-1),
                action_right_hand=action_right_hand.reshape(-1),
                state_left_wrist=state_left_wrist_C,
                state_right_wrist=state_right_wrist_C,
                state_left_hand=state_left_hand.reshape(-1),
                state_right_hand=state_right_hand.reshape(-1)
            )
        )
    # sort by idx to ensure proper order
    frames.sort(key=lambda r: r.idx)
    return frames


def _pack_state(fr: FrameRec) -> Tuple[np.ndarray, np.ndarray]:
    """Return (state, full_state).

    - state: concatenation of left/right qpos
    - full_state: concatenation of left/right (qpos, qvel, torque)
    """
    state = np.concatenate([fr.state_left_wrist, fr.state_right_wrist, fr.state_left_hand, fr.state_right_hand], dtype=np.float32)
    full_state = np.concatenate(
        [fr.state_left_wrist, fr.state_right_wrist, fr.state_left_hand, fr.state_right_hand],
        dtype=np.float32,
    )
    return state, full_state

def _pack_action(fr: FrameRec) -> Tuple[np.ndarray, np.ndarray]:
    """Return (action, full_action).

    - action: concatenation of left/right qpos
    - full_action: concatenation of left/right (qpos, qvel, torque)
    """
    action = np.concatenate([fr.action_left_wrist, fr.action_right_wrist, fr.action_left_hand, fr.action_right_hand], dtype=np.float32)
    full_action = np.concatenate(
        [fr.action_left_wrist, fr.action_right_wrist, fr.action_left_hand, fr.action_right_hand],
        dtype=np.float32,
    )
    return action, full_action


# ------------------------- TFDS Builder -------------------------

class ImagesJSONRLDS(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, input_root: str, name: str, camera_key: str, height: Optional[int], width: Optional[int], **kwargs):
        self.name = name
        self.input_dir = Path(input_root).joinpath(name)
        self.camera_key = camera_key
        self.height = height
        self.width = width
        super().__init__(**kwargs)

    def _info(self):
        obs = {
            "frame_idx": tfds.features.Tensor(shape=(1,), dtype=np.int32),
            "image": tfds.features.Image(shape=(self.height, self.width, 3), dtype=np.uint8, encoding_format="jpeg"),
            "state": tfds.features.Tensor(shape=(44,), dtype=np.float32),
            "full_state": tfds.features.Tensor(shape=(44,), dtype=np.float32),
            "action_label": tfds.features.Text(),
            "latest_reasoning": tfds.features.Text(),
        }
        return rlds_base.build_info(
            rlds_base.DatasetConfig(
                name=self.name,
                observation_info=obs,
                action_info=tfds.features.Tensor(shape=(44,), dtype=np.float32),
                step_metadata_info={
                    "language_instruction": tfds.features.Text(),
                    "episode_idx": tfds.features.Scalar(dtype=tf.int64),
                },
            ),
            self,
        )
    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples()}

    def _iter_episode_dirs(self) -> List[Tuple[Path, int]]:
        eps: List[Tuple[Path, int]] = []
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Input dir not found: {self.input_dir}")
        for p in sorted(self.input_dir.iterdir()):
            if not p.is_dir():
                continue
            m = EP_DIR_RE.match(p.name)
            if not m:
                continue
            eps.append((p, int(m.group(1))))
        eps.sort(key=lambda x: x[1])
        return eps

    def _generate_examples(self):
        language_instruction = self.name.replace('_', ' ')
        for ep_global_idx, (ep_dir, ep_num) in enumerate(self._iter_episode_dirs()):
            frames = _parse_data_json(ep_dir, color_key=self.camera_key)
            if len(frames) < 2:
                # Need at least two frames to form (s, a=next_s)
                continue

            steps: List[Dict[str, Any]] = []
            for i in range(len(frames) - 1):
                if i % 2 == 0:
                    fr = frames[i]

                    img_path = ep_dir / fr.image_rel
                    if not img_path.exists():
                        # fallback: try to construct from idx
                        fallback = ep_dir / "colors" / f"{fr.idx:06d}_{self.camera_key}.jpg"
                        img_path = fallback if fallback.exists() else img_path
                    image = _load_rgb(img_path, self.height, self.width)

                    s, s_full = _pack_state(fr)
                    s_next, _ = _pack_action(fr)

                    steps.append(
                        {
                            "observation": {
                                "frame_idx": np.array([fr.idx], dtype=np.int32),
                                "image": image,  # HxWx3 uint8; TFDS encodes as JPEG
                                "state": s.astype(np.float32),
                                "full_state": s_full.astype(np.float32),
                                "action_label": language_instruction,
                                "latest_reasoning": "",
                            },
                            "action": s_next.astype(np.float32),
                            "is_first": (i == 0),
                            "is_last": (i == len(frames) - 2),
                            "is_terminal": (i == len(frames) - 2),
                            "language_instruction": language_instruction,
                            "episode_idx": np.int64(ep_num),
                        }
                    )

            # Yield one RLDS example per episode (as a list of steps)
            yield str(ep_global_idx), {"steps": steps}


# ------------------------- Optional CoT export (compatible with your helper) -------------------------

def build_cot_from_rlds_steps(
    steps,
    data_path: str,
) -> Dict[str, Any]:
    """
    Record reasoning only at "background" or the first frame to form boundary segments:
    each segment has start_frame = boundary frame;
    end_frame = the frame before the next boundary (last segment uses the last frame).
    """
    recs: List[Dict[str, Any]] = []
    for step in steps:
        task_instruction = str(step.get("language_instruction", "").numpy().decode("utf-8")).strip()
        obs = step["observation"]
        fi = obs["frame_idx"].numpy().item()
        action_label = str(obs["action_label"].numpy().decode("utf-8")).strip()
        lr = str(obs["latest_reasoning"].numpy().decode("utf-8")).strip()

        # Boundary condition: first frame or action_label == "background"
        # If language_instruction is empty, fallback to latest_reasoning
        desc = lr

        recs.append({
            "frame_idx": fi,
            "action_label": action_label,
            "desc": desc
        })
    scene_key = step['episode_idx'].numpy().item()
    recs.sort(key=lambda x: x["frame_idx"])
    if not recs:
        return {
            scene_key: {
                "data_path": data_path,
                "task raw_instruction": task_instruction,
                "reasoning_segments": [],
            }
        }

    # Mark boundaries: first frame + all "background" frames (dedupe adjacent same desc)
    boundaries: List[Dict[str, Any]] = []
    last_desc = None
    for idx, r in enumerate(recs):
        fi = r["frame_idx"]
        al = (r["action_label"] or "").strip().lower()
        is_first = (idx == 0)
        is_last = (idx == len(recs)-1)
        is_bg = (al == "background")
        if is_first or is_bg:
            desc = (r["desc"] or "").strip()
            boundaries.append({"frame_idx": fi, "desc": desc})

        if is_last and not is_bg:
            desc = (r["desc"] or "").strip()
            boundaries.append({"frame_idx": fi, "desc": desc})
        elif is_last and is_bg:
            desc = ("").strip()
            boundaries.append({"frame_idx": fi, "desc": desc})
    # Build segments: start = current boundary frame; end = frame before next boundary; last segment's end = last frame
    last_frame_idx = recs[-1]["frame_idx"]
    segments: List[Dict[str, Any]] = []
    start_f = boundaries[0]["frame_idx"]
    desc = boundaries[0]["desc"]
    for i, b in enumerate(boundaries):
        end_f = boundaries[i]["frame_idx"]
        if(i < len(boundaries) - 1 and desc != boundaries[i+1]["desc"]):
            desc = boundaries[i+1]["desc"]
            segments.append({
                "start_frame": int(start_f),
                "end_frame": int(end_f),
                "reasoning": desc,
            })
            start_f = boundaries[i+1]["frame_idx"]

    return {
        scene_key: {
            "data_path": data_path,
            "task raw_instruction": task_instruction,
            "reasoning_segments": segments,
        }
    }


# ------------------------- CLI -------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert data.json + images dataset to RLDS (TFDS)")
    parser.add_argument('--input_root', type=str, required=True, help='Path to the dataset root that contains <task_name>/episode_xxxxxx/')
    parser.add_argument('--task_name', type=str, required=True, help='Task name; also used as language instruction and TFDS dataset name')
    parser.add_argument('--output_dir', type=str, required=True, help='Where to write TFDS files')
    parser.add_argument('--camera_key', type=str, default='color_0', help='Which color stream to use: color_0 | color_1 | color_2')
    parser.add_argument('--height', type=int, default=320, help='Optional resize height (0 = keep original)')
    parser.add_argument('--width', type=int, default=240, help='Optional resize width (0 = keep original)')
    args = parser.parse_args()

    height = args.height if args.height > 0 else None
    width = args.width if args.width > 0 else None

    builder = ImagesJSONRLDS(
        input_root=args.input_root,
        name=args.task_name,
        camera_key=args.camera_key,
        height=height,
        width=width,
        data_dir=args.output_dir,
    )

    builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            try_download_gcs=False,
            verify_ssl=False,
            beam_options=None,
            beam_runner=None,
        )
    )

    # Optionally: iterate the dataset and produce a lightweight reasoning file
    dataset = tfds.load(name=args.task_name, split='train', shuffle_files=False, data_dir=args.output_dir)
    all_cot: Dict[str, Any] = {}
    out_dir = Path(args.output_dir) / args.task_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for episode in tqdm(dataset, desc="Iterating episodes"):
        steps = episode['steps']
        cot = build_cot_from_rlds_steps(steps, data_path=str(out_dir))
        all_cot.update(cot)
    with (out_dir / "reasoning.json").open("w", encoding="utf-8") as f:
        json.dump(all_cot, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
