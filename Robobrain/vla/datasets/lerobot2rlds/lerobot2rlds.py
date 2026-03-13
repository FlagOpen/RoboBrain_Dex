import argparse
import logging
import os
from functools import partial
from pathlib import Path
import torch

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from tensorflow_datasets.core.file_adapters import FileFormat
from tensorflow_datasets.core.utils.lazy_imports_utils import apache_beam as beam
from tensorflow_datasets.rlds import rlds_base
import json

os.environ["NO_GCE_CHECK"] = "true"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
tfds.core.utils.gcs_utils._is_gcs_disabled = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def generate_config_from_features(features, encoding_format, **kwargs):
    action_info = {
        **{
            "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
            )
            for k, v in features.items()
            if "action" in k  # for compatibility with actions.action_key and action
        },
    }
    action_info = action_info if len(action_info) > 1 else action_info.popitem()[1]
    return dict(
        observation_info={
            # **{
            #     k.split(".")[-1]: tfds.features.Image(
            #         shape=v["shape"], dtype=np.uint8, encoding_format=encoding_format, doc=v["names"]
            #     )
            #     for k, v in features.items()
            #     if "observation.image" in k and "depth" not in k
            # },
            **{
                k.split(".")[-1]: tfds.features.Image(
                    shape=v["shape"], dtype=np.uint8, encoding_format=encoding_format, doc=v["names"]
                )
                for k, v in features.items()
                if "image" in k and "depth" not in k
            },
            **{
                k.split(".")[-1]: tfds.features.Tensor(shape=v["shape"][:-1], dtype=np.float32, doc=v["names"])
                for k, v in features.items()
                if "observation.image" in k and "depth" in k
            },
            # **{
            #     "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
            #         shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
            #     )
            #     for k, v in features.items()
            #     if "observation.state" in k  # for compatibility with observation.states.state_key and observation.state
            # },
            **{
                "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                )
                for k, v in features.items()
                if "eef_pos" in k  # for compatibility with observation.states.state_key and observation.state
            },
            **{
                "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                )
                for k, v in features.items()
                if "eef_rot_axis_angle" in k  # for compatibility with observation.states.state_key and observation.state
            },
            **{
                "_".join(k.split(".")[2:]) or k.split(".")[-1]: tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                )
                for k, v in features.items()
                if "gripper_width" in k
                # for compatibility with observation.states.state_key and observation.state
            },
            **{"episode_index": tfds.features.Tensor(
                shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                ) for k, v in features.items() if "episode_index" in k},
            **{"frame_idx":tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                ) for k, v in features.items() if "frame_index" in k},
            **{"time_stamp":tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                ) for k, v in features.items() if "timestamp" in k},
            **{"start_reasoning_index":tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                )for k, v in features.items() if "frame_index" in k},
            **{"end_reasoning_index":tfds.features.Tensor(
                    shape=v["shape"], dtype=np.dtype(v["dtype"]), doc=v["names"]
                )for k, v in features.items() if "frame_index" in k},

        },
        
        action_info=action_info,
        step_metadata_info={
            "language_instruction": tfds.features.Text(),
        },
        citation=kwargs.get("citation", ""),
        homepage=kwargs.get("homepage", ""),
        overall_description=kwargs.get("overall_description", ""),
        description=kwargs.get("description", ""),
    )

def _get_thought(thoughts: list[dict], step: int) -> dict[str, str|None]:
    for thought in thoughts:
        end_step = thought['end_step']
        if end_step == -1:
            end_step = 1e9
        if thought['start_step'] <= step < end_step:
            return thought

    raise ValueError(f"No thought found for step {step}")


def parse_step(data_item,data_reasoning):
    observation_info = {
        # **{
        #     # lerobot image is (C, H, W) and in range [0, 1]
        #     k.split(".")[-1]: np.array(v * 255, dtype=np.uint8).transpose(1, 2, 0)
        #     for k, v in data_item.items()
        #     if "observation.image" in k and "depth" not in k
        # },
        **{
            # lerobot image is (C, H, W) and in range [0, 1]
            k.split(".")[-1]: np.array(v * 255, dtype=np.uint8).transpose(1, 2, 0)
            for k, v in data_item.items()
            if "image" in k and "depth" not in k
        },
        **{
            # lerobot depth is (1, H, W) and in range [0, 1]
            k.split(".")[-1]: v.float().squeeze()
            for k, v in data_item.items()
            if "observation.image" in k and "depth" in k
        },
        **{"_".join(k.split(".")[2:]) or k.split(".")[-1]: v for k, v in data_item.items() if "eef_pos" in k},
        **{"_".join(k.split(".")[2:]) or k.split(".")[-1]: v for k, v in data_item.items() if "eef_rot_axis_angle" in k},
        **{"_".join(k.split(".")[2:]) or k.split(".")[-1]: v.unsqueeze(0) for k, v in data_item.items() if "gripper_width" in k},
        **{"episode_index": v.unsqueeze(0) for k, v in data_item.items() if "episode_index" in k},
        **{"frame_idx":v.unsqueeze(0) for k, v in data_item.items() if "frame_index" in k},
        **{"time_stamp":v.unsqueeze(0) for k, v in data_item.items() if "timestamp" in k},
        # **{"state": torch.cat([data_item["eef_pos"],data_item["eef_rot_axis_angle"],data_item["gripper_width"].unsqueeze(0)],dim=0),},

    }
    action_info = {
        **{"_".join(k.split(".")[2:]) or k.split(".")[-1]: v for k, v in data_item.items() if "action" in k},
    }
    action_info = action_info if len(action_info) > 1 else action_info.popitem()[1]

    ##sample reasoing content
    episode_id=data_item["episode_index"].item()
    # episode_start_interval = data_reasoning[episode_id]['episode_start_interval']
    frame_idx=data_item["index"].item()
    # use relative indexing within the episode to obtain reasoning content
    reasonings=data_reasoning[episode_id]['segments']
    if len(reasonings)==2: #robot data with QA
        reasoning_dict = _get_thought(reasonings, frame_idx)
        possible_content_idx = np.random.RandomState(42).randint(0, len(reasoning_dict['possible_contents']))
        chosen_content = reasoning_dict['possible_contents'][possible_content_idx]
        start_reasoning_index=reasoning_dict["start_step"]
        end_reasoning_index=reasoning_dict["end_step"]
        # ========== first segment ==========
        # where the robot reasons to identify the object to move to
        if chosen_content['updated_content'] is not None:
            language_instruction = chosen_content['updated_content_w_instruction']
        else:
            # ref_interval_start_step, ref_interval_end_step = reference_start_step, reference_end_step
            language_instruction = chosen_content['content']
    else:
        ##synthstic QA data
        reasoning_dict=reasonings
        possible_content_idx = np.random.RandomState(42).randint(0, len(reasoning_dict))
        chosen_content = reasoning_dict[possible_content_idx]
        language_instruction=chosen_content["content"]+chosen_content["updated_content"]
        start_reasoning_index=0
        end_reasoning_index=-1 #hard code

    # return observation_info, action_info, data_item["task"]

    observation_info["start_reasoning_index"]=np.array([start_reasoning_index], dtype=np.int64)
    observation_info["end_reasoning_index"]=np.array([end_reasoning_index], dtype=np.int64)

    return observation_info, action_info, language_instruction


class DatasetBuilder(tfds.core.GeneratorBasedBuilder, skip_registration=True):
    def __init__(self, raw_dir, name, dataset_config, enable_beam, *, file_format=None, **kwargs):
        self.name = name
        self.VERSION = kwargs["version"]
        self.raw_dir = raw_dir
        self.dataset_config = dataset_config
        self.enable_beam = enable_beam
        self.__module__ = "lerobot2rlds"
        super().__init__(file_format=file_format, **kwargs)

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return rlds_base.build_info(
            rlds_base.DatasetConfig(
                name=self.name,
                **self.dataset_config,
            ),
            self,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        dl_manager._download_dir.rmtree(missing_ok=True)
        return {
            "train": self._generate_examples(),
        }

    def _generate_examples(self):
        """Yields examples."""

        def _generate_examples_beam(episode_index, raw_dir):
            episode = []
            dataset = LeRobotDataset("", raw_dir, episodes=[episode_index])
            logging.info(f"processing episode {episode_index}")
            for data_item in dataset:
                observation_info, action_info, language_instruction = parse_step(data_item)
                episode.append(
                    {
                        "observation": observation_info,
                        "action": action_info,
                        "language_instruction": language_instruction,
                        "is_first": data_item["frame_index"].item() == 0,
                        "is_last": data_item["frame_index"].item()
                        == dataset.meta.episodes[episode_index]["length"] - 1,
                        "is_terminal": data_item["frame_index"].item()
                        == dataset.meta.episodes[episode_index]["length"] - 1,
                    }
                )
            return episode_index, {"steps": episode}

        def _generate_examples_regular():
            dataset = LeRobotDataset("", self.raw_dir)
            episode = []
            current_episode_index = 0

            ##add cot
            reasoning_json_path = self.raw_dir.joinpath('cot.json')

            if reasoning_json_path is not None:
                with open(reasoning_json_path, 'r') as f:
                    _loaded = json.load(f)
                    data_reasoning = {}
                    for k, v in _loaded.items():
                        if k.isdigit():
                            data_reasoning[int(k)] = v
                        else:
                            data_reasoning[k] = v

            for data_item in dataset:
                if data_item["episode_index"] != current_episode_index:
                    episode[-1]["is_last"] = True
                    episode[-1]["is_terminal"] = True
                    yield f"{current_episode_index}", {"steps": episode}
                    current_episode_index = data_item["episode_index"]
                    episode.clear()

                observation_info, action_info, language_instruction = parse_step(data_item,data_reasoning)
                episode.append(
                    {
                        "observation": observation_info,
                        "action": action_info,
                        "language_instruction": language_instruction,
                        "is_first": data_item["frame_index"].item() == 0,
                        "is_last": False,
                        "is_terminal": False,
                    }
                )
            episode[-1]["is_last"] = True
            episode[-1]["is_terminal"] = True
            yield f"{current_episode_index}", {"steps": episode}

        if self.enable_beam:
            metadata = LeRobotDatasetMetadata("", self.raw_dir)
            return beam.Create(list(metadata.episodes.keys())) | beam.Map(
                partial(_generate_examples_beam, raw_dir=self.raw_dir)
            )
        else:
            # NOTE: we should return a generator, not yield
            return _generate_examples_regular()


def main(src_dir, output_dir, task_name, version, encoding_format, enable_beam, **kwargs):
    raw_dataset_meta = LeRobotDatasetMetadata("", root=src_dir)
 
    dataset_config = generate_config_from_features(raw_dataset_meta.features, encoding_format, **kwargs)

    dataset_builder = DatasetBuilder(
        raw_dir=src_dir,
        name=task_name,
        data_dir=output_dir,
        version=version,
        dataset_config=dataset_config,
        enable_beam=enable_beam,
        file_format=FileFormat.TFRECORD,
    )

    if enable_beam:
        logging.warning("beam processing is enabled. Some episodes might be lost, a bug with apache beam.")
        logging.warning("disable beam processing if your dataset is small or you want to save all episodes.")
        from apache_beam.options.pipeline_options import PipelineOptions
        from apache_beam.runners import create_runner

        if "threading" in kwargs["beam_run_mode"]:
            logging.warning("multi_threading might have issues when sharding and saving.")
            logging.warning("recommend using multi_processing instead.")

        beam_options = PipelineOptions(
            direct_running_mode=kwargs["beam_run_mode"],
            direct_num_workers=kwargs["beam_num_workers"],
        )
        beam_runner = create_runner("DirectRunner")
    else:
        beam_options = None
        beam_runner = None

    dataset_builder.download_and_prepare(
        download_config=tfds.download.DownloadConfig(
            try_download_gcs=False,
            verify_ssl=False,
            beam_options=beam_options,
            beam_runner=beam_runner,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir", type=Path, help="Path to the local lerobot dataset.")
    parser.add_argument("--output-dir", type=Path, help="Path to the output directory.")
    parser.add_argument("--task-name", type=str, help="Task name.")
    parser.add_argument("--enable-beam", action="store_true", help="Enable beam processing.")
    parser.add_argument("--beam-run-mode", choices=["multi_threading", "multi_processing"], default="multi_processing")
    parser.add_argument("--beam-num-workers", type=int, default=5)
    parser.add_argument("--encoding-format", type=str, choices=["jpeg", "png"], default="jpeg")
    parser.add_argument("--version", type=str, help="x.y.z", default="0.1.0")
    parser.add_argument("--citation", type=str, help="Citation.", default="")
    parser.add_argument("--homepage", type=str, help="Homepage.", default="")
    parser.add_argument("--overall-description", type=str, help="Overall description.", default="")
    parser.add_argument("--description", type=str, help="Description.", default="")
    args = parser.parse_args()

    main(**vars(args))
