"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any, Dict
import numpy as np
from scipy.spatial.transform import Rotation as R

import tensorflow as tf

from Robobrain.vla.datasets.rlds.oxe.utils.droid_utils import droid_baseact_transform, droid_finetuning_transform
from Robobrain.vla.datasets.rlds.utils.data_utils import (
    binarize_gripper_actions,
    invert_gripper_actions,
    rel2abs_gripper_actions,
    relabel_bridge_actions,
)

@tf.function
def quaternion_to_6d_tf(quat_xyzw: tf.Tensor) -> tf.Tensor:
   
    quat_xyzw = tf.cast(quat_xyzw, tf.float32)
    quat_xyzw = tf.math.l2_normalize(quat_xyzw, axis=-1)
    x, y, z, w = tf.unstack(quat_xyzw, axis=-1)
    two = tf.constant(2.0, tf.float32)


    r00 = 1.0 - two*(y*y + z*z)
    r01 = two*(x*y - z*w)
    r02 = two*(x*z + y*w)

    r10 = two*(x*y + z*w)
    r11 = 1.0 - two*(x*x + z*z)
    r12 = two*(y*z - x*w)

    rot6d = tf.stack([r00, r01, r02, r10, r11, r12], axis=-1)
    rot6d.set_shape([None, 6])  
    return rot6d


def bridge_oxe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key in ["observation", "action"]:
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]

    return trajectory


def bridge_orig_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory.keys():
        if key == "traj_metadata":
            continue
        elif key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    # print(trajectory.keys(), trajectory['observation'].keys())
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def ppgm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def rt1_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def kuka_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.reshape(eef_value, (-1, 7))
    gripper_value = tf.io.decode_compressed(trajectory["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1, 1))
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def taco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["robot_obs"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["robot_obs"][:, 7:8]
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def jaco_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state_eef"] = trajectory["observation"]["end_effector_cartesian_pos"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["end_effector_cartesian_pos"][:, -1:]

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_cable_routing_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def roboturk_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def viola_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def toto_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def language_table_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][..., 0]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., -1:]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def nyu_franka_play_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    return trajectory


def maniskill_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )
    return trajectory


def cmu_franka_exploration_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_sirius_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def bc_z_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def tokyo_pr2_opening_fridge_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def kaist_nonprehensible_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["end_effector_pose"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )
    return trajectory


def asu_table_top_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 7:8]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def berkeley_fanuc_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:7]

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["gripper_state"]),
        ),
        axis=-1,
    )
    return trajectory


def cmu_playing_with_food_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def gnm_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"],
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def dobbe_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]
    return trajectory


def roboset_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def libero_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    return trajectory

def lerobot_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    #copy from libero_dataset_transform
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )

    trajectory["observation"]["EEF_state"] = tf.concat([trajectory["observation"]["eef_pos"],
                                                        trajectory["observation"]["eef_rot_axis_angle"]],
                                                       axis=1)
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_width"]
    trajectory["observation"]["proprio"] = tf.concat([trajectory["observation"]["EEF_state"],
                                                        trajectory["observation"]["gripper_state"]],
                                                       axis=1)


    return trajectory

def human_dataset_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms human data into the expected format by adding dummy actions.
    
    Args:
        sample (Dict[str, Any]): A dictionary containing human data observations.
        
    Returns:
        Dict[str, Any]: Transformed sample with dummy actions added.
    """
    # Extract the observation from the sample

    observation = sample["observation"]
    if "state" in observation and observation["state"].shape[1] == 44:
        state = observation["state"]
        left_arm_7d = state[:, 0:7]
        right_arm_7d = state[:, 7:14]
        left_hand = state[:, 14:29]
        right_hand = state[:, 29:44]

        # left arm 4d quat -> 6d rotation
        left_arm_trans = left_arm_7d[:, 0:3]
        left_arm_rot_6d = quaternion_to_6d_tf(left_arm_7d[:, 3:7])
        # right arm 4d quat -> 6d rotation
        right_arm_trans = right_arm_7d[:, 0:3]
        right_arm_rot_6d = quaternion_to_6d_tf(right_arm_7d[:, 3:7])
        new_state = tf.concat([left_arm_trans, left_arm_rot_6d, right_arm_trans, right_arm_rot_6d, left_hand, right_hand], axis=-1)
        sample["observation"]["state"] = new_state
    
    if "action" in sample and sample["action"].shape[1] == 44:
        action = sample["action"]
        left_arm_7d = action[:, 0:7]
        right_arm_7d = action[:, 7:14]
        left_hand = action[:, 14:29]
        right_hand = action[:, 29:44]

        # left arm 4d quat -> 6d rotation
        left_arm_trans = left_arm_7d[:, 0:3]
        left_arm_rot_6d = quaternion_to_6d_tf(left_arm_7d[:, 3:7])
        # right arm 4d quat -> 6d rotation
        right_arm_trans = right_arm_7d[:, 0:3]
        right_arm_rot_6d = quaternion_to_6d_tf(right_arm_7d[:, 3:7])
        new_action = tf.concat([left_arm_trans, left_arm_rot_6d, right_arm_trans, right_arm_rot_6d, left_hand, right_hand], axis=-1)
        sample["action"] = new_action
    
    return sample

def G1_dataset_transform(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transforms human data into the expected format by adding dummy actions.
    
    Args:
        sample (Dict[str, Any]): A dictionary containing human data observations.
        
    Returns:
        Dict[str, Any]: Transformed sample with dummy actions added.
    """
    # Extract the observation from the sample

    observation = sample["observation"]
    if "state" in observation and observation["state"].shape[1] == 70:
        state = observation["state"]
        left_arm_7d = state[:, 0:7]
        right_arm_7d = state[:, 7:14]
        left_hand = state[:, 14:29]
        right_hand = state[:, 29:44]

        # left arm 4d quat -> 6d rotation
        left_arm_trans = left_arm_7d[:, 0:3]
        left_arm_rot_6d = quaternion_to_6d_tf(left_arm_7d[:, 3:7])
        # right arm 4d quat -> 6d rotation
        right_arm_trans = right_arm_7d[:, 0:3]
        right_arm_rot_6d = quaternion_to_6d_tf(right_arm_7d[:, 3:7])
        state_joint = state[:, 44:70]
        new_state = tf.concat([left_arm_trans, left_arm_rot_6d, right_arm_trans, right_arm_rot_6d, left_hand, right_hand, state_joint], axis=-1)
        sample["observation"]["state"] = new_state
    
    if "action" in sample and sample["action"].shape[1] == 70:
        action = sample["action"]
        left_arm_7d = action[:, 0:7]
        right_arm_7d = action[:, 7:14]
        left_hand = action[:, 14:29]
        right_hand = action[:, 29:44]

        # left arm 4d quat -> 6d rotation
        left_arm_trans = left_arm_7d[:, 0:3]
        left_arm_rot_6d = quaternion_to_6d_tf(left_arm_7d[:, 3:7])
        # right arm 4d quat -> 6d rotation
        right_arm_trans = right_arm_7d[:, 0:3]
        right_arm_rot_6d = quaternion_to_6d_tf(right_arm_7d[:, 3:7])
        action_joint = action[:, 44:70]
        new_action = tf.concat([left_arm_trans, left_arm_rot_6d, right_arm_trans, right_arm_rot_6d, left_hand, right_hand, action_joint], axis=-1)
        sample["action"] = new_action
    
    return sample

# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    # Downstream task
    "pour_the_drink": human_dataset_transform,
    # Pretrain task
    "h2o": human_dataset_transform,
    "ph2d": human_dataset_transform,
    "egodex_part1": human_dataset_transform,
    "egodex_part2": human_dataset_transform,
    "egodex_part3": human_dataset_transform,
    "egodex_part4": human_dataset_transform,
    "egodex_part5": human_dataset_transform,
    "oakink": human_dataset_transform,
    "arctic": human_dataset_transform,
    "fourier": human_dataset_transform,
    "fourier_pc": human_dataset_transform,
    "fourier_all": human_dataset_transform,
    "hrdverse": human_dataset_transform,
    # H2O
    "h2o_subject1_h1": human_dataset_transform,
    "h2o_subject1_h2": human_dataset_transform,
    "h2o_subject1_k1": human_dataset_transform,
    "h2o_subject1_k2": human_dataset_transform,
    "h2o_subject1_o1": human_dataset_transform,
    "h2o_subject1_o2": human_dataset_transform,
    "h2o_subject2_h1": human_dataset_transform,
    "h2o_subject2_h2": human_dataset_transform,
    "h2o_subject2_k1": human_dataset_transform,
    "h2o_subject2_k2": human_dataset_transform,
    "h2o_subject2_o1": human_dataset_transform,
    "h2o_subject2_o2": human_dataset_transform,
    "h2o_subject3_h1": human_dataset_transform,
    "h2o_subject3_h2": human_dataset_transform,
    "h2o_subject3_k1": human_dataset_transform,
    "h2o_subject3_k2": human_dataset_transform,
    "h2o_subject3_o1": human_dataset_transform,
    "h2o_subject3_o2": human_dataset_transform,
    # PH2D
    "ph2d_grasp": human_dataset_transform,
    "ph2d_grasp_chocolate": human_dataset_transform,
    "ph2d_grasp_coke_random": human_dataset_transform,
    "ph2d_grasp_lars": human_dataset_transform,
    "ph2d_grasp_water": human_dataset_transform,
    "ph2d_grasp_pepsi": human_dataset_transform,
    "ph2d_grasp_zedbox": human_dataset_transform,
    "ph2d_grasping_mtn_bottle": human_dataset_transform,
    "ph2d_grasping_three_items": human_dataset_transform,
    "ph2d_pick": human_dataset_transform,
    "ph2d_pick_blackcube": human_dataset_transform,
    "ph2d_pick_brownbox": human_dataset_transform,
    "ph2d_pick_color_pad_left": human_dataset_transform,
    "ph2d_pick_dynamixel": human_dataset_transform,
    "ph2d_pick_on_color_pad_left": human_dataset_transform,
    "ph2d_pick_on_color_pad_right": human_dataset_transform,
    "ph2d_pick_on_color_pad_right_far": human_dataset_transform,
    "ph2d_pick_on_color_pad_right_far_far": human_dataset_transform,
    "ph2d_pick_orange": human_dataset_transform,
    "ph2d_picking_three_cat_straw": human_dataset_transform,
    "ph2d_pour_kura": human_dataset_transform,
    "ph2d_pour": human_dataset_transform,
    "ph2d_pour_mtn": human_dataset_transform,
    "ph2d_pour_party_cup": human_dataset_transform,
    "ph2d_pour_random": human_dataset_transform,
    "ph2d_pour_tea": human_dataset_transform,
    "ph2d_pour_water": human_dataset_transform,
    "ph2d_pouring_costco_water": human_dataset_transform,
    # Egodex
    "egodex_part1_add_remove_lid": human_dataset_transform,
    "egodex_part1_arrange_topple_dominoes": human_dataset_transform,
    "egodex_part1_assemble_disassemble_legos": human_dataset_transform,
    "egodex_part1_assemble_disassemble_soft_legos": human_dataset_transform,
    "egodex_part1_assemble_disassemble_structures": human_dataset_transform,
    "egodex_part1_assemble_disassemble_tiles": human_dataset_transform,
    "egodex_part1_assemble_jenga": human_dataset_transform,
    "egodex_part1_boil_serve_egg": human_dataset_transform,
    "egodex_part1_braid_unbraid": human_dataset_transform,
    "egodex_part1_build_unstack_lego": human_dataset_transform,
    "egodex_part1_charge_uncharge_airpods": human_dataset_transform,
    "egodex_part1_charge_uncharge_device": human_dataset_transform,
    "egodex_part1_clean_cups": human_dataset_transform,
    "egodex_part1_clean_surface": human_dataset_transform,
    "egodex_part1_clean_tableware": human_dataset_transform,
    "egodex_part1_clip_unclip_papers": human_dataset_transform,
    "egodex_part1_color": human_dataset_transform,
    "egodex_part1_crumple_flatten_paper": human_dataset_transform,
    "egodex_part1_deal_gather_cards": human_dataset_transform,
    "egodex_part1_declutter_desk": human_dataset_transform,
    "egodex_part1_dry_hands": human_dataset_transform,
    "egodex_part1_fidget_magnetic_spinner_rings": human_dataset_transform,
    "egodex_part1_flip_coin": human_dataset_transform,
    "egodex_part1_flip_pages": human_dataset_transform,
    "egodex_part1_fry_bread": human_dataset_transform,
    "egodex_part1_fry_egg": human_dataset_transform,
    "egodex_part2_assemble_disassemble_furniture_bench_chair": human_dataset_transform,
    "egodex_part2_assemble_disassemble_furniture_bench_desk": human_dataset_transform,
    "egodex_part2_assemble_disassemble_furniture_bench_drawer": human_dataset_transform,
    "egodex_part2_assemble_disassemble_furniture_bench_lamp": human_dataset_transform,
    "egodex_part2_assemble_disassemble_furniture_bench_square_table": human_dataset_transform,
    "egodex_part2_assemble_disassemble_furniture_bench_stool": human_dataset_transform,
    "egodex_part2_basic_fold": human_dataset_transform,
    "egodex_part2_basic_pick_place": human_dataset_transform,
    "egodex_part2_fold_stack_unstack_unfold_cloths": human_dataset_transform,
    "egodex_part2_fold_unfold_paper_basic": human_dataset_transform,
    "egodex_part2_fold_unfold_paper_origami": human_dataset_transform,
    "egodex_part2_insert_remove_furniture_bench_cabinet": human_dataset_transform,
    "egodex_part2_insert_remove_furniture_bench_round_table": human_dataset_transform,
    "egodex_part3_gather_roll_dice": human_dataset_transform,
    "egodex_part3_insert_dump_blocks": human_dataset_transform,
    "egodex_part3_insert_remove_airpods": human_dataset_transform,
    "egodex_part3_insert_remove_bagging": human_dataset_transform,
    "egodex_part3_insert_remove_bookshelf": human_dataset_transform,
    "egodex_part3_insert_remove_cups_from_rack": human_dataset_transform,
    "egodex_part3_insert_remove_drawer": human_dataset_transform,
    "egodex_part3_insert_remove_plug_socket": human_dataset_transform,
    "egodex_part3_insert_remove_shirt_in_tube": human_dataset_transform,
    "egodex_part3_insert_remove_tennis_ball": human_dataset_transform,
    "egodex_part3_insert_remove_usb": human_dataset_transform,
    "egodex_part3_insert_remove_utensils": human_dataset_transform,
    "egodex_part3_knead_slime": human_dataset_transform,
    "egodex_part3_load_dispense_ice": human_dataset_transform,
    "egodex_part3_lock_unlock_key": human_dataset_transform,
    "egodex_part3_make_sandwich": human_dataset_transform,
    "egodex_part3_measure_objects": human_dataset_transform,
    "egodex_part3_open_close_insert_remove_box": human_dataset_transform,
    "egodex_part3_open_close_insert_remove_case": human_dataset_transform,
    "egodex_part3_open_close_insert_remove_tupperware": human_dataset_transform,
    "egodex_part3_paint_clean_brush": human_dataset_transform,
    "egodex_part3_peel_place_sticker": human_dataset_transform,
    "egodex_part3_pick_up_and_put_down_case_or_bag": human_dataset_transform,
    "egodex_part3_play_piano": human_dataset_transform,
    "egodex_part4_pick_place_food": human_dataset_transform,
    "egodex_part4_play_mancala": human_dataset_transform,
    "egodex_part4_play_reset_connect_four": human_dataset_transform,
    "egodex_part4_point_and_click_remote": human_dataset_transform,
    "egodex_part4_pour": human_dataset_transform,
    "egodex_part4_push_pop_toy": human_dataset_transform,
    "egodex_part4_put_away_set_up_board_game": human_dataset_transform,
    "egodex_part4_put_in_take_out_glasses": human_dataset_transform,
    "egodex_part4_put_toothpaste_on_toothbrush": human_dataset_transform,
    "egodex_part4_rake_smooth_zen_garden": human_dataset_transform,
    "egodex_part4_roll_ball": human_dataset_transform,
    "egodex_part4_scoop_dump_ice": human_dataset_transform,
    "egodex_part4_screw_unscrew_allen_fixture": human_dataset_transform,
    "egodex_part4_screw_unscrew_bottle_cap": human_dataset_transform,
    "egodex_part4_screw_unscrew_fingers_fixture": human_dataset_transform,
    "egodex_part4_set_up_clean_up_chessboard": human_dataset_transform,
    "egodex_part4_sleeve_unsleeve_cards": human_dataset_transform,
    "egodex_part4_slot_batteries": human_dataset_transform,
    "egodex_part4_sort_beads": human_dataset_transform,
    "egodex_part4_staple_paper": human_dataset_transform,
    "egodex_part4_stock_unstock_fridge": human_dataset_transform,
    "egodex_part4_sweep_dustpan": human_dataset_transform,
    "egodex_part5_stack": human_dataset_transform,
    "egodex_part5_stack_remove_jenga": human_dataset_transform,
    "egodex_part5_stack_unstack_bowls": human_dataset_transform,
    "egodex_part5_stack_unstack_cups": human_dataset_transform,
    "egodex_part5_stack_unstack_plates": human_dataset_transform,
    "egodex_part5_stack_unstack_tupperware": human_dataset_transform,
    "egodex_part5_thread_unthread_bead_necklace": human_dataset_transform,
    "egodex_part5_throw_and_catch_ball": human_dataset_transform,
    "egodex_part5_throw_collect_objects": human_dataset_transform,
    "egodex_part5_tie_and_untie_shoelace": human_dataset_transform,
    "egodex_part5_tie_untie_rubberband": human_dataset_transform,
    "egodex_part5_type_keyboard": human_dataset_transform,
    "egodex_part5_use_chopsticks": human_dataset_transform,
    "egodex_part5_use_rubiks_cube": human_dataset_transform,
    "egodex_part5_vertical_pick_place": human_dataset_transform,
    "egodex_part5_wash_fruit": human_dataset_transform,
    "egodex_part5_wash_kitchen_dishes": human_dataset_transform,
    "egodex_part5_wash_put_away_dishes": human_dataset_transform,
    "egodex_part5_wipe_kitchen_surfaces": human_dataset_transform,
    "egodex_part5_wipe_screen": human_dataset_transform,
    "egodex_part5_wrap": human_dataset_transform,
    "egodex_part5_wrap_unwrap_food": human_dataset_transform,
    "egodex_part5_write": human_dataset_transform,
    "egodex_part5_zip_unzip_bag": human_dataset_transform,
    "egodex_part5_zip_unzip_case": human_dataset_transform,

    "bridge_oxe": bridge_oxe_dataset_transform,
    "transfored_rlds/wild_move_to":lerobot_dataset_transform, ##add lerobot vl data
    "wild_move_to_6932":lerobot_dataset_transform, # add
    "grab_the_apple_and_put_it_into_the_basket":human_dataset_transform, #add
    "pick_up_the_cup_and_pour_it_into_the_container":human_dataset_transform, #add
    "pick_up_the_cup_and_place_it_in_the_basket":human_dataset_transform, #add
    "pick_up_the_croissant_on_the_table_and_put_it_in_the_container":human_dataset_transform, #add
    "pick_up_the_croissant_from_the_table_and_place_it_in_the_container":human_dataset_transform, #add
    "pick_up_the_mouse_from_the_table_and_place_it_in_the_container":human_dataset_transform, #add
    "pick_up_the_mouse_and_put_it_in_the_white_basket":human_dataset_transform, #add
    "put_the_yellow_tape_measure_into_the_white_plastic_container":human_dataset_transform, #add
    "put_the_yellow_tape_measure_in_the_white_basket":human_dataset_transform, #add
    "put_the_yellow_tape_measure_in_the_black_basket":human_dataset_transform, #add
    "put_the_purple_eggplant_in_the_white_pan":human_dataset_transform, #add
    "put_the_purple_eggplant_in_the_white_basket":human_dataset_transform, #add
    "put_the_pink_cup_in_the_pan":human_dataset_transform, #add
    "put_the_pink_cup_in_the_basket":human_dataset_transform, #add
    "put_the_lettuce_in_the_container":human_dataset_transform, #add
    "put_the_cup_in_the_container":human_dataset_transform, #add
    "add_remove_lid":human_dataset_transform, #add
    "arrange_topple_dominoes":human_dataset_transform, #add
    "assemble_disassemble_legos":human_dataset_transform, #add
    "assemble_disassemble_soft_legos":human_dataset_transform, #add
    "assemble_disassemble_structures":human_dataset_transform, #add
    "assemble_disassemble_tiles":human_dataset_transform, #add
    "assemble_jenga":human_dataset_transform, #add
    "boil_serve_egg":human_dataset_transform, #add
    "braid_unbraid":human_dataset_transform, #add
    "build_unstack_lego":human_dataset_transform, #add
    "charge_uncharge_airpods":human_dataset_transform, #add
    "charge_uncharge_device":human_dataset_transform, #add
    "clean_cups":human_dataset_transform, #add
    "clean_surface":human_dataset_transform, #add
    "clean_tableware":human_dataset_transform, #add
    "clip_unclip_papers":human_dataset_transform, #add
    "color":human_dataset_transform, #add
    "crumple_flatten_paper":human_dataset_transform, #add
    "deal_gather_cards":human_dataset_transform, #add
    "declutter_desk":human_dataset_transform, #add
    "dry_hands":human_dataset_transform, #add
    "fidget_magnetic_spinner_rings":human_dataset_transform, #add
    "flip_coin":human_dataset_transform, #add
    "flip_pages":human_dataset_transform, #add
    "fry_bread":human_dataset_transform, #add
    "fry_egg":human_dataset_transform, #add
    "wash_kitchen_dishes":human_dataset_transform, #add
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    "ppgm": ppgm_dataset_transform,
    "ppgm_static": ppgm_dataset_transform,
    "ppgm_wrist": ppgm_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
    "droid": droid_baseact_transform,
    "fmb": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": tdroid_dataset_transform,
    "tdroid_pour_corn_in_pot": tdroid_dataset_transform,
    "tdroid_flip_pot_upright": tdroid_dataset_transform,
    "tdroid_move_object_onto_plate": tdroid_dataset_transform,
    "tdroid_knock_object_over": tdroid_dataset_transform,
    "tdroid_cover_object_with_towel": tdroid_dataset_transform,
    ### DROID Finetuning datasets
    "droid_wipe": droid_finetuning_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "libero_10_no_noops_mini": libero_dataset_transform,
    "libero_goal_no_noops_mini": libero_dataset_transform,
    "libero_goal_no_noops_half": libero_dataset_transform,
    "libero_10_no_noops_half": libero_dataset_transform,
    "libero_goal_no_noops_quad": libero_dataset_transform,
    "libero_10_no_noops_quad": libero_dataset_transform,
    "libero_combined": libero_dataset_transform,
    ### Human Dataset
    "ego4d_split_1": human_dataset_transform,
    "ego4d_split_2": human_dataset_transform,
    "ego4d_split_3": human_dataset_transform,
    "ego4d_split_4": human_dataset_transform,
}
