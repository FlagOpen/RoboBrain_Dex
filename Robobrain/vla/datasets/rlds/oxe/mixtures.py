"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

# fmt: off
OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    # === Human Ego Data for Dex Uni Token ===
    "egodex": [ 
        ("h2o", 1.0),
        ("ph2d", 1.0),
        ("oakink", 1.0),
        ("arctic", 1.0),
        ("egodex_part1", 0.038),
        ("egodex_part2", 0.039),
        ("egodex_part3", 0.042),
        ("egodex_part4", 0.041),
        ("egodex_part5", 0.040),
        ("fourier_all", 0.14),
        ("hrdverse", 0.7)
        # other human ego data
        
    ],
    # === Real Robot Data ===
    "robot": [ 
        ("grab_the_apple_and_put_it_into_the_basket", 1.0),
        # other real world data
    ],
    
    "oxe": [ 
        ("fractal20220817_data", 1.0),
    ],
    # === Bridge V2 Dataset ===
    "bridge": [
        ("bridge_oxe", 1.0),                                      # Version of Bridge V2 in Open-X GCP Bucket
        # ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
    ],

    "droid": [
        ("droid", 1.0),
    ],
    

    "roboset": [
        ("roboset", 1.0),
    ],

    "stanford_robocook_converted_externally_to_rlds": [
        ("stanford_robocook_converted_externally_to_rlds", 1.0),
    ],

    # === [Moderate-Scale] Bridge++ Mixtures ===
    "bridge_rt_1": [
        # ("bridge_oxe", 1.0)                                   # Version of Bridge V2 in Open-X GCP Bucket
        ("bridge_orig", 1.0),                                   # Original Version of Bridge V2 from Project Website
        ("fractal20220817_data", 1.0),                          # Google RT-1 Robot Data (Large-Scale)
    ],

    "rt_1": [
        ("fractal20220817_data", 1.0),
    ],
    
    # === T-DROID Dataset ===
    "tdroid_carrot_in_bowl": [
        ("tdroid_carrot_in_bowl", 1.0),
    ],
    "tdroid_pour_corn_in_pot": [
        ("tdroid_pour_corn_in_pot", 1.0),
    ],
    "tdroid_flip_pot_upright": [
        ("tdroid_flip_pot_upright", 1.0),
    ],
    "tdroid_move_object_onto_plate": [
        ("tdroid_move_object_onto_plate", 1.0),
    ],
    "tdroid_knock_object_over": [
        ("tdroid_knock_object_over", 1.0),
    ],
    "tdroid_cover_object_with_towel": [
        ("tdroid_cover_object_with_towel", 1.0),
    ],

    # === DROID Finetuning Datasets ===
    "droid_wipe": [
        ("droid_wipe", 1.0),
    ],

    # === LIBERO Datasets (Modified Versions) ===
    "libero_spatial_no_noops": [
        ("libero_spatial_no_noops", 1.0),
    ],
    "libero_object_no_noops": [
        ("libero_object_no_noops", 1.0),
    ],
    "libero_goal_no_noops": [
        ("libero_goal_no_noops", 1.0),
    ],
    "libero_10_no_noops": [
        ("libero_10_no_noops", 1.0),
    ],
    "libero_10_no_noops_mini": [
        ("libero_10_no_noops_mini", 1.0),
    ],
    "libero_goal_no_noops_mini": [
        ("libero_goal_no_noops_mini", 1.0),
    ],
    "libero_goal_no_noops_half": [
        ("libero_goal_no_noops_half", 1.0),
    ],
    "libero_10_no_noops_half": [
        ("libero_10_no_noops_half", 1.0),
    ],
    "libero_goal_no_noops_quad": [
        ("libero_goal_no_noops_quad", 1.0),
    ],
    "libero_10_no_noops_quad": [
        ("libero_10_no_noops_quad", 1.0),
    ],
    "libero_combined": [
        ("libero_combined", 1.0),
    ],
}
# fmt: on
