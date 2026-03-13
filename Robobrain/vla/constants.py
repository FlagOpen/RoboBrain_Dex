"""
Important constants for VLA training and evaluation.

Attempts to automatically identify the correct constants to set based on the Python command used to launch
training or evaluation. If it is unclear, defaults to using the LIBERO simulation benchmark constants.
"""
import sys
from enum import Enum

# token constants
IGNORE_INDEX = -100

# Defines supported normalization schemes for action and proprioceptive state.
class NormalizationType(str, Enum):
    # fmt: off
    NORMAL = "normal"               # Normalize to Mean = 0, Stdev = 1
    BOUNDS = "bounds"               # Normalize to Interval = [-1, 1]
    BOUNDS_Q99 = "bounds_q99"       # Normalize [quantile_01, ..., quantile_99] --> [-1, ..., 1]
    # fmt: on


# Define constants for each robot platform
PRETRAIN_CONSTANTS = {
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS_Q99,
}

FINETUNE_CONSTANTS = {
    "ACTION_PROPRIO_NORMALIZATION_TYPE": NormalizationType.BOUNDS,
}

# Function to detect robot platform from command line arguments
def detect_robot_platform():
    cmd_args = " ".join(sys.argv).lower()

    if "finetune" in cmd_args:
        return "FINETUNE"
    else:
        # Default to LIBERO if unclear
        return "PRETRAIN"


# Determine which robot platform to use
ROBOT_PLATFORM = detect_robot_platform()

# Set the appropriate constants based on the detected platform
if ROBOT_PLATFORM == "PRETRAIN":
    constants = PRETRAIN_CONSTANTS
elif ROBOT_PLATFORM == "FINETUNE":
    constants = FINETUNE_CONSTANTS
    
# Assign constants to global variables
ACTION_PROPRIO_NORMALIZATION_TYPE = constants["ACTION_PROPRIO_NORMALIZATION_TYPE"]

# Print which robot platform constants are being used (for debugging)
print(f"Using {ROBOT_PLATFORM} constants:")
print(f"  ACTION_PROPRIO_NORMALIZATION_TYPE = {ACTION_PROPRIO_NORMALIZATION_TYPE}")

