import tensorflow_datasets as tfds

data_dir = "Your Dataset Directory"  # Replace with your local RLDS dataset path

# Load the RLDS dataset
dataset = tfds.load(
    name="wild_move_to",
    split="train",
    data_dir=data_dir
)

for i, trajectory in enumerate(dataset.take(1)):
    print(f"Trajectory {i}")

    steps = trajectory["steps"]

    # Convert steps to an iterable
    for j, step in enumerate(steps.as_numpy_iterator()):
        print(f"  Step {j}")
        print("    Action:", step["action"])
        print("    EEF pos:", step["observation"]["eef_pos"])
        print("    Gripper width:", step["observation"]["gripper_width"])
        if j >= 2:  # just preview first 3 steps
            break

    # If you have language instruction per trajectory (not per step), it would be here
    if "language_instruction" in trajectory:
        print("Task instruction:", trajectory["language_instruction"].numpy().decode())
