import tensorflow_datasets as tfds
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

data_dir = "Your Dataset Directory"
output_dir = "visualized_images_wild_move_to"
os.makedirs(output_dir, exist_ok=True)

# Load the RLDS dataset
dataset = tfds.load(
    name="wild_move_to_6932",
    split="train",
    data_dir=data_dir
)

print(f"Dataset loaded with {len(list(dataset))} trajectories.")

for i, trajectory in enumerate(dataset):
    # if i !=3 :
    #     continue
    print(f"Trajectory {i}")

    steps = trajectory["steps"]

    trajectory_dir = os.path.join(output_dir, f"trajectory_{i:03d}")
    os.makedirs(trajectory_dir, exist_ok=True)

    # Convert steps to an iterable
    for j, step in enumerate(steps.as_numpy_iterator()):
        # if j >= 1:  # just preview the first step
        #     break

        print(f"  Step {j}")
        print("    Action:", step["action"])
        print("    EEF pos:", step["observation"]["eef_pos"])
        print("    Gripper width:", step["observation"]["gripper_width"])
        
        if language_instruction := step.get("language_instruction"):
            print("    Language instruction:", language_instruction.decode())

        image = step["observation"]["image"]
        print("    Image shape:", image.shape)
        print("    Image type:", image.dtype)
        print(type(image))
        pil_image = Image.fromarray(image)
        scale_factor = 2
        new_size = (image.shape[1] * scale_factor, image.shape[0] * scale_factor)
        
        # Use LANCZOS resampling for better quality
        upscaled = pil_image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Apply sharpening filter
        sharpened = upscaled.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
        
        # Enhance contrast slightly
        enhancer = ImageEnhance.Contrast(sharpened)
        enhanced = enhancer.enhance(1.2)
        
        
        # Create figure with image and text
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Display the enhanced image
        ax.imshow(enhanced)
        ax.axis('off')
        
        # Add title with step info
        ax.set_title(f"Trajectory {i}, Step {j}", fontsize=16, fontweight='bold', pad=20)
        
        # Add language instruction as text below the image
        if language_instruction:
            # Wrap long text
            import textwrap
            wrapped_text = textwrap.fill(language_instruction.decode(), width=80)
            
            # Add text box with instruction
            textstr = f"Language Instruction:\n{wrapped_text}"
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1)
            ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=12,
                   verticalalignment='bottom', bbox=props, fontfamily='monospace')
        
        # Add robot state info in top-right corner
        # eef_pos = step["observation"]["eef_pos"]
        # gripper = step["observation"]["gripper_width"]
        # action = step["action"]
        
        # info_text = f"EEF: [{eef_pos[0]:.2f}, {eef_pos[1]:.2f}, {eef_pos[2]:.2f}]\nGripper: {gripper[0]:.3f}"
        # info_props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, pad=0.5)
        # ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        #        verticalalignment='top', horizontalalignment='right', 
        #        bbox=info_props, fontfamily='monospace')
        
        plt.tight_layout()
        
        # Save the annotated image
        image_path = os.path.join(trajectory_dir, f"annotated_step_{j:03d}.png")
        plt.savefig(image_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        # Also save the plain enhanced image
        # plain_image_path = os.path.join(trajectory_dir, f"enhanced_step_{j:03d}.png")
        # enhanced.save(plain_image_path, quality=95, optimize=True)
        
        print(f"    Saved annotated image to: {image_path}")


    # print(f"  Number of steps: {len(steps)}")

    # If you have language instruction per trajectory (not per step), it would be here
    # if "language_instruction" in trajectory:
    #     print("Task instruction:", trajectory["language_instruction"].numpy().decode())
