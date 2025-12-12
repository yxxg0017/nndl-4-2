
import os
import shutil
import subprocess
from pathlib import Path

# --- Configuration ---
CONTENT_IMAGE = 'input/content/cornell.jpg'
STYLE_IMAGES = [
    'input/style/antimonocromatismo.jpg',
    'input/style/la_muse.jpg',
    'input/style/sketch.png'
]
TRAIN_SCRIPT = 'train_task2.py'

# Temporary directory for the single content image
TEMP_CONTENT_DIR = 'temp_content'
# Base temporary directory for style images
TEMP_STYLE_DIR_BASE = 'temp_style'

# Directory to save the final models
SAVE_DIR = 'experiments'
MAX_ITER = 10000

# --- Main execution ---
def main():
    """Prepares directories, runs training for each style, and cleans up."""
    print("--- Starting training process for multiple styles ---")

    # 1. Prepare the content directory once
    print(f"Creating temporary content directory: {TEMP_CONTENT_DIR}")
    os.makedirs(TEMP_CONTENT_DIR, exist_ok=True)
    shutil.copy(CONTENT_IMAGE, os.path.join(TEMP_CONTENT_DIR, os.path.basename(CONTENT_IMAGE)))

    # Ensure the main save directory exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. Loop through each style image and train
    for style_image_path in STYLE_IMAGES:
        style_name = Path(style_image_path).stem
        print(f"\n--- Training with style: {style_name} ---")
        
        # Create a unique temporary directory for the current style
        current_temp_style_dir = f"{TEMP_STYLE_DIR_BASE}_{style_name}"
        os.makedirs(current_temp_style_dir, exist_ok=True)
        shutil.copy(style_image_path, os.path.join(current_temp_style_dir, os.path.basename(style_image_path)))

        # 3. Construct and run the training command
        command = [
            'python',
            TRAIN_SCRIPT,
            '--content_dir', TEMP_CONTENT_DIR,
            '--style_dir', current_temp_style_dir,
            '--save_dir', SAVE_DIR,
            '--log_dir', 'logs',
            '--max_iter', str(MAX_ITER),
            '--n_threads', '0'  # Necessary for Windows compatibility
        ]
        
        print(f"Running command: {' '.join(command)}")
        try:
            subprocess.run(command, check=True)
            print(f"--- Training for style '{style_name}' completed successfully ---")
        except subprocess.CalledProcessError as e:
            print(f"Error during training for style '{style_name}': {e}")
        except FileNotFoundError:
            print(f"Error: Could not find the training script at {TRAIN_SCRIPT}. Please check the path.")
            # Stop if the main script is missing
            break 
        finally:
            # Clean up the temporary style directory for this iteration
            print(f"Cleaning up temporary style directory: {current_temp_style_dir}")
            shutil.rmtree(current_temp_style_dir)

    # 4. Final cleanup of the content directory
    print("\nCleaning up temporary content directory...")
    shutil.rmtree(TEMP_CONTENT_DIR)
    print("--- All training processes complete. ---")

if __name__ == '__main__':
    main()

