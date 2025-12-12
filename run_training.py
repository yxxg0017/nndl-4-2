
import os
import shutil
import subprocess

# --- Configuration ---
CONTENT_IMAGE = 'input/content/cornell.jpg'
STYLE_IMAGE = 'input/style/woman_with_hat_matisse.jpg'
TRAIN_SCRIPT = 'train_task2.py'

TEMP_CONTENT_DIR = 'task2_content'
TEMP_STYLE_DIR = 'task2_style'
SAVE_DIR = 'models/task2_models'
MAX_ITER = 10000

# --- Main execution ---
def main():
    """Prepares directories, runs training, and cleans up."""
    print("--- Starting training process for Task 2 ---")

    # 1. Create temporary directories
    print(f"Creating temporary directories: {TEMP_CONTENT_DIR}, {TEMP_STYLE_DIR}")
    os.makedirs(TEMP_CONTENT_DIR, exist_ok=True)
    os.makedirs(TEMP_STYLE_DIR, exist_ok=True)
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 2. Copy images to temporary directories
    print("Copying images...")
    shutil.copy(CONTENT_IMAGE, os.path.join(TEMP_CONTENT_DIR, os.path.basename(CONTENT_IMAGE)))
    shutil.copy(STYLE_IMAGE, os.path.join(TEMP_STYLE_DIR, os.path.basename(STYLE_IMAGE)))

    # 3. Construct and run the training command
    command = [
        'python',
        TRAIN_SCRIPT,
        '--content_dir', TEMP_CONTENT_DIR,
        '--style_dir', TEMP_STYLE_DIR,
        '--save_dir', SAVE_DIR,
        '--max_iter', str(MAX_ITER),
        '--n_threads', '0' # Necessary for Windows compatibility
    ]
    
    print(f"Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print("--- Training completed successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"Error during training: {e}")
    except FileNotFoundError:
        print(f"Error: Could not find the training script at {TRAIN_SCRIPT}. Please check the path.")
    finally:
        # 4. Clean up temporary directories
        print("Cleaning up temporary directories...")
        shutil.rmtree(TEMP_CONTENT_DIR)
        shutil.rmtree(TEMP_STYLE_DIR)
        print("--- Cleanup complete ---")

if __name__ == '__main__':
    main()

