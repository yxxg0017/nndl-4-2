
import argparse
from pathlib import Path
import os

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Import necessary components from the project's files
import net
from function import adaptive_instance_normalization, coral

# --- Core Style Transfer Logic (adapted from test.py) ---

def test_transform(size, crop):
    """Create image transformation pipeline for testing."""
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)

def style_transfer(vgg, decoder, content, style, alpha=1.0):
    """Performs AdaIN style transfer."""
    assert 0.0 <= alpha <= 1.0
    with torch.no_grad():
        content_f = vgg(content)
        style_f = vgg(style)
        feat = adaptive_instance_normalization(content_f, style_f)
        feat = feat * alpha + content_f * (1 - alpha)
        output = decoder(feat)
    return output

def run_style_transfer(vgg, decoder, content_path, style_path, output_path, alpha=1.0, content_size=512, style_size=512, crop=False):
    """Loads images, runs style transfer, and saves the output."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and transform images
    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    
    content_img = Image.open(content_path).convert('RGB')
    style_img = Image.open(style_path).convert('RGB')

    content = content_tf(content_img).unsqueeze(0).to(device)
    style = style_tf(style_img).unsqueeze(0).to(device)

    # Perform style transfer
    output = style_transfer(vgg, decoder, content, style, alpha)
    
    # Save the output image
    save_image(output.cpu(), output_path)
    print(f"Saved output to {output_path}")

# --- Main Task Execution ---

def main():
    """Main function to execute all requested tasks."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_root = Path('output')
    output_root.mkdir(exist_ok=True, parents=True)

    # --- Setup Models ---
    print("Setting up models...")
    vgg_path = 'models/vgg_normalised.pth'
    
    vgg = net.vgg
    vgg.load_state_dict(torch.load(vgg_path))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    vgg.eval()

    decoder = net.decoder
    decoder.to(device)
    decoder.eval()

    print("--- Task 2: Style Transfer at Different Training Iterations ---")
    # This task requires trained models from the training step.
    task2_output_dir = output_root / 'task2'
    task2_output_dir.mkdir(exist_ok=True, parents=True)
    
    content_task2 = 'input/content/cornell.jpg'
    style_task2 = 'input/style/woman_with_hat_matisse.jpg'
    iterations = [1000, 5000, 8000, 10000] # 10%, 50%, 80%, 100% of 10k

    for i in iterations:
        decoder_path = f'experiments/decoder_iter_{i}.pth'
        if os.path.exists(decoder_path):
            print(f"Processing with decoder from iteration {i}...")
            decoder.load_state_dict(torch.load(decoder_path))
            output_path = task2_output_dir / f'cornell_stylized_iter_{i}.jpg'
            run_style_transfer(vgg, decoder, content_task2, style_task2, output_path)
        else:
            print(f"Decoder model not found: {decoder_path}. Skipping.")

    print("\n--- Task 3: Style Transfer with Custom Images ---")
    # This task also requires a trained decoder. We'll use the final one from Task 2.
    task3_output_dir = output_root / 'task3'
    task3_output_dir.mkdir(exist_ok=True, parents=True)
    final_decoder_path = 'experiments/decoder_iter_10000.pth'

    if os.path.exists(final_decoder_path):
        decoder.load_state_dict(torch.load(final_decoder_path))
        
        content_task3_1 = 'myimage/1.jpg'
        content_task3_2 = 'myimage/2.jpg'
        style_task3 = 'input/style/la_muse.jpg' # Choosing a style

        print(f"Stylizing {content_task3_1} with {style_task3}")
        output_path_1 = task3_output_dir / '1_stylized_la_muse.jpg'
        run_style_transfer(vgg, decoder, content_task3_1, style_task3, output_path_1)

        print(f"Stylizing {content_task3_2} with {style_task3}")
        output_path_2 = task3_output_dir / '2_stylized_la_muse.jpg'
        run_style_transfer(vgg, decoder, content_task3_2, style_task3, output_path_2)
    else:
        print(f"Final decoder model not found: {final_decoder_path}. Skipping Task 3.")

    print("\n--- Task 4: Style Transfer with Alpha Blending ---")
    task4_output_dir = output_root / 'task4'
    task4_output_dir.mkdir(exist_ok=True, parents=True)

    if os.path.exists(final_decoder_path):
        decoder.load_state_dict(torch.load(final_decoder_path))

        content_task4 = 'myimage/1.jpg' # Choosing one of the photos
        style_task4 = 'input/style/sketch.png' # Choosing a different style
        alphas = [0.2, 0.5, 0.8]

        print(f"Stylizing {content_task4} with {style_task4} using different alphas.")
        for alpha in alphas:
            print(f"Processing with alpha = {alpha}...")
            output_path = task4_output_dir / f'1_stylized_sketch_alpha_{alpha}.jpg'
            run_style_transfer(vgg, decoder, content_task4, style_task4, output_path, alpha=alpha)
    else:
        print(f"Final decoder model not found: {final_decoder_path}. Skipping Task 4.")

if __name__ == '__main__':
    main()
