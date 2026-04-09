import cv2
import numpy as np
from pathlib import Path

def generate_masks(real_video_path, fake_video_path, output_img_dir, output_mask_dir, num_frames=10):
    """
    Extracts frames from real/fake video pairs, calculates the difference to create a mask,
    and saves the fake frames and corresponding masks for U-Net training.
    """
    cap_real = cv2.VideoCapture(str(real_video_path))
    cap_fake = cv2.VideoCapture(str(fake_video_path))
    
    Path(output_img_dir).mkdir(parents=True, exist_ok=True)
    Path(output_mask_dir).mkdir(parents=True, exist_ok=True)
    
    frame_count = 0
    saved_count = 0
    
    while cap_real.isOpened() and cap_fake.isOpened() and saved_count < num_frames:
        ret_r, frame_r = cap_real.read()
        ret_f, frame_f = cap_fake.read()
        
        if not ret_r or not ret_f:
            break
            
        # Extract every 10th frame to ensure diversity
        if frame_count % 10 == 0:
            # Resize real frame to match fake frame dimensions
            frame_r = cv2.resize(frame_r, (frame_f.shape[1], frame_f.shape[0]))
            diff = cv2.absdiff(frame_r, frame_f)
            gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
            
            frame_f_resized = cv2.resize(frame_f, (224, 224))
            mask_resized = cv2.resize(mask, (224, 224))
            
            base_name = f"{Path(fake_video_path).stem}_f{frame_count}"
            cv2.imwrite(f"{output_img_dir}/{base_name}.jpg", frame_f_resized)
            cv2.imwrite(f"{output_mask_dir}/{base_name}.png", mask_resized)
            
            saved_count += 1
            
        frame_count += 1

    cap_real.release()
    cap_fake.release()

if __name__ == "__main__":
    # Example usage:
    # generate_masks("data/raw/real/001.mp4", "data/raw/fake/001_002.mp4", "data/train/images", "data/train/masks")
    pass