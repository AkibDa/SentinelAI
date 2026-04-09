import os
from pathlib import Path
from generate_masks import generate_masks

# 1. Point these to your extracted Kaggle folders
REAL_DIR = "archive/FaceForensics++_C23/original"
FAKE_DIR = "archive/FaceForensics++_C23/Deepfakes"

# 2. Where the U-Net training data will go
OUTPUT_TRAIN_IMG = "data/train/images"
OUTPUT_TRAIN_MASK = "data/train/masks"

def process_dataset():
    # Get lists of video files
    fake_videos = [f for f in os.listdir(FAKE_DIR) if f.endswith('.mp4')]
    
    print(f"Found {len(fake_videos)} fake videos. Starting extraction...")
    
    # Process the first 50 videos for testing (Change this number later!)
    for fake_video in fake_videos[:50]: 
        fake_path = os.path.join(FAKE_DIR, fake_video)
        
        # In FaceForensics, fake videos are usually named "source_target.mp4" (e.g., "001_002.mp4")
        # The real video is usually the target, e.g., "002.mp4"
        target_id = fake_video.split('_')[1] 
        real_path = os.path.join(REAL_DIR, target_id)
        
        if os.path.exists(real_path):
            print(f"Processing pair: {target_id} and {fake_video}")
            # Extract 10 frames per video pair
            generate_masks(real_path, fake_path, OUTPUT_TRAIN_IMG, OUTPUT_TRAIN_MASK, num_frames=10)
        else:
            print(f"Could not find matching real video for {fake_video}")

if __name__ == "__main__":
    process_dataset()
    print("Data preparation complete! You can now run train.py")