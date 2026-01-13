"""
Example usage of the /api/generate_mask endpoint

This script demonstrates how to:
1. Upload a video
2. Request a mask generation with positive and negative points
3. Retrieve and save the mask
"""

import requests
import base64
from PIL import Image
from io import BytesIO

# Configuration
SERVER_URL = "http://localhost:8000"  # Adjust if your server runs on a different port

def test_mask_generation():
    # Step 1: Upload a video (replace with your actual video path)
    print("Step 1: Upload video...")
    # Uncomment and modify this section when you have a video to test
    # with open("path/to/your/video.mp4", "rb") as f:
    #     files = {"file": f}
    #     response = requests.post(f"{SERVER_URL}/api/upload_video", files=files)
    #     video_id = response.json()["video_id"]
    #     print(f"Video uploaded with ID: {video_id}")

    # For testing, use an existing video_id
    video_id = "your_video_id_here"

    # Step 2: Generate mask with positive and negative points
    print("\nStep 2: Generate mask...")

    # Example:
    # - 1 positive point in the center of the object you want to segment
    # - 1 negative point outside the object to exclude unwanted areas
    mask_request = {
        "video_id": video_id,
        "video_time": 1.0,  # Get frame at 1 second
        "points": [
            {"x": 0.5, "y": 0.5, "label": 1},   # Positive point (center)
            {"x": 0.3, "y": 0.3, "label": 1},   # Another positive point
            {"x": 0.1, "y": 0.1, "label": 0},   # Negative point (background)
        ]
    }

    response = requests.post(f"{SERVER_URL}/api/generate_mask", json=mask_request)

    if response.status_code == 200:
        result = response.json()
        print(f"Status: {result['status']}")

        # Step 3: Decode and save the mask
        mask_base64 = result["mask"]
        mask_data = base64.b64decode(mask_base64)
        mask_image = Image.open(BytesIO(mask_data))

        # Save the mask
        output_path = f"mask_{video_id}.png"
        mask_image.save(output_path)
        print(f"Mask saved to: {output_path}")
        print(f"Mask dimensions: {mask_image.size}")
    else:
        print(f"Error: {response.status_code}")
        print(response.json())

if __name__ == "__main__":
    test_mask_generation()
