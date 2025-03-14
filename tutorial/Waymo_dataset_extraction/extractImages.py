import pyarrow.parquet as pq
import numpy as np
import cv2
import os
from tqdm import tqdm

def extract_images(image_file, segmentation_file, output_dir="extracted_images"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load image and segmentation data from parquet files
    image_table = pq.read_table(image_file)
    segmentation_table = pq.read_table(segmentation_file)

    # Define column names
    img_column = '[CameraImageComponent].image'

    seg_timestamp_col = 'key.frame_timestamp_micros'
    seg_camera_col = 'key.camera_name'

    # Create a set of keys from segmentation data to filter corresponding images
    segmentation_keys = set(
        (segmentation_table[seg_timestamp_col][i].as_py(),
         segmentation_table[seg_camera_col][i].as_py())
        for i in range(segmentation_table.num_rows)
    )

    # Iterate through image data and extract images that match segmentation keys with progress bar
    print()
    for idx in tqdm(range(image_table.num_rows), desc="Processing Images"):
        key_tuple = (image_table[seg_timestamp_col][idx].as_py(),
                     image_table[seg_camera_col][idx].as_py()
        )
        if key_tuple in segmentation_keys:
            image_data = image_table[img_column][idx].as_py()
            if image_data:
                # Decode the image from byte data
                img = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
                if img is not None:
                    # Format filename and save image
                    filename = f"image_{key_tuple[0]}_{key_tuple[1]}.png"
                    image_path = os.path.join(output_dir, filename)
                    if not os.path.exists(image_path):
                        cv2.imwrite(image_path, img)

    print(f"Image extraction completed. Saved to {output_dir}")