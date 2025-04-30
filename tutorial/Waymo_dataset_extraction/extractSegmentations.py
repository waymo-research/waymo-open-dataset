import pyarrow.parquet as pq
import numpy as np
import cv2
import os
from tqdm import tqdm

def extract_segmentations(segmentation_file, image_file, output_dir="extracted_segmentations"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load segmentation and image data from parquet files
    segmentation_table = pq.read_table(segmentation_file)
    image_table = pq.read_table(image_file)

    # Define column names for segmentation components
    seg_column = '[CameraSegmentationLabelComponent].panoptic_label'
    seg_timestamp_col = 'key.frame_timestamp_micros'
    seg_camera_col = 'key.camera_name'

    # Create a set of image keys to ensure segmentation has a corresponding image
    img_timestamp_col = 'key.frame_timestamp_micros'
    img_camera_col = 'key.camera_name'
    image_keys = set(
        (image_table[img_timestamp_col][i].as_py(),
         image_table[img_camera_col][i].as_py())
        for i in range(image_table.num_rows)
    )

    # Iterate through segmentation data and extract masks with progress bar
    print()
    for idx in tqdm(range(segmentation_table.num_rows), desc="Processing Segmentations"):
        key_tuple = (segmentation_table[seg_timestamp_col][idx].as_py(),
                     segmentation_table[seg_camera_col][idx].as_py())
        seg_data = segmentation_table[seg_column][idx].as_py()
        
        # Only save segmentation if corresponding image exists
        if seg_data and key_tuple in image_keys:
            mask = cv2.imdecode(np.frombuffer(seg_data, np.uint8), cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                filename = f"segmask_{key_tuple[0]}_{key_tuple[1]}.png"
                mask_path = os.path.join(output_dir, filename)
                if not os.path.exists(mask_path):
                    cv2.imwrite(mask_path, mask)

    print(f"Segmentation extraction completed. Saved to {output_dir}")