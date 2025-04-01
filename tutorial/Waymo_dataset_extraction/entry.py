import os
from extractImages import extract_images
from extractSegmentations import extract_segmentations
from parquetToCSV import convert_to_csv

"""
    This script extracts images and segmentation masks from Waymo Open Dataset files.
    It checks if the image and segmentation files exist in the current directory.
    If the files do not exist, it prompts the user to download the files from the provided links.
    
    If the files exist:
    - Extracts images and segmentation masks from the files. (Only if the image has a corresponding segmentation key)
    - Saves the extracted images to the 'extracted_images' directory.
    - Saves the extracted segmentation masks to the 'extracted_segmentations' directory.
    
    For this example the image and segmentation files are downloaded from the following links:
    Image file: https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_image/1024360143612057520_3580_000_3600_000.parquet
    Segmentation file: https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_segmentation/1024360143612057520_3580_000_3600_000.parquet
"""

def main():
    # Image and segmentation files
    image_file = 'validation_camera_image_1024360143612057520_3580_000_3600_000.parquet'
    segmentation_file = 'validation_camera_segmentation_1024360143612057520_3580_000_3600_000.parquet'

    # Prompt user to download files only if they don't exist
    if not os.path.exists(image_file) and not os.path.exists(segmentation_file):
        print("Image and segmentation files not found. Please download the files from the following links:")
        print("Image file: https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_image/1024360143612057520_3580_000_3600_000.parquet")
        print("Segmentation file: https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_segmentation/1024360143612057520_3580_000_3600_000.parquet")
        exit()
    if not os.path.exists(image_file):
        print("Image file not found. https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_image/1024360143612057520_3580_000_3600_000.parquet")
        exit()
    if not os.path.exists(segmentation_file):
        print("Segmentation file not found. https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_segmentation/1024360143612057520_3580_000_3600_000")
        exit()
        
    # Converts parquet files to CSV (This makes the parquet files a human readable format)
    print("Converting parquet files to CSV...")
    convert_to_csv(image_file, output_csv='_image_data.csv')
    convert_to_csv(segmentation_file, output_csv='_segmentation_data.csv')

    # Extract images and segmentation masks that have corresponding segmentation keys
    extract_images(image_file, segmentation_file, output_dir="extracted_images")
    extract_segmentations(segmentation_file, image_file, output_dir="extracted_segmentations")
    
    # Print the number of matching pairs in the image and segmentation folders
    images = os.listdir('extracted_images')
    segmentations = os.listdir('extracted_segmentations')
    
    # Remove the prefix from the image and segmentation files for comparison
    count = 0
    for image in images: images[count] = image.removeprefix("image_"); count += 1
    count = 0
    for segmentation in segmentations: segmentations[count] = segmentation.removeprefix("segmask_"); count += 1
    
    # Find the number of matching pairs
    print(f"\nNumber of matching pairs: {len(set(images).intersection(segmentations))}")

if __name__ == "__main__":
    main()