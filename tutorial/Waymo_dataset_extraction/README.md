
# Waymo Open Dataset Extraction Scripts

## Overview
This program automates the extraction and conversion of images and segmentation masks from the Waymo Open Dataset. It ensures that data is organized, filtered, and transformed into formats suitable for further analysis or machine learning applications.

## Features
- **File Validation:** Automatically checks if the required image and segmentation Parquet files exist. If missing, it prompts the user to download the necessary files.
- **Data Conversion:** Converts Parquet files to CSV format, making them human-readable and easier to analyze.
- **Image and Segmentation Extraction:**
  - Extracts images and segmentation masks only when they have corresponding segmentation keys, ensuring consistency.
  - Saves images to the `extracted_images` directory and segmentation masks to the `extracted_segmentations` directory.
- **Reporting:**
  - Prints the number of extracted images and segmentation masks.
  - Displays the count of matching image-segmentation pairs.
- **Schema Analysis (`schemaExtraction.py`):**
  - Analyzes and displays the schema of Parquet files without metadata for clearer understanding of data structure.

## Installation
Install the required dependencies using the following command:

```bash
pip install pyarrow numpy opencv-python tqdm
```

## Usage
1. **Run the Main Program:**

```bash
python entry.py
```

2. **Download Required Files if Prompted:**
   - [Image File](https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_image/1024360143612057520_3580_000_3600_000.parquet)
   - [Segmentation File](https://storage.googleapis.com/waymo_open_dataset_v_2_0_1/validation/camera_segmentation/1024360143612057520_3580_000_3600_000.parquet)

3. **Review Extracted Data:**
   - Images are saved in the `extracted_images` directory.
   - Segmentation masks are saved in the `extracted_segmentations` directory.
   - CSV files (`_image_data.csv` and `_segmentation_data.csv`) contain converted data for inspection.

## Exported Data
- **Extracted Images:** Saved in the `extracted_images` directory.
- **Extracted Segmentations:** Saved in the `extracted_segmentations` directory.
- **CSV Files:**
  - `_image_data.csv` for image data.
  - `_segmentation_data.csv` for segmentation data.

## Notes
- The program avoids redundant file processing by skipping already existing extractions.
- Progress indicators (via `tqdm`) help visualize long extraction processes.
- Schema details are printed with indentation for better readability and analysis.
- The program ensures a consistent pairing between images and segmentation masks for accurate data correlation.
