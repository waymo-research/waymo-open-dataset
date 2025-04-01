import pyarrow.parquet as pq

# Function to remove metadata from a PyArrow table's schema
def remove_metadata(table):
    # Replace the schema with one that has no metadata
    return table.replace_schema_metadata(None)

# Function to indent schema printout
def format_schema(schema):
    return '\n'.join(['\t' + line for line in str(schema).split('\n')])

# Load the image parquet file
image_file = 'validation_camera_image_1024360143612057520_3580_000_3600_000.parquet'
image_table = pq.read_table(image_file)

# Remove metadata from the image table
image_table = remove_metadata(image_table)

# Print the total number of rows in the image table
print(f"\nTotal Rows in Image Table: {image_table.num_rows}")

# Print the schema of the image table without metadata, indented by one tab
print(f"Image Table Schema:\n{format_schema(image_table.schema)}")

# Load the segmentation parquet file
segmentation_file = 'validation_camera_segmentation_1024360143612057520_3580_000_3600_000.parquet'
segmentation_table = pq.read_table(segmentation_file)

# Remove metadata from the segmentation table
segmentation_table = remove_metadata(segmentation_table)

# Print the total number of rows in the segmentation table
print(f"\n\nTotal Rows in Segmentation Table: {segmentation_table.num_rows}")

# Print the schema of the segmentation table without metadata, indented by one tab
print(f"Segmentation Table Schema:\n{format_schema(segmentation_table.schema)}")