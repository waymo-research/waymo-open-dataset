import pyarrow.parquet as pq
import os

def convert_to_csv(file, output_csv):
    # Check if the  file exists
    if not os.path.exists(file):
        print(f"File not found: {file}")
        return
    
    # Load the parquet file as a pandas DataFrame
    df = pq.read_table(file).to_pandas()

    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"{file} data has been saved to {output_csv}")