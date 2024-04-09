import csv
import os
import pandas as pd
import glob
import sys

def create_product_feature_csv(input_directory, input_file, col_name):
    """
    Creates a CSV file containing product features extracted from the given input file.
    
    Args:
        input_directory (str): Path to the directory containing the input file.
        input_file (str): Name of the input file containing product features.
        col_name (str): Name of the column representing the product feature.
    """
    # Construct the output file path
    output_file = os.path.join(input_directory, col_name + ".csv")
    
    # Open input and output files
    with open(input_file, 'r', encoding="utf-8") as infile, open(output_file, 'w', newline='') as outfile:
        reader = infile.readlines()
        writer = csv.writer(outfile)
    
        # Write CSV header
        writer.writerow(["product_id", col_name])

        # Process each line in the input file
        for line in reader:
            # Split line by ';'
            parts = line.strip().split(';', 1)
            
            # Write data to CSV file
            writer.writerow(parts)

def process_txt_files(input_directory):
    """
    Process all .txt files in the specified directory and create corresponding CSV files.
    
    Args:
        input_directory (str): Path to the directory containing the .txt files.
    """
    # Iterate over files in the directory
    for filename in os.listdir(input_directory):
        # Check if the file is a .txt file
        if filename.endswith(".txt"):
            # Construct the full path to the .txt file
            file_path = os.path.join(input_directory, filename)
            
            # Create product feature CSV for the current .txt file
            create_product_feature_csv(input_directory, file_path, filename.replace(".txt", ""))

def delete_csv_files(directory):
    """
    Deletes all CSV files in the specified directory.

    Args:
        directory (str): Path to the directory containing CSV files.
    """
    # Iterate over files in the directory
    for filename in os.listdir(directory):
        # Check if the file is a CSV file
        if filename.endswith('.csv'):
            # Construct the full path to the file
            filepath = os.path.join(directory, filename)
            # Delete the file
            os.remove(filepath)
            
def merge_csv_files(input_directory):
    """
    Merge all CSV files in the specified directory based on the 'product_id' column.

    Args:
        input_directory (str): Path to the directory containing CSV files.
    
    Returns:
        pd.DataFrame: Merged DataFrame containing data from all CSV files.
    """
    # List all CSV files in the directory
    csv_files = glob.glob(input_directory + "/*.csv")

    # Initialize an empty DataFrame to store the merged data
    merged_df = pd.DataFrame()

    # Iterate through each CSV file
    for file in csv_files:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file)
        
        # Merge the current DataFrame with the merged DataFrame based on 'product_id'
        if merged_df.empty:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on='product_id', how='outer')

    return merged_df

def process(input_directory):

    print("Processing .txt files....")
    print("--------------------------")
    process_txt_files(input_directory)

    print("Created individual CSV files!")
    print("--------------------------")
    print("Merging .CSV Files....")
    merged_df = merge_csv_files(input_directory)

    merged_df.to_csv("Datasets/Turkish/merged.csv", index=False)

    print("--------------------------")

    print("Created Merged .CSV File!")

    print("--------------------------")

    print("Deleting individual .CSV Files....")

    delete_csv_files(input_directory)

    print("--------------------------")

    print("Deleted individual .CSV Files!")

    print("--------------------------")

    print("MERGED DATASET IS READY -> merged.csv")
    
if len(sys.argv) != 2:
    print("Usage: python script.py <input_directory>")
    sys.exit(1)

# Get the input directory from the command-line argument
#input_directory = "Datasets/Turkish/Only_Price_Having_Products/needed/"
input_directory = sys.argv[1]

process(input_directory)
