import pandas as pd
import sys
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

# Reset warnings to default behavior
warnings.resetwarnings()

def split_and_write_csv(input_csv):
    """
    Load the original CSV file, shuffle its rows, split it into two subsets, 
    and write each subset to a separate CSV file.

    Args:
        input_csv (str): Path to the original CSV file.
    """
    # Load the original CSV file

    print("Reading Merged CSV File....")
    df = pd.read_csv(input_csv)
    # Shuffle the rows randomly
    df_shuffled = df.sample(frac=1, random_state=42)  # Setting random_state for reproducibility

    # Determine the size of each subset
    split_index = len(df_shuffled) // 2

    # Split the shuffled DataFrame into two subsets
    subset1_df = df_shuffled.iloc[:split_index]
    subset2_df = df_shuffled.iloc[split_index:]

    # Rename the 'product_id' column to 'id'
    subset1_df.rename(columns={'product_id': 'id'}, inplace=True)
    subset2_df.rename(columns={'product_id': 'id'}, inplace=True)

    # Reset index and create 'id' column starting from 0
    subset1_df['id'] = subset1_df.reset_index(drop=True).index
    subset2_df['id'] = subset2_df.reset_index(drop=True).index

    # Write the subsets to separate CSV files
    subset1_df.to_csv("Datasets/Turkish/Training/tableA.csv", index=False)
    subset2_df.to_csv("Datasets/Turkish/Training/tableB.csv", index=False)

    print("Created TableA.csv!")
    print("Created TableB.csv!")
    
if len(sys.argv) != 2:
    print("Usage: python3 split_merged.py  <input_file>")
    sys.exit(1)

input_file = sys.argv[1]

split_and_write_csv(input_file)

print("Finished Successfully!")
    