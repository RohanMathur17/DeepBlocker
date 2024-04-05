import pandas as pd
import numpy as np
import sys

def create_test_csv(product_id_pairs, random_product_id_non_pairs, no_samples):
    """
    Create a test CSV file by processing and combining data from product_id_pairs_new.csv and random_product_id_pairs_new.csv.
    """
    # Read the CSV files into DataFrames
    df = product_id_pairs
    df_2 = random_product_id_non_pairs

    # Add labels to distinguish between pairs from different sources
    df["label"] = 1
    df_2["label"] = 0

    # Shuffle the DataFrames
    df_shuffled = df.sample(frac=1, random_state=42)
    df_shuffled_2 = df_2.sample(frac=1, random_state=42)

    # Drop duplicates of product_id_1, keeping one instance randomly, and select a subset
    filtered_df = df_shuffled.drop_duplicates(subset='product_id_1', keep='first').sample(n=no_samples, random_state=42)

    # Drop duplicates of product_id_1, keeping one instance randomly, and select a subset
    filtered_df_2 = df_shuffled_2.drop_duplicates(subset='product_id_1', keep='first').sample(n=no_samples, random_state=42)

    # Combine the two subsets
    final_df = pd.concat([filtered_df, filtered_df_2], ignore_index=True)

    # Shuffle the combined DataFrame
    shuffled_df = final_df.sample(frac=1, random_state=42)

    # Reset index and rename columns
    shuffled_df.reset_index(drop=True, inplace=True)
    shuffled_df.rename(columns={"product_id_1":"ltable_id","product_id_2":"rtable_id"}, inplace=True)

    # Write the final DataFrame to a new CSV file
    shuffled_df.to_csv("Datasets/Turkish/Training/test_" + str(no_samples) + ".csv", index=False)

    print("Filtered CSV file created successfully")


if len(sys.argv) != 3:
    print("Usage: python3 make_test.py  <input_file_1> <input_file_2>")
    sys.exit(1)

#product_id_pairs = "Datasets/Turkish/product_id_pairs.csv"
#random_product_id_non_pairs = "Datasets/Turkish/random_product_id_non_pairs.csv"
product_id_pairs = sys.argv[1]
random_product_id_non_pairs = sys.argv[2]

print("Reading the input files...")
product_id_pairs = pd.read_csv(product_id_pairs)
random_product_id_non_pairs = pd.read_csv(random_product_id_non_pairs)


# Call the function to create the test CSV file
create_test_csv(product_id_pairs, random_product_id_non_pairs, 100)

# Call the function to create the test CSV file
create_test_csv(product_id_pairs, random_product_id_non_pairs, 150)

# Call the function to create the test CSV file
create_test_csv(product_id_pairs, random_product_id_non_pairs, 200)
