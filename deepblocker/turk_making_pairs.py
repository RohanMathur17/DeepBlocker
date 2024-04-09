import pandas as pd
import sys

def process_and_write_csvs(tableA, tableB):
    """
    Process and merge two CSV files (tableA.csv and tableB.csv) and write the results to new CSV files.
    """
   
    # Merge tableA and tableB based on 'Product_Classes' column
    merged_df = pd.merge(tableA[["id","Product_Classes"]], tableB[["id","Product_Classes"]], on="Product_Classes", suffixes=('_1', '_2'))
    merged_df.rename(columns={'id_1': 'product_id_1', 'id_2': 'product_id_2'}, inplace=True)
    merged_df.drop(columns=['Product_Classes'], inplace=True)

    # Get unique product classes from tableA and tableB
    unique_product_classes_1 = tableA['Product_Classes'].unique()
    unique_product_classes_2 = tableB['Product_Classes'].unique()

    # Create a DataFrame to store the cartesian product of the unique values
    product_classes_df = pd.DataFrame([(pc1, pc2) for pc1 in unique_product_classes_1 for pc2 in unique_product_classes_2],
                                      columns=['Product_Classes_1', 'Product_Classes_2'])

    # Merge with original dataframes to get corresponding "id" values
    result_df = product_classes_df.merge(tableA[["id","Product_Classes"]], how='left', left_on='Product_Classes_1', right_on='Product_Classes') \
                                   .merge(tableB[["id","Product_Classes"]], how='left', left_on='Product_Classes_2', right_on='Product_Classes') \
                                   [['Product_Classes_1', 'id_x', 'Product_Classes_2', 'id_y']]
    result_df.rename(columns={'id_x': 'product_id_1', 'id_y': 'product_id_2'}, inplace=True)
    result_df.drop(columns=['Product_Classes_1','Product_Classes_2'], inplace=True)

    return merged_df, result_df

if len(sys.argv) != 3:
    print("Usage: python3 turk_making_matches.py  <input_file_1> <input_file_2>")
    sys.exit(1)

#tableA = "Datasets/Turkish/Training/tableA.csv"
#tableB = "Datasets/Turkish/Training/tableB.csv"
tableA = sys.argv[1]
tableB = sys.argv[2]

tableA = pd.read_csv(tableA)
tableB = pd.read_csv(tableB)

product_id_pairs, product_id_non_pairs = process_and_write_csvs(tableA, tableB)

# Write the merged DataFrame to a new CSV file
product_id_pairs.to_csv("Datasets/Turkish/product_id_pairs.csv", index=False)

# Write the result DataFrame to a new CSV file
product_id_non_pairs.to_csv("Datasets/Turkish/random_product_id_non_pairs.csv", index=False)

print("Product ID pairs CSV file created successfully --> product_id_pairs.csv")
print("Product ID non-pairs CSV file created successfully --> random_product_id_non_pairs.csv")
