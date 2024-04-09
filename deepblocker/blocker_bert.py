import numpy as np
import pandas as pd
from pathlib import Path
import blocking_utils
from bert_test import do_bert_embeddings

class BlockerBert:
    def __init__(self, vector_pairing_model, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.vector_pairing_model = vector_pairing_model
       
    def validate_columns(self):
        #Assumption: id column is named as id
        if "id" not in self.cols_to_block:
            self.cols_to_block.append("id")
        self.cols_to_block_without_id = [col for col in self.cols_to_block if col != "id"]

        #Check if all required columns are in left_df
        check = all([col in self.left_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the left dataset")

        #Check if all required columns are in right_df
        check = all([col in self.right_df.columns for col in self.cols_to_block])
        if not check:
            raise Exception("Not all columns in cols_to_block are present in the right dataset")

    def preprocess_datasets(self):
        self.left_df = self.left_df[self.cols_to_block]
        self.right_df = self.right_df[self.cols_to_block]

        self.left_df.fillna(' ', inplace=True)
        self.right_df.fillna(' ', inplace=True)

        self.left_df = self.left_df.astype(str)
        self.right_df = self.right_df.astype(str)


        self.left_df["_merged_text"] = self.left_df[self.cols_to_block_without_id].agg(' '.join, axis=1)
        self.right_df["_merged_text"] = self.right_df[self.cols_to_block_without_id].agg(' '.join, axis=1)

        #Drop the other columns
        self.left_df = self.left_df.drop(columns=self.cols_to_block_without_id)
        self.right_df = self.right_df.drop(columns=self.cols_to_block_without_id)
    
    def block_datasets_bert(self, left_df, right_df, cols_to_block):
        self.left_df = left_df
        self.right_df = right_df
        self.cols_to_block = cols_to_block

        self.validate_columns()
        self.preprocess_datasets()

        print("Performing pre-processing for tuple embeddings ")
        all_merged_text = pd.concat([self.left_df["_merged_text"], self.right_df["_merged_text"]], ignore_index=True)
       
        print("Obtaining tuple embeddings for left table")
        left_tuple_embedding = do_bert_embeddings(self.left_df,"_merged_text", self.tokenizer, self.model)
      
        print("Obtaining tuple embeddings for right table")
        right_tuple_embedding = do_bert_embeddings(self.right_df,"_merged_text", self.tokenizer, self.model)

        print("Indexing the embeddings from the right dataset")
        self.vector_pairing_model.index(right_tuple_embedding)

        print("Querying the embeddings from left dataset")
        topK_neighbors = query_embeddings_in_batches(self.vector_pairing_model, left_tuple_embedding, 1000)
        
        self.candidate_set_df = blocking_utils.topK_neighbors_to_candidate_set(topK_neighbors)
    
        return self.candidate_set_df        
        
def query_embeddings_in_batches(model, left_tuple_embeddings, batch_size):
    num_tuples = left_tuple_embeddings.shape[0]
    embeddings_list = []

    # Process data in batches
    for i in range(0, num_tuples, batch_size):
        # Get the current batch of left tuple embeddings
        batch_embeddings = left_tuple_embeddings[i:i+batch_size]

        # Query embeddings using the model
        batch_query_result = model.query(batch_embeddings)

        # Append the query results to the list
        embeddings_list.append(batch_query_result)

    # Concatenate results from all batches
    query_result = np.concatenate(embeddings_list, axis=0)

    return query_result
