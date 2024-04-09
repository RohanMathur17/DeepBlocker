import pandas as pd
from deep_blocker import DeepBlocker
from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
import time

# AutoEncoderTuple Embedding
start_time = time.time()
cols_to_block=['name','description','price']

left_df = pd.read_csv("Datasets/Textual/Abt-Buy/tableA.csv", encoding='latin1')
right_df = pd.read_csv("Datasets/Textual/Abt-Buy/tableB.csv", encoding='latin1')

tuple_embedding_model = AutoEncoderTupleEmbedding()
topK_vector_pairing_model = ExactTopKVectorPairing(K=20)

db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)

candidate_set_df = db.block_datasets(left_df, right_df,cols_to_block)
golden_df = pd.read_csv("Datasets/Textual/Abt-Buy/test.csv", encoding='latin1')
golden_df = golden_df[golden_df['label']==1]
print(blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df))
print(candidate_set_df.shape)

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")

# HybridTupleTuple Embedding
start_time = time.time()
cols_to_block=['name','description','price']
left_df = pd.read_csv("Datasets/Textual/Abt-Buy/tableA.csv", encoding='latin1')
right_df = pd.read_csv("Datasets/Textual/Abt-Buy/tableB.csv", encoding='latin1')

tuple_embedding_model = HybridTupleEmbedding()
topK_vector_pairing_model = ExactTopKVectorPairing(K=20)

db = DeepBlocker(tuple_embedding_model, topK_vector_pairing_model)

candidate_set_df = db.block_datasets(left_df, right_df, cols_to_block)
golden_df = pd.read_csv("Datasets/Textual/Abt-Buy/test.csv", encoding='latin1')
golden_df = golden_df[golden_df['label']==1]
print(blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df))
print(candidate_set_df.shape)

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
