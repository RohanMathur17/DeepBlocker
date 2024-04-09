import pandas as pd
from deep_blocker import DeepBlocker
from blocker_bert import BlockerBert
from transformers import BertTokenizer, BertModel

from tuple_embedding_models import  AutoEncoderTupleEmbedding, CTTTupleEmbedding, HybridTupleEmbedding
from vector_pairing_models import ExactTopKVectorPairing
import blocking_utils
import time
import sys

start_time = time.time()

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Load pre-trained BERTurk model and tokenizer
bert_turk_tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-base-turkish-uncased")
bert_turk_model = BertModel.from_pretrained("dbmdz/bert-base-turkish-uncased")

turkish_cols_to_block = ["Raw_Entire_Dataset","Raw_Product_Categories","Original_Crawled_URLs"]
left_df = pd.read_csv("Datasets/Turkish/Training/tableA.csv", encoding='latin1')
right_df = pd.read_csv("Datasets/Turkish/Training/tableB.csv", encoding='latin1')
golden_df = pd.read_csv("Datasets/Turkish/Training/test_100.csv", encoding='latin1')
golden_df = golden_df[golden_df['label']==1]

def run_bert(cols_to_block, left_df, right_df, golden_df, bert_tokenizer, bert_model):
    
    topK_vector_pairing_model = ExactTopKVectorPairing(K=20)
    db = BlockerBert(topK_vector_pairing_model, bert_tokenizer, bert_model)
    candidate_set_df = db.block_datasets_bert(left_df, right_df, cols_to_block)
    print(blocking_utils.compute_blocking_statistics(candidate_set_df, golden_df, left_df, right_df))
    print(candidate_set_df.shape)

print("Running English BERT...")
run_bert(turkish_cols_to_block, left_df, right_df, golden_df, bert_tokenizer, bert_model)
print("Done English BERT!")

print("Running Turkish BERT...")
run_bert(turkish_cols_to_block, left_df, right_df, golden_df, bert_turk_tokenizer, bert_turk_model)
print("Done Turkish BERT!")

end_time = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time} seconds")
