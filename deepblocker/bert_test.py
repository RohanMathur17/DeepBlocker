import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
from transformers import BertTokenizer, BertModel

# Function to tokenize text and obtain BERT embeddings for the entire sentence
def get_sentence_embeddings(tokenizer, model, text, pooling_method='mean'):
    # Tokenize text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Obtain embeddings from BERTurk model
    with torch.no_grad():
        outputs = model(**inputs)
    # Extract word embeddings from BERTurk output
    embeddings = outputs.last_hidden_state
    # Apply pooling operation to obtain sentence-level embeddings
    if pooling_method == 'mean':
        sentence_embeddings = torch.mean(embeddings, dim=1)  # Mean pooling
    elif pooling_method == 'max':
        sentence_embeddings, _ = torch.max(embeddings, dim=1)  # Max pooling
    else:
        raise ValueError("Pooling method must be 'mean' or 'max'.")
    # Return sentence embeddings
    return sentence_embeddings.squeeze(dim=1)  # Squeeze singleton dimension

def do_bert_embeddings(df, column, tokenizer, model):
    # Apply the function to obtain BERT embeddings for the entire sentence
    embeddings_list = [get_sentence_embeddings(tokenizer, model, text) for text in df[column]]

    # Stack the embeddings into a single tensor
    sentence_embeddings = torch.stack(embeddings_list)

    # Ensure the desired shape (2, 768)
    sentence_embeddings = sentence_embeddings.squeeze(dim=1)
    
    numpy_array = sentence_embeddings.numpy()
    
    return numpy_array 
