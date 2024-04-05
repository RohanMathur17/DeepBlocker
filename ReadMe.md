# DeepBlocker
In the ever-changing and constantly evolving field of Entity-Matching
(EM), the use of language and deep learning models have been highly
emphasized in enhancing the data processing phases, particularly
the blocking phase. This research project delves deep into the subtle
yet crucial domain of Entity-Matching (EM), with a special focus on
the blocking phase in the vast area of data processing. To make significant improvements in the blocking stage of EM tasks, we present
a novel method that makes use of pre-trained Turkish languag 
models as well as other prominent multi-lingual models.

# Paper and Data
This project is inspired by the Paper - [Deep Learning for Blocking in Entity Matching: A Design Space Exploration](https://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)

Thirumuruganathan's study on deep learning for blocking in
EM is what motivates this particular technique which primarily
looks at the potential of applying deep learning strategies to improve the EM processes. We aim to broaden this investigation
by incorporating pre-trained and multi-lingual models for dealing
with the distinctive problems presented by the Turkish language
datasets. The techniques employed in this project are inspired by
a comparative analysis that includes several baselines such as a
transfer-learning-based language model (not pre-trained in the
Turkish language); a non-linguistic model being a deep learning
solution; and finally, a traditional non-deep-learning model. An
attempt at comparative analysis, this paper tries to establish the
effect of pre-training for language-specificity on blocking stages’
performance and efficacy in the EM domain.

# Requirements

- Benchmark datasets for blocking can be found [here](https://github.com/anhaidgroup/deepmatcher/blob/master/Datasets.md). Note that these are already arranged and downloaded in the ```deepblocker/Datasets/``` folder.
- New Turkish dataset that this project will work upon can be found [here](https://github.com/FurkanGozukara/Record-Linkage/tree/master). Note that the required datasets are already arranged and downloaded in the ```deepblocker/Datasets/Turkish/``` folder.
- To run this repository, you need to clone the repository to your local machine using Git. Use the following command in your terminal or command prompt:
  - ```git clone https://github.com/RohanMathur17/DeepBlocker.git```
  - ```cd Deepblocker```
- Users would be required to install the necessary requirements and dependencies. This can be done by running the following command -
  - ```cd deepblocker```
  - ```pip install -r requirements.txt```
- To run the benchmarking and test out the above datasets, user would require word vectors from [```fastText.```](https://fasttext.cc/) User would be required to download two sets of vectors (English and Turkish) in the ```deepblocker/``` directory.
  - ```curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip```
  - ```unzip wiki.en.zip```
  -  ```curl -O https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.tr.zip```
  -   ```unzip wiki.tr.zip```
 
# How to Run

## Replicate Original Paper

- To first replicate the original paper's evaluation from the benchmark datasets, user would be required to run the different ```.py``` files for each dataset. Note that each ```.py``` file runs the following -
  - ```AutoEncoderTuple Embedding Model```
  - ```HybridTuple Embedding Model```
    
- For each dataset in the ```Datasets/Structured/```, ```Datasets/Textual/```, ```Datasets/Dirty/``` subdirectory, each of the ```.py``` (total eight scripts for eight datasets) files are generated to allow smoother running on lower-end machines. An example of how to run is below -
   - ```python3 main_structured_dblp_acm.py```

## Preprocess New Turkish Baseline Dataset

- To evaluate the new Turkish dataset, you would be required to preprocess the existing raw dataset in ```Datasets/Turkish/Only_Price_Having_Products/needed/```. While the existing preprocessed data already exists in this repository, to recreate them users can run the below scripts -
  - ```python3 turk_preprocessing.py Datasets/Turkish/Only_Price_Having_Products/needed/```
    - This script will be able to merge all the ```.txt``` files in the subdirectory with all the columns matched for each ID.
  - ```python3 split_merged.py Datasets/Turkish/merged.csv```
    - This script will create ```tableA.csv``` and ```tableB.csv``` as required by the original code. This way, the dataset is split into two different sources (Table A and Table B), which will be used for blocking purposes.
  - ```python3 turk_making_pairs.py Datasets/Turkish/Training/tableA.csv Datasets/Turkish/Training/tableB.csv```
    - This script will make ```product_id_pairs.csv``` and ```random_product_id_non_pairs.csv```. These will be used to create our evaluation test sets.
  - ```python3 make_test.py Datasets/Turkish/product_id_pairs.csv Datasets/Turkish/random_product_id_non_pairs.csv```
    - This script will generate the test sets - ```test_100.csv```, ```test_150.csv``` and ```test_200.csv```.

## Run DeepBlocker Embedding Models on New Turkish Datasets
- To run DeepBlocker embeddings on the new turkish datasets, user will have to run the following -
  - ```python3 main_turk.py```
- Note that to utilize and toggle between the different Word Embedding Vectors which we downloaded, users would be required to change the ```configurations.py``` file. Example for this is -
  - Currently the first line is set to - ```FASTTEXT_EMBEDDIG_PATH = "wiki.tr.bin"```. This can be changed to - ```FASTTEXT_EMBEDDIG_PATH = "wiki.en.bin"``` to utilize English Vectors.
- Also, if the users want to experiment with multiple ```K value```, the ```main_turk.py``` file can be modified accordingly.

## Run BERT and BERTTurk Models on New Turkish Datasets
- To run BERT and BERTTurk on the new turkish datasets, user will have to run the following -
  - ```python3 main_bert.py```
- Also, if the users want to experiment with multiple ```K value```, the ```main_turk.py``` file can be modified accordingly.



# Contributors
- This project is part of the final project for CMPT 984 (Special Topics in Databases), as part of the Spring 2024 Semester at Simon Fraser University.
  
| **Name** | **Student ID** | **Email** |
|--------------|--------------|--------------|
| Hardev Khandar| 301543441| hmk9@sfu.ca|
| Rohan Mathur| 301544232| rma135@sfu.ca |


