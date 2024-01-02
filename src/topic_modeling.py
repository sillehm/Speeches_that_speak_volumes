# load packages
import os
import pandas as pd
import numpy as np
import warnings

# Embeddings
from sentence_transformers import SentenceTransformer
# Topic model
from bertopic import BERTopic
# Dimension reduction
from umap import UMAP
# Clustering
from hdbscan import HDBSCAN
# Vectorization
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from transformers import pipeline
from datetime import datetime
import torch
# save model
import pickle

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# importing utility functions
import sys
sys.path.append("utils")

from nlputils import make_embeddings
from nlputils import frq_stopword
from nlputils import combine_stopwords
from nlputils import label_generator


def main():
  # define paths 
  stopword_output_path = 'output/stopwords_list.csv'
  output_csv_path = 'output/topic_table.csv'
  #file path for saving the model as a pickle file
  pickle_file_path = "models/bertopic_model.pkl"

  # load unfiltered data
  df = pd.read_csv('output/ParlaMint_df.csv')
  # load filtered data
  parlamint_df = pd.read_csv('output/ParlaMint_df_filtered.csv')

  # specify data for the model i.e. speeches in our case
  data = parlamint_df['Speech']

  # calculate embeddings
  embedding_model, embeddings = make_embeddings(data)

  # initiate UMAP
  umap_model = UMAP(n_neighbors=15, 
                    n_components=5, 
                    min_dist=0.0, 
                    metric='cosine', 
                    random_state=100)

  # cluster reduced embeddings
  hdbscan_model = HDBSCAN(min_cluster_size=150, metric='euclidean', 
                          cluster_selection_method='eom', prediction_data=True)

  # frquency stopword list 
  frq_stopword_list = frq_stopword(df, stopword_output_path)

  # combine custom and freqency stopwords lists 
  stop_words = combine_stopwords(frq_stopword_list, stopword_output_path)

  # using stopword list with vectorizer 
  vectorizer_model = CountVectorizer(stop_words=stop_words, min_df=30, ngram_range=(1, 3)) 
  # representation model
  mmr_model = MaximalMarginalRelevance(diversity=0.3)
  representation_model = {"MMR": mmr_model}

  # initialising topic model 
  topic_model = BERTopic(

    # pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model, 

    # hyperparameters
    top_n_words=10,
    verbose=True
  )

  # train model
  topics, probs = topic_model.fit_transform(data, embeddings)

  # show topics
  topic_table = topic_model.get_topic_info()

  # generating labels and saving table with topics 
  topic_table = label_generator(topic_table, output_csv_path)

  # save the BERTopic model using pickle
  with open(pickle_file_path, 'wb') as file:
      pickle.dump(topic_model, file)



if __name__=="__main__":
    main()