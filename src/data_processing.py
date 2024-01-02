# import packages 
import os
import tarfile
import pandas as pd
import warnings
import random

from nltk.tokenize import word_tokenize
from collections import Counter
from tqdm import tqdm

import nltk
nltk.download('punkt')

# ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# importing utility functions
import sys
sys.path.append("utils")
from nlputils import extract_files_from_tgz
from nlputils import process_data_files
from nlputils import keyword_filter



def main():
    # define data paths 
    #zip_data_path = 'zip_data/' - UNCOMMENT FIRST TIME RUNNING 
    data_path = 'data/'

    # unzipping tgz files and extracting only the .txt and .tsv files - UNCOMMENT FIRST TIME RUNNING
    #extract_files_from_tgz(zip_data_path, data_path)
    
    # define paths for data processing
    output_tsv_path = 'output/ParlaMint_tsv.csv'
    output_txt_path = 'output/ParlaMint_txt.csv'
    output_df_path = 'output/ParlaMint_df.csv'
    error_csv_path = 'output/error_log.csv'
    

    # process metadata, speech data, merge to one df and create additional columns in the df
    tsv_dataframe, txt_dataframe, df = process_data_files(data_path, output_tsv_path, output_txt_path, output_df_path, error_csv_path)

    # define path for filtering
    filtered_output_path = 'output/ParlaMint_df_filtered.csv'
    
    # filter on keywords
    keywords = ['climate change', 'emissions', 'CO2', 'fossil fuel', 'carbon footpint', 'carbon dioxide', 'climate']
    filtered_df = keyword_filter(keywords, df, filtered_output_path)



if __name__=="__main__":
    main()