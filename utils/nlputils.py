
################################################################################
#########################  Importing packages   ################################
################################################################################
import nltk
nltk.download('punkt')

import random
from nltk.tokenize import word_tokenize
from collections import Counter
import pandas as pd 
import numpy as np
import os 

import tarfile
import pandas as pd
import warnings
import random
from tqdm import tqdm
from transformers import pipeline
import torch
# save model
import pickle

# embeddings
from sentence_transformers import SentenceTransformer
# topic model
from bertopic import BERTopic
# dimension reduction
from umap import UMAP
# clustering
from hdbscan import HDBSCAN
# vectorization
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import MaximalMarginalRelevance
from datetime import datetime


################################################################################
#### Function for unzipping. Provide a directory path and an output path  ######
################################################################################

def extract_files_from_tgz(directory_path, output_folder):
    try:
        # iterate through all files in the directory
        for file_name in os.listdir(directory_path):
            if file_name.endswith(".tgz"):
                file_path = os.path.join(directory_path, file_name)
                output_subfolder = file_name.split('.')[0]  # Use the file name without extension as the output subfolder
                output_path = os.path.join(output_folder, output_subfolder)

                # create the subfolder if it doesn't exist
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                # open the .tgz file in read mode
                with tarfile.open(file_path, 'r:gz') as tar:
                    # extract .txt and .tsv files from subfolders
                    for member in tar.getmembers():
                        if member.isdir() and (member.name.endswith(".txt") or member.name.endswith(".tsv")):
                            subfolder_name = os.path.basename(member.name)
                            relevant_files = [m for m in tar.getmembers() if m.name.startswith(f"{subfolder_name}/") and (m.name.endswith(".txt") or m.name.endswith(".tsv"))]
                            tar.extractall(path=output_path, members=relevant_files)
                            break  

    except Exception as e:
        print(f"Error: {e}")


################################################################################
####### Function for processing and merging tsv and txt files to one df ########
################################################################################

def process_data_files(main_folder_path, tsv_output_path, txt_output_path, df_output_path, error_csv_path):
    # initialize empty dataframes
    tsv_dataframe = pd.DataFrame()
    txt_dataframe = pd.DataFrame()
    df = pd.DataFrame()  
    
    # initialize empty error log
    error_log = []

    try:
        # loop through main folders
        for folder in os.listdir(main_folder_path):
            folder_path = os.path.join(main_folder_path, folder)

            # check if the item in the main folder is a directory (to skip files)
            if os.path.isdir(folder_path):

                # loop through the country directory within the main folder
                for country in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
                    country_path = os.path.join(folder_path, country)

                    # check if the item in the country folder is a directory
                    if os.path.isdir(country_path):

                        try:
                            year = 2015
                            while year <= 2022:
                                year_path = os.path.join(country_path, str(year))

                                if os.path.exists(year_path):
                                    # loop through files in the year subfolder
                                    for filename in os.listdir(year_path):
                                        filepath = os.path.join(year_path, filename)

                                        # check if the file is a tsv file
                                        if filename.endswith('.tsv'):
                                            tsv_df = pd.read_csv(filepath, sep='\t')
                                            tsv_dataframe = tsv_dataframe.append(tsv_df, ignore_index=True)

                                        # check if the file is a txt file
                                        elif filename.endswith('.txt'):
                                            df_txt = pd.read_csv(filepath, sep='\t', header=None)
                                            txt_dataframe = txt_dataframe.append(df_txt, ignore_index=True)


                                year += 1  # move on to the next year

                        except Exception as e: 
                            print(f"Error processing {filename}: {e}")
                            error_log.append((filename, str(e)))


        # specify txt columm names
        txt_dataframe = txt_dataframe.rename(columns={0: 'ID', 1: 'Speech'})

        # merge dataframes on 'ID' column using inner join
        df = pd.merge(tsv_dataframe, txt_dataframe, on='ID', how='inner')

        # create country code column
        df['Country_Code'] = df['Text_ID'].str.split('-', expand=True)[1].str.split('_', expand=True)[0]

        # create year column
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year

    except Exception as e:
        print(f"Error: {e}")

    # save metadata, speeches and the combined data as csv files
    tsv_dataframe.to_csv(tsv_output_path, index = False)
    txt_dataframe.to_csv(txt_output_path, index = False)
    df.to_csv(df_output_path, index = False)

    # save error log
    error_df = pd.DataFrame(error_log, columns=['Filename', 'Error Message'])
    error_df.to_csv(error_csv_path, index=False)
    
    return tsv_dataframe, txt_dataframe, df



################################################################################
##########   Function for filtering speeches on specified keywords   ###########
################################################################################

def keyword_filter(keywords, df, filtered_output_path):
    filtered_df = df[df['Speech'].str.contains('|'.join(keywords), case=False)]
    
    # saving filtered dataframe
    filtered_df.to_csv(filtered_output_path, index = False)

    return filtered_df 


################################################################################
###################   Function for making embeddings    ########################
################################################################################

def make_embeddings(df):
  embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
  embeddings = embedding_model.encode(df, show_progress_bar=True)
  # save the embeddings 
  np.save("output/embeddings", embeddings)

  return embedding_model, embeddings


################################################################################
###########   Function for creating frequnecy based stopword list   ############
################################################################################

def frq_stopword(df, stopword_output_path):
    # get unique country codes from the 'Country_Code' column
    unique_countries = df['Country_Code'].unique()

    # set the desired percentage for each country
    percentage = 10
    
    # randomly sample speeches from each country based on the desired percentage
    sampled_speeches = []
    for country in unique_countries:
        country_speeches = df.loc[df['Country_Code'] == country, 'Speech'].tolist()
        num_speeches_to_sample = int(len(country_speeches) * (percentage / 100))
        sampled_speeches.extend(random.sample(country_speeches, num_speeches_to_sample))

    # tokenization and frequency analysis for sampled speeches
    all_words_sampled = [word.lower() for speech in sampled_speeches for word in word_tokenize(speech)]
    word_freq_sampled = Counter(all_words_sampled)

    num_words_to_filter = 187  # CAN BE ADJUSTED 
    frq_stopword_list = [word for word, _ in word_freq_sampled.most_common(num_words_to_filter)]

    # save stopword list
    with open(stopword_output_path, 'w') as file:
        for item in frq_stopword_list:
            file.write(str(item) + '\n')
    
    return frq_stopword_list


################################################################################
###  Function for combining freq based stopword list with custom stop words  ###
################################################################################

def combine_stopwords(frq_stopword_list, stopword_output_path):
 # our custom list of stopwords
  custom_stopwords = ["lords", "lord", "noble", "re", "lady", "uk", 
                      "applause", "ladies", "gentlemen", "Austria", "austrian", 
                      "galicia", "basque", "Bosnia", "bosnian", "Herzegovina", 
                      "herzegovian", "Belgium", "belgian", "Bulgaria", "bulgarian", 
                      "Czechia", "czech", "Denmark", "danish", "Estonia","estonian", 
                      "Spain", "spanish", "Finland", "finnish", "France", "french", 
                      "Great Britain", "United Kingdom", "british", "Greece", "greek", 
                      "Croatia", "croatian", "Hungary", "hungarian", "Iceland", "icelandic", 
                      "Italy", "italian", "Latvia", "latvian", "Netherlands", "dutch", "Norway", 
                      "norwegian", "Poland", "polish", "Serbia", "serbian", "Sweden","swedish", 
                      "Slovenia", "slovenian", "Turkey", "turkish", "Ukraine", "ukrainian", 
                      "portugal", "portuguese"]
  # converting to a set to combine
  custom_stopwords = set(custom_stopwords)
  en_stopwords_list = set(frq_stopword_list)  
  stop_words = list(map(str.lower, en_stopwords_list.union(custom_stopwords)))
  # save list
  with open(stopword_output_path, 'w') as file:
    for item in stop_words:
        file.write(str(item) + '\n')

  return stop_words


################################################################################
#####################    Function for generating labels   ######################
################################################################################


def label_generator(df, output_csv_path):
    # define the text generation pipeline
    pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-alpha", torch_dtype=torch.bfloat16, device_map="auto")

    # initialize empty list 
    generated_labels = []

    for keywords_list in df['MMR']:
        # convert the list to a string if it's not already
        keywords_str = ' '.join(map(str, keywords_list))

        # create the prompt using the keywords
        messages = [
            {"role": "system", "content": "You are a helpful, respectful and honest assistant for labeling topics."},
            {"role": "user", "content": f"I have a list of keywords that represent a topic. The topic is described by the following keywords: {keywords_str}. Based on the information about the topic, please create a meaningful, informative label of at most three words for this topic. Make sure you to only return the label and nothing more."},
        ]

        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

        # extract the generated label from the output
        generated_label = outputs[0]["generated_text"]

        # append the generated label to the list
        generated_labels.append(generated_label)

    # create a column in the df for this created list 
    df['generated_label'] = generated_labels

    # define a function to extract the label
    def extract_label(generated_label):
        # use regex to find everything between the last '\n' and the end of the string
        label = pd.Series(generated_label).str.extract(r'\n(.*?$)', expand=False)

        # remove double quotes if present
        if label is not None and not label.empty:
            label = label.str.replace('"', '')

        return label

    # apply the function to create the 'label' column
    df['label'] = df['generated_label'].apply(extract_label)

    # save df including label column
    df.to_csv(output_csv_path, index=False)

    return df


################################################################################
#####################  Function to clean for visualzations   ###################
################################################################################

def clean_for_vis(topic_table, parlamint_df):
   # manually labelling our categories  
    indices_to_labels = {
        0: 'irrelevant',
        1: 'transport',
        2: 'food production',
        45: 'food production',
        43: 'transport',
        3: 'emissions',
        4: 'energy',
        5: 'energy',
        6: 'energy',
        13: 'energy',
        16: 'energy',
        33: 'energy',
        38: 'energy',
        42: 'energy',
        47: 'energy',
        48: 'energy',
        7: 'extreme weather',
        25: 'extreme weather',
        39: 'extreme weather',
        40: 'extreme weather',
        8: 'democracy',
        9: 'democracy',
        37: 'democracy',
        11: 'international climate negotiations',
        12: 'international climate negotiations',
        17: 'international climate negotiations',
        21: 'international climate negotiations',
        14: 'energy',
        15: 'economy',
        18: 'citizen protection',
        19: 'economy',
        20: 'global challenges',
        22: 'global challenges',
        34: 'global challenges',
        36: 'global challenges',
        46: 'global challenges',
        43: 'economy',
        29: 'biodiversity',
        31: 'education',
        52: 'steel sector',
        10: 'climate policy',
        26: 'climate policy',
        30: 'climate policy',
        32: 'climate policy',
        23: 'irrelevant',
        24: 'recycling',
        27: 'forestation',
        28: 'irrelevant',
        35: 'irrelevant',
        41: 'irrelevant',
        44: 'economy',
        49: 'forestation',
        50: 'EU politics',
        51: 'global challenges'
    }

    # create a new column 'human_labels' and set labels based on row indices
    topic_table['human_labels'] = topic_table.index.map(indices_to_labels)

    # manually labeling each country with it's code 
    country_code_mapping = {
        'GB': 'United Kingdom',
        'BA': 'Bosnia and Herzegovina',
        'PL': 'Poland',
        'CZ': 'Czech Republic',
        'ES': 'Spain',
        'BG': 'Bulgaria',
        'GR': 'Greece',
        'DK': 'Denmark',
        'PT': 'Portugal',
        'HU': 'Hungary',
        'FI': 'Finland',
        'TR': 'Turkey',
        'SI': 'Slovenia',
        'RS': 'Serbia',
        'EE': 'Estonia',
        'AT': 'Austria',
        'IS': 'Iceland',
        'BE': 'Belgium',
        'SE': 'Sweden',
        'UA': 'Ukraine',
        'NO': 'Norway',
        'HR': 'Croatia',
        'IT': 'Italy',
        'FR': 'France',
        'LV': 'Latvia',
        'NL': 'Netherlands'
    }

    parlamint_df['Country'] = parlamint_df['Country_Code'].map(country_code_mapping)


    return topic_table, parlamint_df


