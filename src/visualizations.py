import re
import pandas as pd
# to load the saved model
import pickle
from bertopic import BERTopic
# to save the interactive plot 
import plotly.offline as pyo

# importing utility functions
import sys
sys.path.append("utils")

from nlputils import clean_for_vis


def main():
    # load in filtered data 
    parlamint_df = pd.read_csv('output/ParlaMint_df_filtered.csv')

    # specify data (speeches) for the model 
    speech_data = parlamint_df['Speech']

    # empty dataframe for timestamps and speeches 
    df = pd.DataFrame()

    # loading in the saved topic table 
    topic_table = pd.read_csv('output/topic_table.csv')

    # adding columns we need for visualizing, our categories and country name connected to country code 
    topic_table, parlamint_df = clean_for_vis(topic_table, parlamint_df)

    df['timestamps'] = parlamint_df.Date.to_list()
    df['speeches'] = speech_data.to_list()

    speeches = df['speeches']
    timestamps = df['timestamps']

    custom_labels = topic_table.human_labels.to_list()

    # loading in the model 
    pickle_file_path = 'models/bertopic_model.pkl'
    with open(pickle_file_path, 'rb') as file:
        loaded_model = pickle.load(file)

    loaded_model.set_topic_labels(custom_labels)

    # look at topics over time 
    topics_over_time = loaded_model.topics_over_time(speeches, timestamps, datetime_format="%Y-%m-%d", nr_bins=20)

    # visualize topics over time with custom labels
    fig_time = loaded_model.visualize_topics_over_time(topics_over_time, 
                                                       custom_labels=True, 
                                                       topics=[0, 1, 2, 3, 6, 9, 14, 19])

 

    # topics by class
    classes = parlamint_df['Country']

    topics_per_class = loaded_model.topics_per_class(speeches, classes=classes)

    fig_classes = loaded_model.visualize_topics_per_class(topics_per_class, 
                                            custom_labels= True, 
                                            topics=[0, 1, 2, 3, 6, 9, 14, 19])

    # customize colors 
    custom_colors = ["#1f5eb4", "#b4eeb4", "#00ced1", "#f6546a", "#cbbeb5", "#6897bb", "#008000", "#ffc0cb"]

    # iterate through to update with custom colors for the fig_time plot 
    for i, trace in enumerate(fig_time.data):
        trace.marker.color = custom_colors[i]

    # ppdate layout to show the legend
    fig_time.update_layout(showlegend=True)

    # # iterate through to update with custom colors for the fig_classes plot 
    for i, trace in enumerate(fig_classes.data):
        trace.marker.color = custom_colors[i]

    # update layout to show the legend
    fig_classes.update_layout(showlegend=True)

    # save the interactive plots to an HTML file
    pyo.plot(fig_time, filename='figures/topics_over_time.html')
    pyo.plot(fig_classes, filename='figures/topics_across_class.html')
  


if __name__=="__main__":
    main()
