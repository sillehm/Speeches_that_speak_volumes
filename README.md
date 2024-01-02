# Speeches that speak volumes: Uncovering Climate Change Themes in European Parliament

This repository is the product for the NLP exam project, Fall 2023, by authors Anja Feibel Meerwald and Sille Hasselbalch Markussen.  

## The data

The dataset is a collection of parliamentary speeches from across Europe which can be found and downloaded [here](https://www.clarin.si/repository/xmlui/handle/11356/1864).


# Repository 

| Folder         | Description          
| ------------- |:-------------:
| data   | This folder is hidden due to the size of the dataset (but should be added to run the scripts)  
| zip_data  | This is where the downloaded zipped data should be placed, also hidden due to size 
| figures  | Visual representations of the data       
| output  |  Contains error log, stopword list and table of resulting topics. Tsv, txt, filtered and unfiltered dataframes will appear here after code is run, but are hidden due to size 
| models  | The saved topic model is here    
| src  | Py scripts 
| Uutils  | Functions used for the various py scripts        


## To run the scripts 

As the dataset is too large to store in the repo, use the link above to access the data. Download the data and create a folder called  ```zip_data``` within the Speeches_that_speak_volumes folder, along with the other folders in the repo. If it is the first time running the scripts, a couple lines should be unhastagged (they are marked, "UNCOMMENT FIRST TIME RUNNING") to unzip the data files. Then the code will run without making any changes. If the downloaded data is placed elsewhere, then the path should be updated in the code.

1. Clone the repository, either on ucloud or something like worker2.
2. From the command line, at the /Speeches_that_speak_volumes/ folder level, run the following lines of code. 

This will create a virtual environment and install the correct requirements.
``` 
bash setup.sh
```
While this runs the scripts and deactivates the virtual environment when it is done. 
```
bash run.sh
```

The scripts can also be run individually if that is preferred, given that they may take a few hours to run all together. 

This is done by running each script from the command line (see below)
```
python3 src/data_processing.py
python3 src/topic_modeling.py
python3 src/visualizations.py
```

This has been tested on an ubuntu system on ucloud and therefore could have issues when run another way.

