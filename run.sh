#!/usr/bin/env bash

#activate virtual environment
source ./env/bin/activate

# run the code
python3 src/data_processing.py
python3 src/topic_modeling.py
python3 src/visualizations.py

# deactive the venv
deactivate