# Disaster Response Pipeline Project

## Installation

Extra libraries are compiled in requirement.txt. Just use `pip install -r requirements.txt` and it will install all libraries used in the project.

## Project Motivation

This is a project that is part of the Udacity Data Scientist Nanodegree, which aims to teach us how use data engineering and machine learning pipeline for ETL and prediction, and also how to deploy these pipelines into a webapp.

The dataset consists of real messages that were sent during disaster events, and the task of the ML model is to predict whether each message is disaster related, and which type of disaster does it belong to. So it is a problem of multi label prediction.

The dataset was severly imbalanced in most of the labels, therefore I opted for using a `BalancedRandomForestClassifier` in order to deal with the imbalance by using resampling strategies with each estimator to balance the data.


## File Descriptions

* `data/disaster_messages.csv` contains the original dataset of messages
* `data/disaster_categories.csv` contains the classification labels of messages
* `data/DisasterResponse.db` is an SQL database containing the cleaned data use for ML pipeline.
* The ETL pipeline can be found in `data/process_data.py` 
* The ML pipeline can be found in `models/train_classifier.py`

## What Can You Do With The Project?

To run the webapp you have:
1. run the ETL pipeline that cleans and processes that data use 
`python data/process_data.py data/disaster_messages.py data/disaster_categories.py data/DisasterResponse.db`


2. run the ML pipeline which trains and saves the classifier use 
`python models/train_classifier.py data/DisasterResponse.db models/classifier`


3. Then run the web app run `python run.py` in the commandline and then go to http://0.0.0.0:3001/


* You can improve the results of my model by tweaking hyperparameters or the changing the model itself in `models/train_classifier.py`. 
* You can also use improve upon the ETL pipeline which is found in `data/process_data.py`.


