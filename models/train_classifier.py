import sys
import pandas as pd
import numpy as np

import nltk 
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sqlalchemy import create_engine

from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, make_scorer, classification_report

from skmultilearn.model_selection import IterativeStratification
from skmultilearn.model_selection import iterative_train_test_split

from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

from imblearn.ensemble import BalancedRandomForestClassifier

seed = 42


def load_data(database_filepath):
    # load data from database
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('clean', engine)
    X = df['message']
    y = df.iloc[:, 4:]

    return X, y, y.columns.tolist()


def tokenize(text):
    tokens = []
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    for token in word_tokenize(text.lower()):
        if token.isalpha() and not token in stopwords.words('english'):
            token = lemmatizer.lemmatize(token)
            token = stemmer.stem(token)
            tokens.append(token)
    return tokens

def iterative_train_test_split(X, y, train_size):
    """Custom iterative train test split which
    'maintains balanced representation with respect
    to order-th label combinations.'
    """
    stratifier = IterativeStratification(
        n_splits=2, order=1, sample_distribution_per_fold=[1.0-train_size, train_size, ])
    train_indices, test_indices = next(stratifier.split(X, y))
    X_train, y_train = X[train_indices], y[train_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def build_model(X_train, y_train):
    """Function to find best model parameters using sklearn's GridSearchCV.'"""
    pipeline = Pipeline([
        ('vect', CountVectorizer(ngram_range=(1, 1))),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(BalancedRandomForestClassifier(random_state=seed)))
    ])
    
    parameters = {
    'clf__estimator__n_estimators': [50],
    'clf__estimator__max_features': ['auto', 0.3],
    }
#    scorer = make_scorer(f1_score, average = 'weighted')
    mskf = MultilabelStratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    cv = GridSearchCV(pipeline, parameters, cv=mskf, verbose=3, scoring='f1_weighted')
    cv.fit(X_train, y_train)
    
    print("Best Parameters:", cv.best_params_, "\nBest Score:", cv.best_score_)
    return pipeline.set_params(**cv.best_params_)



def evaluate_model(model, X_test, y_test, category_names):
    """
    Function to evaluate model using classification report separately on labels, and 
    major scores weighted by label support on all labels such as precision, recall and accuracy.
    """
    y_pred = model.predict(X_test) 
    
    # change all child_alone predicted labels to 0
    y_pred[:, 13] = 0
    
    print('X_test shape:', X_test.shape)
    print('y_test shape:', y_test.shape)
    print('y_pred shape:', y_pred.shape)
    
    print(classification_report(y_test, y_pred, target_names=category_names))
    
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)

    print('Precision:', precision)
    print('Recall:', recall)
    print('Accuracy:', accuracy)


def save_model(model, model_filepath):
    """Function to save model using pickle to a specified filepath."""
    pickle.dump(model, open(model_filepath, 'wb')) 


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        
        X_train, X_test, y_train, y_test = iterative_train_test_split(X.values, y.values, train_size=0.8)
        
        print('Building model...')
        model = build_model(X_train, y_train)
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
