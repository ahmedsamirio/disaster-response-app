import json
import plotly
import pickle
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """Function to tokenize text and remove stopwords and punctuation marks"""
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in stopwords.words('english')]
    return tokens

def most_common_words(tokens, n=30):
    """Function to get n most common words in text."""
    # make an nltk frequency distribution of tokens
    fdist = nltk.FreqDist(tokens)
    return fdist.most_common(n)
    
def find_collocations(tokens, n=30):
    """Function that returns a series the n highest score bigram collocations."""
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    collocations = finder.score_ngrams(bigram_measures.likelihood_ratio)[:n]
    
    # join bigrams into one str
    collocations = [('-'.join(bigrams), score) for bigrams, score in collocations]
    top_n = pd.Series(dict(collocations))  
    return top_n

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('clean', engine)

# load model
model = pickle.load(open("../models/classifier.pkl", "rb"))


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    tokens = tokenize(' '.join(df['message']))
    words_counts = most_common_words(tokens)
    words, counts = zip(*words_counts)
    
    collocations = find_collocations(tokens)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=words,
                    y=counts,
                )
            ],
            
            'layout': {
                'title': 'Top 30 Words (Excluding Stopwords)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=collocations.index,
                    y=collocations,
                )
            ],
            
            'layout': {
                'title': 'Top 30 Collocations (Bigrams with High Likelihood Despite Low Token Frequency)',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Collocation"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    
    # remove child-alone label
    classification_labels[13] = 0
    
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
