import pandas as pd
from flask import Flask, jsonify, request
import pickle
import joblib as jl

# load model
pac = jl.load('pac.pkl.z')
tfidf_vectorizer = jl.load('tfidf_vectorizer.pkl.z')

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])
def predict():
    # get data
    data = request.get_data()

    #data_df = pd.DataFrame({'text' : [data]})

    # predictions
    tfidf = tfidf_vectorizer.transform([data])
    result = pac.predict(tfidf)

    # return data
    return result[0]

if __name__ == '__main__':
    app.run(port = 5000, debug=True)