from flask import render_template, Flask, url_for, request
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # Now we can build our model
from sklearn.multiclass import OneVsRestClassifier  # Binary Relevance
from sklearn.metrics import f1_score  # Performance metric
from sklearn.model_selection import train_test_split
import re
import numpy as np
import pickle
from util import synopsis_analysis
import pandas as pd

# lets call the vectorizing method globally



app = Flask(__name__)


@app.route('/')
def home():
    """we need to call the html page
    for our predictor"""

    return render_template('index.html')


@app.route('/train')
def train():
    """ We use the training data to
    build our model and save the model to
    the output directory, so we can use it
    during prediction, we need to binarize the genre,
    transform target variable"""

    train_data = pd.read_csv(r"C:\Users\Sami\Desktop\genre\input\data\train.csv")
    cleanedata = synopsis_analysis(train_data)

    multilabelbinarizer = MultiLabelBinarizer()
    multilabelbinarizer.fit(cleanedata['new_genre'])

    trainLabels = multilabelbinarizer.transform(cleanedata['new_genre'])

    tfidf_vectorizer = TfidfVectorizer(max_df=0.75, max_features=10000)
    # we can create a TF-IDF vectorization
    trainDataVectorized = tfidf_vectorizer.fit_transform(cleanedata['synopsis_clean'])


    logregressor = LogisticRegression()
    clf = OneVsRestClassifier(logregressor)

    # Now we can fit the model on the training data
    clf.fit(trainDataVectorized, trainLabels)

    model_objects = (clf, tfidf_vectorizer,  multilabelbinarizer)

    # Save tuple
    pickle.dump(model_objects, open("model.pkl", 'wb'))

    return render_template('train.html')


@app.route('/predict')
def predict():
    """ Now we can get the data
    of the movie(synopsis) and predict
    the genre of the movie using the trained model
    """
    # Restore tuple
    pickled_model, tfidf_vectorizer,  multilabelbinarizer = pickle.load(open("model.pkl", 'rb'))
    test_data = pd.read_csv(r"C:\Users\Sami\Desktop\genre\input\data\test.csv")
    cleanedata = synopsis_analysis(test_data)

    valDataVectorized = tfidf_vectorizer.transform(cleanedata['synopsis_clean'])
    # now we can make a prediction on the validation set
    genrePrediction = pickled_model.predict(valDataVectorized)

    # Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
    #submission = pd.DataFrame({'movie_id': test_data['movie_id'], 'Survived': predictions})

    # Visualize the first 5 rows
    #submission.head()
    #prediction = pd.DataFrame([genrePrediction], columns=['genrePrediction']).to_csv('prediction.csv')

    movie_code, prediction_genre, prediction, movie_synopsis = [], [], [], []
    for i in range(0,7169):
        """idx = cleanedata.sample(1).index[0]
        print("Movie id: ", test_data['movie_id'][i],
              "Movie Synopsis: ", test_data['synopsis_clean'][i],
              "\nPredicted genre: ", multilabelbinarizer.inverse_transform(genrePrediction)[i])"""
        movie_code.append(test_data['movie_id'][i])
        genLabels = multilabelbinarizer.inverse_transform(genrePrediction)[i]
        appengen= ' '.join([str(item) for item in genLabels if item is not ','])
        prediction_genre.append(appengen)
        prediction.append(genrePrediction[i])
        movie_synopsis.append(test_data['synopsis_clean'][i])

    submission = pd.DataFrame({'movie_id': movie_code, 'predicted_genres': prediction_genre})
    filename = 'Movie Genre Predictions 1.csv'

    submission.to_csv(filename, index=False)

    print('Saved file: ' + filename)




    #print(multilabelbinarizer.inverse_transform(yPredictor)[0])
    return render_template('result.html')


if __name__ == '__main__':

    app.run()
