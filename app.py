import joblib
from flask import Flask, request, render_template
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

# Load trained model and tools
model = joblib.load("random_forest_sentiment.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

def preprocess(text):
    return text  # You can add cleaning steps here later

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None

    if request.method == 'POST':
        product = request.form.get('product', '')
        review = request.form.get('review', '')

        if not review.strip():
            prediction = "Please enter a review."
        else:
            combined = preprocess(product + " " + review)
            vector = tfidf.transform([combined])
            pred = model.predict(vector)[0]
            prediction = label_encoder.inverse_transform([pred])[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)

   