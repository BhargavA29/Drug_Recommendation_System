from flask import Flask, render_template, request
from bs4 import BeautifulSoup
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

stop = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

app = Flask(__name__)

vectorizer = joblib.load('model/tfidfvectorizer.pkl')
model = joblib.load('model/passmodel.pkl')


df = pd.read_csv('drugsComTrain_raw.csv')

def top_drugs_extractor(condition, df):
    df_top = df[(df['rating'] >= 9) & (df['usefulCount'] >= 100)].sort_values(by=['rating', 'usefulCount'],ascending=[False, False])
    drug_lst = df_top[df_top['condition'] == condition]['drugName'].head(3).tolist()
    return drug_lst

def review_to_words(raw_review):
    # 1. Delete HTML
    review_text = BeautifulSoup(raw_review, 'html.parser').get_text()
    # 2. Make a space
    letters_only = re.sub('[^a-zA-Z]', ' ', review_text)
    # 3. lower letters
    words = letters_only.lower().split()
    # 5. Stopwords
    meaningful_words = [w for w in words if not w in stop]
    # 6. lemmatization
    lemmatize_words = [lemmatizer.lemmatize(w) for w in meaningful_words]
    # 7. space join words
    return ' '.join(lemmatize_words)

# Function to predict condition and get top drugs
def predict_condition_and_drugs(text, df):
    text = review_to_words(text)
    tfidf_vector = vectorizer.transform([text])
    prediction = model.predict(tfidf_vector)[0]
    top_drugs = top_drugs_extractor(prediction, df)
    return prediction, top_drugs

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/input')
def index2():
    return render_template('input.html')    

@app.route('/about')
def index3():
    return render_template('about.html')   

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text_input = request.form['text_input']
        condition, top_drugs = predict_condition_and_drugs(text_input, df)
        return render_template('predict.html', condition=condition, drugs=top_drugs)

if __name__ == '__main__':
    app.run(debug=True)
