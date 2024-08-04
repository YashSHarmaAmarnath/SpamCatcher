from flask import Flask, request, render_template
import joblib
import string
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

vectorizer = joblib.load('vectorizer_model.joblib')
clf = joblib.load('model.joblib')

stemmer = PorterStemmer()
stopword_set = set(stopwords.words('english'))

def text_vector(text_to_convert):
    email_text = text_to_convert.lower()
    email_text = email_text.translate(str.maketrans('', '', string.punctuation)).split()
    email_text = [word.replace('\r\n', ' ') for word in email_text]
    email_text = [stemmer.stem(word) for word in email_text if word not in stopword_set]
    email_text = ' '.join(email_text)
    return vectorizer.transform([email_text])

@app.route('/', methods=['GET'])
def hello():
    return render_template("index.html", spam=None)

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    op = clf.predict(text_vector(text))[0]
    return render_template("index.html", spam=op,text=text)

if __name__ == '__main__':
    app.run(debug=True)
