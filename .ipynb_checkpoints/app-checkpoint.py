from flask import Flask, request, jsonify, render_template
import pickle as pkl
from text_processing import text_processing

app = Flask(__name__)

with open('logistic_regression.pkl', 'rb') as file:
    model = pkl.load(file)
with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorizer = pkl.load(file)
 

def predict_text(text):
    # We assume the text is already in the desired format for the model
    text_features = vectorizer.transform([text_processing(text)]) 
    print(model)
    prediction = model.predict(text_features)
    propensity = model.predict_proba(text_features)[0][1]  # Probability of the spam class
    return {"prediction": str(prediction[0]), "propensity": float(propensity)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/score', methods=['POST'])
def score():
    data = request.json
    text = data.get('text', '')
    result = predict_text(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)