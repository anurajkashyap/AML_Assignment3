from flask import Flask, request, render_template, url_for, redirect
import pickle
import score

app = Flask(__name__)


with open('logistic_regression.pkl', 'rb') as file:
    model = pickle.load(file)

threshold = 0.5



'''def home():
    return render_template('spam_page.html')'''


@app.route("/")
def spam():
    # Get the input text from the form
    txt = request.form['sent']
    
    # Get the prediction and propensity score using the pre-trained model
    pred,prop = score.score(txt, model, threshold)
    
    # Determine the label based on the prediction
    label = "Spam" if pred == 1 else "Not spam"
    
    # Generate the response message
    ans = f"""The sentence "{txt}" is {label} with propensity {prop}."""
    
    # Render the result page with the response message
    return render_template('result_page.html', ans = ans)

if __name__ == '__main__': 
    # Run the Flask app
    app.run(debug=True, use_reloader=True)