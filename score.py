def score(text, model, threshold):

    import text_processing as txt
    import pickle

    processed_text = txt.text_processing(text)

    with open('tfidf_vectorizer.pkl', 'rb') as file:
        vectorizer = pickle.load(file)

    vectorized_text = vectorizer.transform([processed_text])

    #with open('logistic_regression.pkl', 'rb') as file:
    #    model = pickle.load(file)

    probabilities = model.predict_proba(vectorized_text)[0]

    propensity = probabilities[1]

    prediction = bool(propensity > threshold)

    return prediction, propensity
