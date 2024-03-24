def text_processing(text):
    import re
    import nltk
    # Convert text to lowercase
    text = text.lower()

    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    text = nltk.word_tokenize(text)

    y = []
    for token in text:
        if token not in nltk.corpus.stopwords.words('english'):
            y.append(token)
    
    return " ".join(y[1:])