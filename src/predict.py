from src.preprocess import clean_text

def predict_emotion(model, vectorizer, text):

    text = clean_text(text)

    vec = vectorizer.transform([text])

    return model.predict(vec)[0]