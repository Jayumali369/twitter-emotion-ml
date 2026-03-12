from src.train_model import train
from src.predict import predict_emotion

model, vectorizer = train('data/dataset.csv')

while True:

    tweet = input("Enter tweet: ")

    emotion = predict_emotion(model, vectorizer, tweet)

    print("Emotion:", emotion)