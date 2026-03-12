import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import clean_text
from patterns.model_factory import ModelFactory


def train(dataset):

    # Load dataset
    data = pd.read_csv(dataset, sep=";")

    # Remove missing values
    data = data.dropna()

    # Clean text
    data["clean"] = data["text"].apply(clean_text)

    # Feature extraction
    vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)

    X = vectorizer.fit_transform(data["clean"])
    y = data["emotion"]

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Get model from factory
    model = ModelFactory.get_model("naive_bayes")

    # Train model
    model.fit(X_train, y_train)

    # Predictions
    pred = model.predict(X_test)

    # Accuracy
    print("\nModel Accuracy:", accuracy_score(y_test, pred))

    # Detailed evaluation
    print("\nClassification Report:\n")
    print(classification_report(y_test, pred, zero_division=0))

    # -----------------------------
    # Emotion Distribution Graph
    # -----------------------------
    data["emotion"].value_counts().plot(kind="bar")

    plt.title("Emotion Distribution")
    plt.xlabel("Emotion")
    plt.ylabel("Count")

    plt.show()

    # -----------------------------
    # WordCloud Visualization
    # -----------------------------
    text = " ".join(data["clean"])

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")

    plt.title("Most Common Words in Tweets")

    plt.show()

    return model, vectorizer