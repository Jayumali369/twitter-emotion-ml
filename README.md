# Twitter Emotion Classification using Machine Learning

#Project Structure

twitter_emotion_project/
│
├── data/
│   └── dataset.csv
│
├── src/
│   ├── preprocess.py
│   ├── train_model.py
│   └── predict.py
│
├── patterns/
│   └── model_factory.py
│
├── main.py
└── README.md

This project analyzes Twitter comments and classifies emotions using Machine Learning.

## Emotions Detected
- Joy
- Anger
- Sadness
- Fear
- Love
- Surprise

## Technologies Used
- Python
- Scikit-learn
- NLP (Natural Language Processing)
- TF-IDF Vectorization
- Naive Bayes Classifier
- Matplotlib
- WordCloud

## Project Workflow

Tweet Text
↓
Text Preprocessing
↓
TF-IDF Feature Extraction
↓
Machine Learning Model
↓
Emotion Prediction
↓
Visualization

## Dataset

The dataset contains tweets labeled with emotions such as joy, anger, sadness, fear, love, and surprise.

## Run the Project

Install dependencies:

pip install pandas scikit-learn nltk matplotlib wordcloud

Run:

python main.py

## Example

Input:
I love this product

Output:
Emotion: joy
