Fake News Detection using NLP & Deep Learning
Overview

This project implements an end-to-end NLP pipeline for fake news classification. It compares classical feature extraction techniques (TF-IDF, Word2Vec) with deep learning architectures (LSTM, GRU) and transformer-based approaches.

The goal was to evaluate performance trade-offs between traditional machine learning models and sequence-based neural networks for text classification.

Tech Stack

Python

Pandas, NumPy

Scikit-learn

TensorFlow / Keras

Gensim (Word2Vec)

Streamlit (for interactive demo)

Models Implemented

TF-IDF + Naive Bayes

Word2Vec + Classifier

LSTM

GRU

Transformer-based model

Project Structure

notebook/ – model training & experimentation

classifier.py – model logic

main.py – execution script

streamlit_app.py – interactive UI for predictions

utils.py – preprocessing & helper functions

Notes

Trained model weights are not included in this repository due to size limitations. Models can be trained using the provided notebooks.
