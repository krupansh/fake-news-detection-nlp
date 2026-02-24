# Fake News Detection using NLP & Deep Learning

## Overview

This project implements an **end-to-end NLP pipeline** for fake news classification.  
It compares **classical feature extraction techniques** (TF-IDF, Word2Vec) with **deep learning architectures** (LSTM, GRU) and transformer-based models.

The objective was to evaluate **performance trade-offs** between traditional machine learning models and sequence-based neural networks for text classification.


## Tech Stack

- **Python**
- **Pandas, NumPy**
- **Scikit-learn**
- **TensorFlow / Keras**
- **Gensim (Word2Vec)**
- **Streamlit** (interactive demo application)


## Models Implemented

- **TF-IDF + Naive Bayes**
- **Word2Vec + Classifier**
- **LSTM**
- **GRU**
- **Transformer-based model**


## Pipeline Workflow

1. Text preprocessing (cleaning, tokenization, normalization)
2. Feature extraction (TF-IDF / Word2Vec embeddings)
3. Model training & validation
4. Performance evaluation using:
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
5. Deployment via Streamlit interface


## Project Structure
##notebook/ → Model experimentation & training##
##classifier.py → Model logic##
##main.py → Execution script##
##streamlit_app.py → Interactive UI##
utils.py → Preprocessing helpers##



## Notes

- Trained model weights are excluded due to size limitations.
- Models can be retrained using the provided notebooks.
- Focus is on comparative evaluation of classical vs deep learning approaches.
