# Sentiment Analysis

This project focuses on sentiment analysis of tweets related to airline experiences using machine learning and deep learning techniques. It involves preprocessing the text data, building classification models, and evaluating their performance.

## Overview

The project includes the following major components:

1. Data Collection: The dataset consists of tweets labeled as either complaints or non-complaints regarding airline experiences.

2. Data Preprocessing: Text data preprocessing involves tasks such as lowercasing, removing special characters, stopwords, and entity mentions, tokenization, and vectorization using TF-IDF.

3. Machine Learning Model: A Multinomial Naive Bayes classifier is trained on the preprocessed TF-IDF vectors to classify tweets.

4. Deep Learning Model: A fine-tuned BERT (Bidirectional Encoder Representations from Transformers) model is implemented for tweet classification.

5. Evaluation: Both models are evaluated using metrics such as AUC-ROC curve and accuracy on a validation set.

## Dependencies

- Python 3.x
- pandas
- scikit-learn
- tqdm
- numpy
- matplotlib
- torch
- transformers
- nltk

## Usage

1. Clone the repository:

3. Download the dataset from the provided link and unzip it in the `data` directory.

4. Execute the Jupyter notebook `sentiment_analysis.ipynb` to run the code cells step-by-step.

## Results

The performance of the models is evaluated based on AUC-ROC curve and accuracy:

- Multinomial Naive Bayes:
  - AUC: 0.8269
  - Accuracy: 72.65%

- BERT Classifier:
  - AUC: 0.8963
  - Accuracy: 80.59%

## Acknowledgements

- The BERT model is implemented using the Hugging Face Transformers library.

