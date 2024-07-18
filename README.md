# Identification of Molecular Substances from NMR Spectroscopy

## Overview
We aim to expedite chemical research by developing a model to classify chemical structures using NMR spectroscopy data. This model will enable rapid identification of compounds and comprehension of their environmental interactions, thereby streamlining drug design.

## Research Goal
In this research, we will analyze an unknown molecule's spectral data to predict the functional groups present in the molecule. Our model will take input data and process it to output binary classes for various chemical labels.

## Input Data
The input to our algorithm includes:
- **5 features** about the reaction environment
- **20 features** detailing numerical values in ppm (parts per million) about the chemical shifts

## Methodology
We employ several machine learning techniques to predict the functional groups:
- Multilabel Logistic Regression
- Fully Connected Neural Network
- Mixed Fully Connected Neural Network and RNN-LSTM
- XGBoost

## Labels
Our model predicts binary classes for each of the following 12 labels:
- Alkane
- Alkene
- Aldehyde/Ketone
- Alcohol
- Amide
- Amine
- Ether
- Carboxylic Acid
- Alkyne
- Ester
- Aromatic Ring
- Nitrile

## Conclusion
By leveraging these advanced machine learning models, we aim to significantly improve the speed and accuracy of chemical compound identification, thereby aiding in faster and more efficient drug design and environmental interaction analysis.

