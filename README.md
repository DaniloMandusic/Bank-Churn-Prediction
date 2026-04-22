# Bank Churn Prediction with MLP Neural Network

End-to-end machine learning pipeline for predicting bank customer churn using a Multi-Layer Perceptron (MLP) neural network.

## Dataset
- Source: [Bank Churn Dataset](https://www.kaggle.com/datasets/abbas829/bank-customer-churn)
- Features: CustomerID, Surname, Credit Score, Location, Gender, Age, Tenure, Bank Account Balance, Number of products, Credit Card Possesion, Bank Account Activity, Estimated Salary
- Target: Churn (Exited / Not Exited)

## Tech Stack
- Python
- Scikit-learn
- Matplotlib / Seaborn
- Pytorch Lightning

## Model
- Multi-Layer Perceptron (MLP)
- Input layer + hidden layers + output layer
- Activation functions: ReLU, Sigmoid
- Loss: Binary Crossentropy

## Training
- 100 epochs
- Early stop on validation loss with patience 10
- Saving top 1 checkpoint

## Model Selection
- 3 iterations of cross validation

## Evaluation
- Accuracy: 0.86
- Precision: 0.85
- Recall: 0.86
- F1-score: 0.85
