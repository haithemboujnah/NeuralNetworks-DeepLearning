
# Neural Networks and Deep Learning

A collection of practical implementations of neural networks using Python, Scikit-learn, and TensorFlow/Keras.

## Projects

### 1. Handwritten Digit Recognition – MNIST
A classification project using a Multi-Layer Perceptron (MLP) to recognize handwritten digits.

**Features:**
- Data preprocessing and normalization.
- MLP model implementation using Scikit-learn.
- Model training and evaluation.
- Confusion matrix and classification report.
- Cross-validation and hyperparameter optimization using GridSearchCV.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.

### 2. Bike Sharing Demand Prediction
A regression project using an Artificial Neural Network (ANN) to predict hourly bike rental demand.

**Features:**
- Data cleaning and preprocessing.
- Feature scaling using StandardScaler.
- ANN architecture implementation using TensorFlow/Keras.
- Model training with Early Stopping.
- Model evaluation using MAE, MSE, and R².
- Actual vs. predicted values visualization.

**Technologies:** Python, Pandas, NumPy, Scikit-learn, TensorFlow, Keras, Matplotlib.

## Repository Structure

```text
neural-networks-and-deep-learning/
│
├── mnist-classification/
│   ├── main.py
│   └── requirements.txt
│
├── bike-sharing-prediction/
│   ├── main.py
│   └── requirements.txt
│
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/haithemboujnah/NeuralNetworks-DeepLearning
cd NeuralNetworks-DeepLearning
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Running the Projects

**MNIST Classification:**

```bash
cd mnist-classification
python main.py
```

**Bike Sharing Prediction:**

```bash
cd bike-sharing-prediction
python main.py
```

## Learning Objectives

- Understand the fundamentals of artificial neural networks.
- Implement MLP models for classification tasks.
- Build ANN models for regression tasks.
- Apply data preprocessing and feature scaling.
- Evaluate neural network performance.
- Optimize model hyperparameters.

## Author

**Haithem Boujnah**

---

*Exploring neural networks through practical machine learning and deep learning applications.*
