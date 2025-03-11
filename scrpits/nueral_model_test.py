import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, GRU, Dropout
import mlflow
import mlflow.tensorflow

# Load preprocessed data
pre_credit_data = pd.read_csv('../data/preprocessed_creditcard_data.csv')
pre_fraud_data_df = pd.read_csv('../data/preprocessed_fraud_data.csv')

# Define functions for model building and evaluation

def build_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_rnn_model(input_shape):
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=input_shape))
    model.add(GRU(units=32))
    model.add(Dense(1, activation='sigmoid'))
    return model

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(units=32))
    model.add(Dense(1, activation='sigmoid'))
    return model

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name, experiment_name):
    # MLflow tracking
    mlflow.set_tracking_uri('http://localhost:5000')  # Set your MLflow tracking server
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param('model', model_name)

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

        # Evaluate the model
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)

        accuracy = accuracy_score(y_test, y_pred_binary)
        precision = precision_score(y_test, y_pred_binary)
        recall = recall_score(y_test, y_pred_binary)
        f1 = f1_score(y_test, y_pred_binary)
        roc_auc = roc_auc_score(y_test, y_pred)

        # Log metrics to MLflow
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('precision', precision)
        mlflow.log_metric('recall', recall)
        mlflow.log_metric('f1_score', f1)
        mlflow.log_metric('roc_auc', roc_auc)

        # Log model to MLflow
        mlflow.tensorflow.log_model(model, artifact_path='model')

        print(f'Model: {model_name}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'ROC AUC: {roc_auc:.4f}')

# Data Preparation

# Credit Card Data
X_credit = pre_credit_data.drop('Class', axis=1)
y_credit = pre_credit_data['Class']
X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)

# Fraud Data
X_fraud = pre_fraud_data_df.drop('class', axis=1)
y_fraud = pre_fraud_data_df['class']
X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)

# Reshape data for CNN, RNN, and LSTM
X_credit_train = X_credit_train.values.reshape(X_credit_train.shape[0], X_credit_train.shape[1], 1)
X_credit_test = X_credit_test.values.reshape(X_credit_test.shape[0], X_credit_test.shape[1], 1)
X_fraud_train = X_fraud_train.values.reshape(X_fraud_train.shape[0], X_fraud_train.shape[1], 1)
X_fraud_test = X_fraud_test.values.reshape(X_fraud_test.shape[0], X_fraud_test.shape[1], 1)

# Model Training and Evaluation

# Credit Card Data
print("Credit Card Data - CNN")
cnn_model_credit = build_cnn_model(X_credit_train.shape[1:])
train_and_evaluate_model(cnn_model_credit, X_credit_train, y_credit_train, X_credit_test, y_credit_test, 'CNN', 'Credit Card Fraud Detection')

print("Credit Card Data - RNN")
rnn_model_credit = build_rnn_model(X_credit_train.shape[1:])
train_and_evaluate_model(rnn_model_credit, X_credit_train, y_credit_train, X_credit_test, y_credit_test, 'RNN', 'Credit Card Fraud Detection')

print("Credit Card Data - LSTM")
lstm_model_credit = build_lstm_model(X_credit_train.shape[1:])
train_and_evaluate_model(lstm_model_credit, X_credit_train, y_credit_train, X_credit_test, y_credit_test, 'LSTM', 'Credit Card Fraud Detection')

# Fraud Data
print("Fraud Data - CNN")
cnn_model_fraud = build_cnn_model(X_fraud_train.shape[1:])
train_and_evaluate_model(cnn_model_fraud, X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test, 'CNN', 'Fraud Detection')

print("Fraud Data - RNN")
rnn_model_fraud = build_rnn_model(X_fraud_train.shape[1:])
train_and_evaluate_model(rnn_model_fraud, X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test, 'RNN', 'Fraud Detection')

print("Fraud Data - LSTM")
lstm_model_fraud = build_lstm_model(X_fraud_train.shape[1:])
train_and_evaluate_model(lstm_model_fraud, X_fraud_train, y_fraud_train, X_fraud_test, y_fraud_test, 'LSTM', 'Fraud Detection')