from sklearn.linear_model import LogisticRegression # importing Sklearn's logistic regression's module
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing # preprossing is what we do with the data before we run the learning algorithm
from sklearn.model_selection import train_test_split 
import numpy as np
from scipy.io import arff
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.utils import shuffle


# import math

import matplotlib.pyplot as plt

# Load the .arff file
connect_4_dataset = arff.loadarff('../connect-4.arff')

# Convert to a numpy array
data = np.asarray(connect_4_dataset[0].tolist(), dtype=np.float32)

loaded = arff.loadarff('../connect-4.arff')

data = np.asarray(loaded[0].tolist(), dtype=np.float32)
X = data[:, :-1]
y = data[:, -1]

# extract the classes
X_zeros = X[y == 0]
y_zeros = y[y == 0]
X_ones = X[y == 1]
y_ones = y[y == 1]
X_twos = X[y == 2]
y_twos = y[y == 2]

max_from_each_class = min(X_zeros.shape[0], X_ones.shape[0], X_twos.shape[0])
print("max_from_each_class:", max_from_each_class)

# shuffle
np.random.seed(10)
X_zeros, y_zeros = shuffle(X_zeros, y_zeros)
X_ones, y_ones = shuffle(X_ones, y_ones)
X_twos, y_twos = shuffle(X_twos, y_twos)

# take only the first max_from_each_class elements
X_zeros = X_zeros[:max_from_each_class]
y_zeros = y_zeros[:max_from_each_class]
X_ones = X_ones[:max_from_each_class]
y_ones = y_ones[:max_from_each_class]
X_twos = X_twos[:max_from_each_class]
y_twos = y_twos[:max_from_each_class]

# split into train, test, and validation
X_zeros_train, X_zeros_testval, y_zeros_train, y_zeros_testval = train_test_split(X_zeros, y_zeros, test_size=0.2)
X_ones_train, X_ones_testval, y_ones_train, y_ones_testval = train_test_split(X_ones, y_ones, test_size=0.2)
X_twos_train, X_twos_testval, y_twos_train, y_twos_testval = train_test_split(X_twos, y_twos, test_size=0.2)
X_zeros_test, X_zeros_val, y_zeros_test, y_zeros_val = train_test_split(X_zeros_testval, y_zeros_testval, test_size=0.5)
X_ones_test, X_ones_val, y_ones_test, y_ones_val = train_test_split(X_ones_testval, y_ones_testval, test_size=0.5)
X_twos_test, X_twos_val, y_twos_test, y_twos_val = train_test_split(X_twos_testval, y_twos_testval, test_size=0.5)

# concatenate
X_train = np.concatenate((X_zeros_train[:max_from_each_class], X_ones_train[:max_from_each_class], X_twos_train[:max_from_each_class]), axis=0)
y_train = np.concatenate((y_zeros_train[:max_from_each_class], y_ones_train[:max_from_each_class], y_twos_train[:max_from_each_class]), axis=0)
X_test = np.concatenate((X_zeros_test[:max_from_each_class], X_ones_test[:max_from_each_class], X_twos_test[:max_from_each_class]), axis=0)
y_test = np.concatenate((y_zeros_test[:max_from_each_class], y_ones_test[:max_from_each_class], y_twos_test[:max_from_each_class]), axis=0)
X_val = np.concatenate((X_zeros_val[:max_from_each_class], X_ones_val[:max_from_each_class], X_twos_val[:max_from_each_class]), axis=0)
y_val = np.concatenate((y_zeros_val[:max_from_each_class], y_ones_val[:max_from_each_class], y_twos_val[:max_from_each_class]), axis=0)

# convert 2s to 0s
# X_train[X_train == 2] = 0
# X_test[X_test == 2] = 0
# X_val[X_val == 2] = 0

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)
print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

X_train = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_val = np.hstack((np.ones((X_val.shape[0], 1)), X_val))


# Check to make sure everything is as expected
print('X_train:' + str(X_train.shape))
print('y_train:' + str(y_train.shape))
print('X_val: \t'  + str(X_val.shape))
print('y_val: \t'  + str(y_val.shape))
print('X_test: '  + str(X_val.shape))
print('y_test: '  + str(X_val.shape))

# ! Class value composition in training, validation, and test set
print("y_train zeros", y_train[y_train==0].shape[0])
print("y_train ones", y_train[y_train==1].shape[0])
print("y_train twos", y_train[y_train==2].shape[0])

print("y_val zeros", y_val[y_val==0].shape[0])
print("y_val ones", y_val[y_val==1].shape[0])
print("y_val twos", y_val[y_val==2].shape[0])

print("y_test zeros", y_test[y_test==0].shape[0])
print("y_test ones", y_test[y_test==1].shape[0])
print("y_test twos", y_test[y_test==2].shape[0])

X_train_val = np.vstack((X_train, X_val))
y_train_val = np.concatenate((y_train, y_val))

# Check to make sure everything is as expected
print('X_train_val:' + str(X_train_val.shape))
print('y_train_val:' + str(y_train_val.shape))

def one_vs_rest_encoding(y, digit = '0'):
    y_encoded = np.where(y == int(digit), 1, 0)
    return  y_encoded

# ! Class value composition in training, validation, and test set
print("y_train_val zeros", y_train_val[y_train_val==0].shape[0])
print("y_train_val ones", y_train_val[y_train_val==1].shape[0])
print("y_train_val twos", y_train_val[y_train_val==2].shape[0])


def logistic_regression_train_val(num_iterations, feature_degree, lambd, regularization):
    
    
    # Create the 3 classifiers
    labels = "012"
    w_vals = {}
    w_trains = {}
    val_scores = {}
    confusion_matrices = {}
    train_scores = {}
    for i in range(len(labels)):

        # Perform one-vs-rest for labels[i]
        y_encoded = one_vs_rest_encoding(y_train, labels[i])

        poly = PolynomialFeatures(feature_degree) # * generate all types of polynomial features up to degree d
        X_tr_poly = poly.fit_transform(X_train) # * transforms the training data to have those polynomial features
        
        
        if regularization == 'l1':
            if lambd > 0:
                logreg = LogisticRegression(penalty=regularization, C=lambd, solver='liblinear', max_iter=num_iterations)
            else:
                logreg = LogisticRegression(penalty=regularization, solver='liblinear', max_iter=num_iterations)
        else:
            if lambd > 0:
                logreg = LogisticRegression(penalty=regularization, C=lambd, max_iter=num_iterations)
            else:
                logreg = LogisticRegression(penalty=regularization, max_iter=num_iterations)                
                            
        logreg.fit(X_tr_poly, y_encoded) # ! Train
            
        w_vals[i] = logreg.coef_
        
        w_trains[i] = logreg.coef_
        
        train_scores[i] = accuracy_score(y_encoded, logreg.predict(X_tr_poly))

        X_val_poly = poly.transform(X_val) # * transforms the validation data to have those polynomial features

        y_pred_val = logreg.predict(X_val_poly)

        y_encoded_val = one_vs_rest_encoding(y_val, labels[i])

        cm = confusion_matrix(y_encoded_val, y_pred_val)

        confusion_matrices[labels[i]] = cm

        # compute the accuracy of the classifier
        val_accuracy = accuracy_score(y_encoded_val, y_pred_val)

        val_scores[i] = val_accuracy

    for i in range(len(labels)):
         print("Validation set Model", i, "{:.2%}".format( val_scores[i]))
        
    # We will create a numpy array of length N, where N is the number of examples in the validation set. 
    # `combined_model_evaluation_1` will hold either a 1 or a 0, depending on whether the handwritten digit was predicted correctly or not.
    combined_model_evaluation_1 = np.zeros(len(y_train))

    y_predict_train = np.zeros(len(y_train))
    
# Loop through each sample in the validation set and assign it a label based on the highest score. 
    # Store either a 1 if the number was predicted correctly, or a 0 if the number was predicted incorrectly.
    for i in range(len(X_train)):
        
        label_scores = np.zeros(len(labels))
        
        for j in range(len(labels)):
            X_train_i_2d = X_tr_poly[i].reshape(1, -1)  # Reshape X_train to a 2D array with shape (1, 43).
            label_scores[j] = X_train_i_2d @ w_trains[j].T  # Compute the score for each label.
        
        index_label_max_score = np.argmax(label_scores) # Get the index of the label with the highest score.
        y_predict_train[i] = labels[index_label_max_score]  # Assign the predicted label to `y_predict_val`.
        if int(labels[index_label_max_score]) == int(y_train[i]):  # Check if the prediction is correct.
            combined_model_evaluation_1[i] = 1  # If the prediction is correct, assign 1 to `combined_model_evaluation_1`.
        else:
            combined_model_evaluation_1[i] = 0  # If the prediction is incorrect, assign 0 to `combined_model_evaluation_1`.

    # Print the accuracy score as a percentage
    accuracy_train = np.sum(combined_model_evaluation_1) / len(y_train)
    print("Final model accuracy score on train set: {:.2%}".format(accuracy_train))


    combined_model_evaluation_2 = np.zeros(len(y_val))
    y_predict_val = np.zeros(len(y_val))

    # Loop through each sample in the validation set and assign it a label based on the highest score. 
    # Store either a 1 if the number was predicted correctly, or a 0 if the number was predicted incorrectly.
    for i in range(len(X_val)):
        
        label_scores = np.zeros(len(labels))
        
        for j in range(len(labels)):
            X_val_i_2d = X_val_poly[i].reshape(1, -1)  # Reshape X_val to a 2D array with shape (1, 43).
            label_scores[j] = X_val_i_2d @ w_vals[j].T  # Compute the score for each label.
        
        index_label_max_score = np.argmax(label_scores) # Get the index of the label with the highest score.
        y_predict_val[i] = labels[index_label_max_score]  # Assign the predicted label to `y_predict_val`.
        if int(labels[index_label_max_score]) == int(y_val[i]):  # Check if the prediction is correct.
            combined_model_evaluation_2[i] = 1  # If the prediction is correct, assign 1 to `combined_model_evaluation_1`.
        else:
            combined_model_evaluation_2[i] = 0  # If the prediction is incorrect, assign 0 to `combined_model_evaluation_1`.

    # Print the accuracy score as a percentage
    accuracy_val = np.sum(combined_model_evaluation_2) / len(y_val)
    print("Final model accuracy score on validation set: {:.2%}".format(accuracy_val))
    return accuracy_train, accuracy_val

def logistic_regression_train_test(num_iterations, feature_degree, lambd, regularization):
    # Create the 3 classifiers
    labels = "012"
    w_tests = {}
    test_scores = {}

    confusion_matrices = {}

    for i in range(len(labels)):

        # Perform one-vs-rest for labels[i]
        y_encoded = one_vs_rest_encoding(y_train_val, labels[i])

        poly = PolynomialFeatures(feature_degree) # * generate all types of polynomial features up to degree d
        X_tr_val_poly = poly.fit_transform(X_train_val) # * transforms the training data to have those polynomial features
        
        if regularization == 'l1':
            if lambd > 0:
                logreg = LogisticRegression(penalty=regularization, C=lambd, solver='liblinear', max_iter=num_iterations)
            else:
                logreg = LogisticRegression(penalty=regularization, solver='liblinear', max_iter=num_iterations)
        else:
            if lambd > 0:
                logreg = LogisticRegression(penalty=regularization, C=lambd, max_iter=num_iterations)
            else:
                logreg = LogisticRegression(penalty=regularization, max_iter=num_iterations)                
        # logreg = LogisticRegression(penalty='l2', C=1.0)
        
        logreg.fit(X_tr_val_poly, y_encoded)
        
        w_tests[i] = logreg.coef_
                
        X_test_poly = poly.transform(X_test) # * transforms the validation data to have those polynomial features

        y_pred_test = logreg.predict(X_test_poly)

        y_encoded_test = one_vs_rest_encoding(y_test, labels[i])
        
        cm = confusion_matrix(y_encoded_test, y_pred_test)

        confusion_matrices[labels[i]] = cm

        # compute the accuracy of the classifier
        test_accuracy = accuracy_score(y_encoded_test, y_pred_test)
        test_scores[i] = test_accuracy

    for i in range(len(labels)):
        print("Model", i, "{:.2%}".format(test_scores[i]))
        

    # We will create a numpy array of length N, where N is the number of examples in the validation set. 
    # `combined_model_evaluation_1` will hold either a 1 or a 0, depending on whether the handwritten digit was predicted correctly or not.
    combined_model_evaluation_3 = np.zeros(len(y_test))

    y_predict_test = np.zeros(len(y_test))

    # Loop through each sample in the validation set and assign it a label based on the highest score. 
    # Store either a 1 if the number was predicted correctly, or a 0 if the number was predicted incorrectly.
    for i in range(len(X_test)):
        
        label_scores = np.zeros(len(labels))
        
        for j in range(len(labels)):
            X_test_i_2d = X_test_poly[i].reshape(1, -1)  # Reshape X_val to a 2D array with shape (1, 43).
            label_scores[j] = X_test_i_2d @ w_tests[j].T  # Compute the score for each label.
            
        index_label_max_score = np.argmax(label_scores) # Get the index of the label with the highest score.
        y_predict_test[i] = labels[index_label_max_score]  # Assign the predicted label to `y_predict_test`.
        if int(labels[index_label_max_score]) == int(y_test[i]):  # Check if the prediction is correct.
            combined_model_evaluation_3[i] = 1  # If the prediction is correct, assign 1 to `combined_model_evaluation_1`.
        else:
            combined_model_evaluation_3[i] = 0  # If the prediction is incorrect, assign 0 to `combined_model_evaluation_1`.

    # Print the accuracy score as a percentage
    accuracy = np.sum(combined_model_evaluation_3) / len(y_test)
    print("Final model accuracy score on test set: {:.2%}".format(accuracy))
    return accuracy



# Define the regularization parameters
# regularizations = ['l1', 'l2']
# lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
# num_iterations = 100
# feature_transformation = 1

# # Initialize the accuracy table
# accuracy_table = np.zeros((6, 9))

# # Populate the accuracy table
# for j, regularization in enumerate(regularizations):
#     for k, lambd in enumerate(lambdas):
#         accuracy_table[j*3, k], accuracy_table[j*3+1, k] = logistic_regression_train_val(num_iterations, feature_transformation, lambd, regularization)  # compute train accuracy
#         accuracy_table[j*3+2, k] = logistic_regression_train_test(num_iterations, feature_transformation, lambd, regularization)  # compute test accuracy

# print(accuracy_table)

# # Format the accuracy values as strings
# accuracy_table_str = np.array([['{:.4f}'.format(val) for val in row] for row in accuracy_table])

# print(accuracy_table_str)



regularizations = ['l1', 'l2']
lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
num_iterations = 100
feature_transformation = 3

# Initialize the accuracy table
accuracy_table = np.zeros((4, 9))

# Populate the accuracy table
for j, regularization in enumerate(regularizations):
    for k, lambd in enumerate(lambdas):
        accuracy_table[j*2, k], accuracy_table[j*2+1, k] = logistic_regression_train_val(num_iterations, feature_transformation, lambd, regularization)  # compute train accuracy
        # accuracy_table[j*2+2, k] = logistic_regression_train_test(num_iterations, feature_transformation, lambd, regularization)  # compute test accuracy

print(accuracy_table)

# Format the accuracy values as strings
accuracy_table_str = np.array([['{:.4f}'.format(val) for val in row] for row in accuracy_table])

print(accuracy_table_str)
