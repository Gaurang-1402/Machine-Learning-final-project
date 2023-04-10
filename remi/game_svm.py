import numpy as np
import pickle

weights = np.load('svm_weights_poly3.npy')
with open('svm_polynomial_features_poly3.pkl', 'rb') as file:
    poly = pickle.load(file)

def EvalBoardSVM(board):
    if board.valid == False:
        return 0
    # format board into a 1D array
    x = poly.transform(board.pieces.flatten().reshape(1, -1))
    # perform poly 3 transform
    score = np.dot(weights.T, x.T)
    guess = np.argmax(score)
    if guess == 2:
        return -1
    return guess