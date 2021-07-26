#University of St Andrews
#Machine Learning Module CS5014
#Assignment 1
#Student Ashraf Sinan
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, 
                             balanced_accuracy_score, 
                             confusion_matrix, 
                             precision_recall_curve, 
                             plot_confusion_matrix, 
                             average_precision_score)

from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler


def loadData(dataPath = 'crx.data.txt'):
    
    """ Load the data
    Parameters
    ----------
    dataPath: str
        The path of the CSV file to load
    Returns
    -------
    data
        The loaded dataset as a Pandas Data Frame

    """
    
    data = pd.read_csv(dataPath,header=None, names=headers, na_values = '?')
    return data

def removeRecordsWithNA(data):
    #drop rows that contains null values
    totalRecords = len(data)
    fullData = data.dropna(axis = 0)
    fullRecords = len(fullData)
    print("We deleted " + str(totalRecords - fullRecords) +
          " out of " + str(totalRecords) + ", The remnant records are:" +  str(fullRecords))
    
    return fullData

def removeDuplication(data):
    #Remove any duplicated values to avoid data leakage
    noDuplicateData = data.drop_duplicates()
    print(str(len(data) - len(noDuplicateData)) + " Duplicates were found and removed!")
    return noDuplicateData

def encodeBinaryCategorical(data, binaryCategoricalFeatures):
    #Encodying bynary categorical variables
    mappedData = data
    for col in binaryCategoricalFeatures:
        mappedData[col] = data[col].map({'a':0, 'b':1, 'f':0, 't':1, '-':0, '+':1})
        
    return mappedData

def encodeOneHotCategorical(data, dummyFeatures):
    #Create dummy variables
    dataDummy = pd.get_dummies(data, columns=dummyFeatures)
    return dataDummy

def scaleContinuousFeatures(data):
    #Scale the data
    minmax =MinMaxScaler()  
    scaledData = minmax.fit_transform(data)
    return scaledData 

def addMultiNonminal(data, contineousFeatures):
    # Add polymonial expansion
    polyNomial = PolynomialFeatures(2, include_bias=False, interaction_only=True)
    polyData = polyNomial.fit_transform(data)
    
    return polyData
    
def preProcessData(data,  contineousFeatures, binaryCategoricalFeatures, dummyFeatures):
    #Remove missing data, remove duplication (if any), and encode categorical variables 
    fullRecordsData = removeRecordsWithNA(data)
    notDuplicatedData = removeDuplication(fullRecordsData)
    binaryEncodedData = encodeBinaryCategorical(notDuplicatedData, binaryCategoricalFeatures)    
    oneHoteEncodedData = encodeOneHotCategorical(binaryEncodedData, dummyFeatures)
   
    return oneHoteEncodedData
   
def showConfusingMatrix(y_test, y_hat):
    print("Confusing Matrix:")
    print(confusion_matrix(y_test, y_hat))
    print("------------------------")
    
def showModelAccuracy(message, model, x_train, x_test, y_train, y_test, y_hat):
    ##Pring model accuracy
    print(message)
    print("Test size: " + str(len(y_test)))
    print("Correctly predicted: " + str(accuracy_score(y_hat, y_test, normalize=False)))
    print("Training Accuracy: "+ str(model.score(x_train, y_train)))
    print("Testing Accuracy: "+ str(accuracy_score(y_hat, y_test)))
    print("Balanced Accuracey: "+ str(balanced_accuracy_score(y_hat, y_test)))
    print("---------------------------------------------------")
    return

def showPrecisionRecall(y_test, prediction):
    #Plot precision and recall and show AP
    precision, recall, thresholds = precision_recall_curve(y_test, prediction[:,1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Precision / Recall plot")
    ax.set_xlabel("Precision")
    ax.set_ylabel("Recall")
    ax.plot(precision, recall)
    
    AP = average_precision_score(y_test, prediction[:,1])
    print("AP = " + str(AP))
    
def trainModelNoPenaltyNoWeight(x_train, x_test, y_train, y_test):
    #Train the model with no hyper-parameters
    scoreCreditModel = LogisticRegression(penalty='none',
                                          class_weight=None,
                                          tol = 1e-5,
                                           max_iter=1000,
                                          solver = 'newton-cg').fit(x_train, y_train)
    ## get the prediction, show the accurcay, confusing matrix, precision and recall
    y_hat = scoreCreditModel.predict(x_test)
    prediction = scoreCreditModel.predict_proba(x_test)
    showModelAccuracy("Training the model with penalty = none and class weight = none: ",
                      scoreCreditModel, x_train, x_test, y_train, y_test, y_hat )
    print("Confusing matrix for training and testing:")
    plot_confusion_matrix(scoreCreditModel, x_train, y_train)
    plot_confusion_matrix(scoreCreditModel, x_test, y_test)
    showPrecisionRecall(y_test, prediction)
    
    return 


def trainModelWithBalancedWeight(x_train, x_test, y_train, y_test):
    #Train model with balanced class weights
    scoreCreditModel = LogisticRegression(penalty='none', 
                                                class_weight='balanced', 
                                                max_iter=1000,
                                                solver = 'newton-cg',).fit(x_train, y_train)
    ## get the prediction, show the accurcay, confusing matrix, precision and recall
    y_hat = scoreCreditModel.predict(x_test)
    prediction = scoreCreditModel.predict_proba(x_test)
    showModelAccuracy("Training the model with penalty = none and class weight = balanced: ",
                      scoreCreditModel, x_train, x_test, y_train, y_test, y_hat )
    print("Confusing matrix for training and testing:")
    plot_confusion_matrix(scoreCreditModel, x_train, y_train)
    plot_confusion_matrix(scoreCreditModel, x_test, y_test)
    showPrecisionRecall(y_test, prediction)
    return 

def trainModelWithPenaltyAndSolver(x_train, x_test, y_train, y_test):
    ## Train model with penalty and solver
    scoreCreditModel = LogisticRegression(penalty='l2', 
                                      random_state=1, 
                                      class_weight='balanced', 
                                      solver= 'newton-cg', 
                                      max_iter=1000,tol=1e-5
                                     ).fit(x_train, y_train)
    ## get the prediction, show the accurcay, confusing matrix, precision and recall
    y_hat = scoreCreditModel.predict(x_test)
    prediction = scoreCreditModel.predict_proba(x_test)
    showModelAccuracy("Training the model with penalty = l2 and, solver = newton-cg and class weight = balanced: ",
                      scoreCreditModel, x_train, x_test, y_train, y_test, y_hat )
    print("Confusing matrix for training and testing:")
    plot_confusion_matrix(scoreCreditModel, x_train, y_train)
    plot_confusion_matrix(scoreCreditModel, x_test, y_test)
    showPrecisionRecall(y_test, prediction)
    return 

## Build the columns' headeres

headers = []
for i in range(1,17):
    headers.append("A"+str(i))
    

#Load the data

data = loadData()

continousFeatures = ["A2", "A3", "A8", "A11", "A14", "A15"]
binaryFeatures = ["A1", "A9", "A10", "A12","A16"]
dummyFeatures = ["A4","A5","A6","A7","A13"]

## Pre-process data by calling the pre-processing function 

data1  = preProcessData(data, continousFeatures, binaryFeatures, dummyFeatures)
y = data1["A16"].values
##get continous features 
cont_features = data1[["A2", "A3", "A8", "A11", "A14", "A15"]]
##polynomial expansion
polyData = addMultiNonminal(cont_features, "")

## get the x data
x = data1.drop("A16", axis = 1).values


## split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

print("Before Scaling, without polynomial expansian:")

## Train three models with different settings


trainModelNoPenaltyNoWeight(x_train, x_test, y_train, y_test)

trainModelWithBalancedWeight(x_train, x_test, y_train, y_test)

trainModelWithPenaltyAndSolver(x_train, x_test, y_train, y_test)


print("After Scaling, with polynomial expansian:")

## Add the polynomial Data

x= np.c_[x,polyData]
## Scale the continous features
x = scaleContinuousFeatures(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=5)

## Train three models with different settings

trainModelNoPenaltyNoWeight(x_train, x_test, y_train, y_test)
trainModelWithBalancedWeight(x_train, x_test, y_train, y_test)
trainModelWithPenaltyAndSolver(x_train, x_test, y_train, y_test)

print("Thanks!")


