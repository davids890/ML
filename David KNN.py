# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 14:33:25 2018

@author: david.shaer xxxxxxxxxxx
"""
'david is here'
import numpy as np
import pandas as pnd
from sklearn import datasets
from scipy import stats
'harry is also here'
#1. Handle the data: write a function that will open the dataset and split it to training
#and testing. You can either:
#A. use the function sklearn.datasets.import_iris() which provides an object where
#.data and .target are the data and the labels
#B*. Download the data from the following link:
#https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
#store it in a text file and use numpy’s genfromtext function to read it and then sort
#the data and the labels.

def Import_data(test_size):
    iris = datasets.load_iris()
    Data = iris.data
    Lable = iris.target
    feature_names = iris.feature_names
    P_data = pnd.DataFrame(Data,columns = feature_names)
    P_data = P_data.assign(Lable = Lable)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(P_data.iloc[:,0:4].values, P_data.iloc[:,4].values, test_size=0.33, random_state=42)
    
    return X_train, X_test, y_train, y_test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# split the data
#def split():
#    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2. Distance function: write a function that can calculate the distance between two
#datasets.

def Oclide_distance(row_1,row_2):
    Distance = np.sqrt(np.sum(np.power((row_1 - row_2),2)))
    # other option:
    # from scipy.spatial import distance
    # dst = distance.euclidean(row_1, row_2)
    return Distance

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3. Nearest neighbours: write a function that searches the whole dataset for the k
#nearest neighbours.

def KNN(k,P_data,Instance_to_check):
    Distance_list = []
    for row in range(0,len(P_data)):
        compare_row = P_data[row,0:4]
        Distance_list.append(Oclide_distance(Instance_to_check,compare_row))
    Sorted_index_list = np.argsort(Distance_list)
    Sorted_data = P_data[Sorted_index_list]
    return Sorted_data[0:k,:]
 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Predict from k nearst neighbours: now that we have the k nearest neighbours
#we can calculate an average of them to predict the result, or for categorical data
#we can do voting i.e. finding the mode ( השכיח ) between these k nearest points.
   
def KNN_prediction(k,data,Instance_to_check):
    Sorted_data = KNN(k,data,Instance_to_check)
    mode = stats.mode(Sorted_data[:,4],axis = 0)[0]
    return mode  

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#5. Calculate the accuracy on the test data: calcualte the prediction on every
#element of the test data and compare to the expected values. Calculate the
#percentage of the data sets that we calculated accurately.
def main(k):
    test_size = 0.33
    X_train, X_test, y_train, y_test = Import_data(test_size)
    X_train = pnd.DataFrame(X_train) ; X_train = X_train.assign(Lable = y_train) ; X_train = X_train.values
    Fit_and_evaluate(X_test,X_train,y_test,k)

def Fit_and_evaluate(X_test,X_train,y_test,k):
    test_result = []
    test_ref = []
    
    for test_index in range(0,len(X_test)):
        Instance_to_check = X_test[test_index,:]
        test_result.append(KNN_prediction(k,X_train,Instance_to_check))
        test_ref.append(y_test[test_index])
    # calculate the accuracy
    bool_arr = [test_result[ind] == test_ref[ind] for ind in range(0,len(test_result))]
    Correct_counter = np.sum(bool_arr)
    Accuracy = (Correct_counter/len(test_result))*100
    print('Accuracy: ',Accuracy,' %')
    
    return  Accuracy

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#6. Main function: write a main function that contains everything and calls all the
#functions that we have written.
k=20
main(k)




#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                           Part 2: Classes 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#Rewrite your code to use class such that it contains:
#1. A DataSet class
#   a. It should be instantiated (__init___ function) with a dataset (with labels)
#   b. It should contain a function which gets a percentage and returns the data, after
#randomly permuting it, split to train and test according to that percentage.  

# import data without split:
iris = datasets.load_iris()
Data = iris.data
Lable = iris.target
feature_names = iris.feature_names
P_data = pnd.DataFrame(Data,columns = feature_names)
P_data = P_data.assign(Lable = Lable)

class DataSet:
    def __init__(self,Data):
        self.Data = Data
        self.X_train
    def Split_data(self,Percentage):
        print('dssa')
        (self.X_train, self.X_test, self.y_train, self.y_test) = Import_data(Percentage)
        return  self.X_train, self.X_test, self.y_train, self.y_test
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2. A KNN class
#a. Contains function which classifies test data
#b. the class should also contain all the relevant functions for the classification
#calculation.
        
#  wait for solution
#
#class Test_set(DataSet):
#    def __init__(self):
#        DataSet.__init__(self)
#    def Classify(self):
#        return Fit_and_evaluate(self.X_test,self.X_train,self.y_test,self.k)
#
#
#Percentage = 0.33 
#C_data = DataSet(P_data)
#t=Test_set(C_data)
#t.X_train
#C_accuracy = Test_set.Classify()
#
















