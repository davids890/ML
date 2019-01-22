# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 11:11:29 2018

@author: david.shaer
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# import packages
import pandas as pnd
import numpy as np
from sklearn.model_selection import train_test_split
from tabulate import tabulate
from sklearn.model_selection import KFold
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 1. Handle Data: Load the data from CSV file and split it into training and test datasets
def Import_data():
    Title = ['# times pregnant',
             'Plasma glucose',
             'Diastolic BP','skin thickness',
             'insulin',
             'Body mass',
             'Diabetes pedigree function','Age','Class(0 or 1)']
    Data = pnd.read_csv('pima-indians-diabetes.csv',names=Title)
    X_train, X_test, y_train, y_test = train_test_split(Data.iloc[:,0:(Data.shape[1]-1)], Data.iloc[:,8:9], test_size=0.22, random_state=42)
  
    return X_train, X_test, y_train, y_test,Title

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def Validation_split(X_train,y_train,train_index,test_index):
    X_V_train = X_train.iloc[train_index,:]
    y_V_train = y_train.iloc[train_index,:]
    
    X_V_test = X_train.iloc[test_index,:]
    y_V_test = y_train.iloc[test_index,:]
    
    return X_V_train, X_V_test, y_V_train, y_V_test    
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#2. Summarize Data (train): summarize the properties in the training dataset by
#calculate for every feature and class (prediction value) the mean and the std.
def mean_and_std(X_tr_column):
    mean = np.mean(X_tr_column)
    std = np.std(X_tr_column)
    return mean,std

def Summarize_data(X_train,Title):
    Mean_arr = []
    std_arr = []
    for col in range(0,X_train.shape[1]):
        mean,std = mean_and_std(X_train.iloc[:,col])
        Mean_arr.append(mean)
        std_arr.append(std)
    print(tabulate([['Mean',Mean_arr],['std',std_arr]],[Title]))
    # other option:
    #X_train.describe()
    #X_train.describe().mean()
    #X_train.describe().std()
    #X_train.hist()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#3. Write a function which make a prediction: Use the summaries of the dataset to
#generate a single prediction, which based on the gaussian distribution with the
#corresponding mean and std of each of the features.

def Gaussian_P(v,E,Std):
    Std_pow = np.power(Std,2)
    Probability = (1/np.sqrt((2*np.pi*Std_pow))) * (np.exp(-((np.power((v-E),2))/(2*Std_pow))))
    return Probability

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# 4. Make Predictions: Generate predictions on the whole test dataset.
# 1. create groups based on features
def Make_predeiction(X_train,y_train,X_test):
    Train = X_train.assign(Class = y_train)
    E_arr_0_class = []
    Std_arr_0_class = []
    predict_0_inst = []
    predict_0_test = []
    
    E_arr_1_class = []
    Std_arr_1_class = []
    predict_1_inst = []
    predict_1_test = []
    
    Zero_class = Train.loc[Train.loc[:,'Class']==0]
    One_class = Train.loc[Train.loc[:,'Class']==1]
    # probability of class:
    P_class_0 = len(Zero_class) / len(X_train)
    P_class_1 = len(One_class) / len(X_train)
    # calculate the E and the Std of each feature for class 0 and class 1
    for feature in range(0,X_train.shape[1]):
        mean,std = mean_and_std(Zero_class.iloc[:,[feature]])
        E_arr_0_class.append(mean)
        Std_arr_0_class.append(std)
            
        mean,std = mean_and_std(One_class.iloc[:,[feature]])
        E_arr_1_class.append(mean)
        Std_arr_1_class.append(std)
     
    for row_inst in range(0,len(X_test)):
        predict_1_inst = []
        predict_0_inst = []
        # for one feature
        for f in range(0,X_test.shape[1]):
            P0 = Gaussian_P(X_test.iloc[row_inst,f],E_arr_0_class[f],Std_arr_0_class[f])
            predict_0_inst.append(P0)
            
            P1 = Gaussian_P(X_test.iloc[row_inst,f],E_arr_1_class[f],Std_arr_1_class[f])
            predict_1_inst.append(P1)
            
        predict_0_inst_val = P_class_0 * np.prod(predict_0_inst)
        predict_1_inst_val = P_class_1 * np.prod(predict_1_inst)
            
        predict_0_test.append(predict_0_inst_val)
        predict_1_test.append(predict_1_inst_val)
    return predict_0_test,predict_1_test
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#5. Evaluate Accuracy: Evaluate the accuracy of predictions made for a test dataset as
#the percentage correct out of all predictions made.
def Evaluation(X_test,predict_0_test,predict_1_test,y_test):
    # Compare between y_test and prediction
    y_pred = []
    for y_ind in range(0,len(predict_0_test)):
        if predict_0_test[y_ind] > predict_1_test[y_ind]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    
    # Calculate the accuracy
    Accuracy = (np.sum((y_pred == y_test)*1)/len(y_test))*100
    print("Naive base accuracy is:{0:7.2f} %".format(Accuracy[0]))
    
    return Accuracy

###############################################################################
#6. Tie it Together: Use all of the code elements to present a complete and standalone
#implementation of the Naive Bayes algorithm.
#* (Optional) Try building it into a class with fit(train) method which calculates the
#mean and std, and predict(test) method which makes a Naive Bayes prediction for
#the test data.


class Naive_Base():
    
    def __init__(self):
        self.Alg_name = 'NaiveBayes'
        self.predict_0_test = ''
        self.predict_1_test = ''
        self.X_train = ''
        self.X_test = ''
        self.y_train = ''
        self.y_test = ''
        self.Title = ''
        
    def Import_data(self):
        X_train, X_test, y_train, y_test,Title = Import_data()
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.Title = Title
        return X_train, X_test, y_train, y_test,Title
        
    def Predict(self,X_train,y_train,X_test):
        predict_0_test,predict_1_test = Make_predeiction(X_train,y_train,X_test)
        self.predict_0_test = predict_0_test
        self.predict_1_test = predict_1_test
        return predict_0_test,predict_1_test
        
    def Evaluation(self,X_test,predict_0_test,predict_1_test,y_test):
        Evaluation(X_test,predict_0_test,predict_1_test,y_test)
    

#def main():
## option 1 - regular option    
#    # Import data
#    X_train, X_test, y_train, y_test,Title = Import_data()
#    # Summarize data
#    Summarize_data(X_train,Title)
#    # Make predeiction - Naive_Base
#    predict_0_test,predict_1_test = Make_predeiction(X_train,y_train,X_test)
#    # Evaluation
#    Evaluation(X_test,predict_0_test,predict_1_test,y_test)
#    
## option 1 - Class option    
##    C_Naive_base = Naive_Base()
##    X_train, X_test, y_train, y_test,Title = C_Naive_base.Import_data()
##    predict_0_test,predict_1_test = C_Naive_base.Predict(X_train,y_train,X_test)
##    C_Naive_base.Evaluation(X_test,predict_0_test,predict_1_test,y_test)=
#    
#    
#    
#    
#
#main()
#

###############################################################################

# option 1 - regular option - using K fold cross validation 
# Import data 
X_train, X_test, y_train, y_test,Title = Import_data()

K_fold_num = 10   
kf = KFold(n_splits=K_fold_num)
K_fold_predict_list = []  
Test_prediction_flage = 1 
Itration_num=0    
for train_index, test_index in kf.split(X_train):
    Itration_num+=1
    if Test_prediction_flage == 1  :
        X_V_train, X_V_test, y_V_train, y_V_test = X_train, X_test, y_train, y_test
        Test_prediction_flage = 0
    else:
        X_V_train, X_V_test, y_V_train, y_V_test = Validation_split(X_train,y_train,train_index,test_index)
        # Summarize data
        Summarize_data(X_V_train,Title)
        # Make predeiction - Naive_Base
        predict_0_test,predict_1_test = Make_predeiction(X_V_train,y_V_train,X_V_test)
        # Evaluation
        Accuracy = Evaluation(X_V_test,predict_0_test,predict_1_test,y_V_test)
        # Add the K accuracy result to a list
        K_fold_predict_list.append(Accuracy)
Validation_avg_accuracy = np.average(K_fold_predict_list)
# Now the traioning and test (without the validation):
# Summarize data
Summarize_data(X_train,Title)
# Make predeiction - Naive_Base
predict_0_test,predict_1_test = Make_predeiction(X_train,y_train,X_test)
# Evaluation
Accuracy = Evaluation(X_test,predict_0_test,predict_1_test,y_test)
print('Validation avg accuracy is: {0:5.2f} %   and the test accuracy is: {1:5.2f} %'.format(Validation_avg_accuracy,Accuracy[0]))



# option 2 - Class option    
#    C_Naive_base = Naive_Base()
#    X_train, X_test, y_train, y_test,Title = C_Naive_base.Import_data()
#    predict_0_test,predict_1_test = C_Naive_base.Predict(X_train,y_train,X_test)
#    C_Naive_base.Evaluation(X_test,predict_0_test,predict_1_test,y_test)=
    

#    X_train, X_test = X_train[train_index], X_test[test_index]
#    y_train, y_test = y_train[train_index], y_test[test_index]

    










 