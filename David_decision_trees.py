# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 13:48:26 2018

@author: david.shaer
"""

#==============================================================================
#                           import packages
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score
#==============================================================================

# import data
url = r'http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
Data = pd.read_csv(url)
X_train, X_test, y_train, y_test = train_test_split(Data.iloc[:,2:], Data.iloc[:,1], test_size=0.7)
s = lambda x: True  if x== 'M' else False   # 1='M'  , 0 = 'B'
y_train = y_train.apply(s)*1
y_test = y_test.apply(s)*1

# Imupraty - Gini
def Find_Impuraty(Data):
#    S = sum(Data)
#    P0 = S/len(Data)
#    P1 = 1-P0
#    Gini_count = P0+P1
#    return 1-Gini_count
          
    arr,count = np.unique(Data,return_counts = True)
    Gini_count = 0
    for index in range(0,len(arr)):
        Gini_count+=(count[index]/len(Data))**2
    return 1-Gini_count


# split data
def split(X_train,y_train):
    
    feature_list = []
    feature_indedx = []
    for features in range(X_train.shape[1]):
        Weighted_average_Gini_list = []
        for inst in range(len(X_train)):
                        
            Indeces_big = X_train.iloc[:,features] >= X_train.iloc[inst,features]
            Indeces_small = X_train.iloc[:,features] < X_train.iloc[inst,features]
            if len(Indeces_big) <= 1 or len(Indeces_small) <= 1:
                continue
            new_table_big = y_train[Indeces_big]
            inpurity_big = Find_Impuraty(new_table_big)
            
            
            new_table_small = y_train[Indeces_small]
            inpurity_small = Find_Impuraty(new_table_small)
            
            Weighted_average_Gini = (sum(Indeces_big)/len(X_train))*inpurity_big + (sum(Indeces_small)/len(X_train))*inpurity_small 
            Weighted_average_Gini_list.append(Weighted_average_Gini)    

        if len(Weighted_average_Gini_list)==0:
            continue
        try:            
            feature_list.append(np.min(Weighted_average_Gini_list))
            feature_indedx.append(np.argmin(Weighted_average_Gini_list)) # the instace index in the feature that gives the minimum Gini
            
        except:
            print(len(Indeces_big))
            print(len(Indeces_small))
            print('shape, ',X_train.shape)   
            print('len, ',len(Weighted_average_Gini_list))
    Node_feature_index = np.argmin(feature_list)
    #Node_feature_instance_val = feature_indedx[Node_feature_index]
    Node_feature_instance_val = X_train.iloc[feature_indedx[Node_feature_index],Node_feature_index]
    #  feature_indedx[Node_feature_index]
    
    #Node_feature_instance_val = X_train.iloc[:,features]
    Gini_value = feature_list[Node_feature_index]
    
#    split_table_index_right =X_train.iloc[:,Node_feature_index]   >= X_train.iloc[Node_feature_instance_val,Node_feature_index] 
#    split_table_index_Left =X_train.iloc[:,Node_feature_index]   < X_train.iloc[Node_feature_instance_val,Node_feature_index] 
    
    split_table_index_right =X_train.iloc[:,Node_feature_index]   >= Node_feature_instance_val 
    split_table_index_Left =X_train.iloc[:,Node_feature_index]   < Node_feature_instance_val 

    
    Right_table = X_train[split_table_index_right]
    Right_table_lable = y_train[split_table_index_right]
    
    Left_table = X_train[split_table_index_Left]
    Left_table_lable = y_train[split_table_index_Left]
    
    return Left_table,Left_table_lable,Right_table,Right_table_lable,Gini_value,Node_feature_index,Node_feature_instance_val

def leaf_check(y_lable):
    return stats.mode(y_lable)[0]


class Node():
    def __init__(self,table,y_lable,depth):
        self.Node_table = table
        self.y_lable = y_lable
        self.Left= None
        self.Right= None 
        self.depth = depth
        self.isleaf = False
        self.class_result= None 
        self.Node_feature_index = None
        self.Node_feature_instance_val = None
        print(self.depth)
        
    def build_tree(self):
        Gini_value = Find_Impuraty(self.y_lable)
        if  self.depth == 19 or Gini_value < 0.00000005:
            self.isleaf = True
            self.class_result = leaf_check(self.y_lable)
            self.feature_value = None
            self.feature_index = None
                        
            return
        else:
            Left_table,Left_table_lable,Right_table,Right_table_lable,Gini_value,Node_feature_index,Node_feature_instance_val = split(self.Node_table,self.y_lable)
            self.feature_index = Node_feature_index
            self.feature_value = Node_feature_instance_val
            print(Gini_value)
                 
            self.Left = Node(Left_table,Left_table_lable,self.depth+1) # left 
            self.Left.build_tree()
            
            self.Right = Node(Right_table,Right_table_lable,self.depth+1) # Right
            self.Right.build_tree()
            
    def predict(self,test_inst):
        if self.isleaf == True:
            return self.class_result
        if test_inst[self.feature_index] >= self.feature_value :
            class_result = self.Right.predict(test_inst)
        elif test_inst[self.feature_index] < self.feature_value :
            class_result = self.Left.predict(test_inst)
        
        return class_result
        
                     
My_tree = Node(X_train,y_train,0)           
My_tree.build_tree()
y_pred = []
for index in range(len(X_test)):
    y_pred.append(My_tree.predict(X_test.iloc[index,:]))
print('Accuracy is:{0:8.2f}%'.format(accuracy_score(y_test,y_pred)*100))




































        