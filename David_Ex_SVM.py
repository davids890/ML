# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 11:38:37 2018

@author: david.shaer
"""
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#                               Import packages
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
import numpy as np
import matplotlib.pyplot as  plt
plt.close('all')
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Iris = datasets.load_iris()
X = Iris.data[:,0:2]
Y = Iris.target
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size = 0.2,random_state=42)

clf = SVC()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
Accuracy = accuracy_score(y_test,y_pred)*100
print('Accuracy is: {} %'.format(Accuracy))

#7 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
New_instances_X = np.array([[4.5,3],[5.5,2.5],[-5,5],[-10,11]])
New_pred = clf.predict(New_instances_X)

#8 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
plt.scatter(X_train[:,0],X_train[:,1],color='b')
#plt.hold(True)
#plt.scatter(New_instances_X[:,0],New_instances_X[:,1],color= 'red')
plt.show()

#9 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
clf.support_vectors_[:,1]
plt.hold(True)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],color = 'r')
plt.legend(('X_train', 'support_vectors'))

#10 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# replace features
X2 = Iris.data[:,2:4]
Y2 = Iris.target
X2_train,X2_test,y2_train,y2_test = train_test_split(X2,Y2,test_size = 0.2,random_state=42)

#11
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# other classifiers
Linear_clf = SVC(kernel = 'linear')
Linear_clf.fit(X2_train,y2_train)
Linear_pred = Linear_clf.predict(X2_test)
Linear_Accuracy = accuracy_score(y2_test,Linear_pred)*100
print('Linear_Accuracy is {}%'.format(Linear_Accuracy))

poly_fit_clf = SVC(kernel = 'poly', degree = 3 )
poly_fit_clf.fit(X2_train,y2_train) 
poly_pred = poly_fit_clf.predict(X2_test)
poly_Accuracy = accuracy_score(y2_test,poly_pred)*100
print('poly_Accuracy is {}%'.format(poly_Accuracy))

RBF_clf = SVC()
RBF_clf.fit(X2_train,y2_train)
RBF_pred = Linear_clf.predict(X2_test)
RBF_Accuracy = accuracy_score(y2_test,RBF_pred)*100
print('RBF_Accuracy is {}%'.format(RBF_Accuracy))


plt.figure(1)
plt.scatter(X2_train[:,0],X2_train[:,1],color='k')
plt.hold(True)
plt.scatter(Linear_clf.support_vectors_[:,0],Linear_clf.support_vectors_[:,1],color = 'r')

plt.figure(2)
plt.scatter(X2_train[:,0],X2_train[:,1],color='k')
plt.hold(True)
plt.scatter(poly_fit_clf.support_vectors_[:,0],poly_fit_clf.support_vectors_[:,1],color = 'r')

plt.figure(3)
plt.scatter(X2_train[:,0],X2_train[:,1],color='k')
plt.hold(True)
plt.scatter(RBF_clf.support_vectors_[:,0],RBF_clf.support_vectors_[:,1],color = 'r')

#12
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Digit


Digits = datasets.load_digits()
print(Digits.data.shape)
plt.gray()
for index in range(0,10):
    plt.figure(index)
    plt.matshow(Digits.images[index])
    plt.show()

# classifier
digit_X_train,digit_X_test,digit_y_train,digit_y_test = train_test_split(Digits.data[:],Digits.target[:],test_size = 0.5,random_state=42)

Penalty = 0.1
digit_clf = SVC(gamma=0.001, C=Penalty)
digit_clf.fit(digit_X_train,digit_y_train)
digit_pred = digit_clf.predict(digit_X_test)
print('Accuracy digit for Penalty ={0:6.2f}  is:{1:6.2f}%'.format(Penalty,accuracy_score(digit_y_test,digit_pred)*100))

Penalty = 1
digit_clf = SVC(gamma=0.001, C=Penalty)
digit_clf.fit(digit_X_train,digit_y_train)
digit_pred = digit_clf.predict(digit_X_test)
print('Accuracy digit for Penalty ={0:6.2f}  is:{1:6.2f}%'.format(Penalty,accuracy_score(digit_y_test,digit_pred)*100))

Penalty = 1000
digit_clf = SVC(gamma=0.001, C=Penalty)
digit_clf.fit(digit_X_train,digit_y_train)
digit_pred = digit_clf.predict(digit_X_test)
print('Accuracy digit for Penalty ={0:6.2f}  is:{1:6.2f}%'.format(Penalty,accuracy_score(digit_y_test,digit_pred)*100))

