#!/bin/python 

import numpy as np
import os
import glob
#from sklearn.svm.classes import SVC
#from sklearn.svm import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
import pickle
import sys
# Performs K-means clustering and save the model to a local file


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} event_name feat_dir feat_dim output_file".format(sys.argv[0]))
        print("feat_dir -- dir of feature files")
        print("output_file -- path to save the svm model")
        print("list_file_path -- path to lst file that will be used for SVM trainings")
        exit(1)

    feat_dir = sys.argv[1]
    output_file = sys.argv[2]
    list_file_path = sys.argv[3]
    
    train_list = []
    train_label = []
    possible_results = ['NULL', 'P001','P002','P003']
    with open(list_file_path,'r') as f:
        for line in f.readlines():
            L = line.replace('\n', ' ').split()
            train_list.append(L[0])
            train_label.append(L[1])
            
    
    n_file= int(len(train_list))
    featMat = []
    for i in range(0, n_file):
        path  = feat_dir + 'bow' + train_list[i] + '.pkl'
        with open(path,'rb') as f:
            x = pickle.load(f)
        featMat.append(x)
        
    print('# VIDEOS TO TRAIN SVM : {}'.format(n_file))
    param_grid = {'C':[0.001,0.01,0.1,1,10,100],'gamma':[0.001,0.01,0.1,1,10,100]}
    grid_search = GridSearchCV(SVC(random_state=0), param_grid, cv=5, return_train_score =True)
    
    #################################%%
    index = []
    tmp_featMat = []
    tmp_label = []
    for i in range(1,4):
        index = np.where(np.asarray(train_label) == 'P00' + str(i))
        for j in range(0, len(index[0])):
            tmp_featMat.append(featMat[index[0][j]])
            tmp_label.append(train_label[index[0][j]])
            
    L = int(len(tmp_label)*3)
    index = np.where(np.asarray(train_label) == 'NULL')
    for j in range(0, len(index[0][0:L])):
    #for j in range(0, len(index[0][0:])):
        tmp_featMat.append(featMat[index[0][j]])
        tmp_label.append(train_label[index[0][j]])
    X_train, X_test, y_train, y_test = train_test_split(tmp_featMat, tmp_label,random_state=10, test_size=0.1)
    grid_search.fit(X_train, y_train)
    
    print("SVM validation test set score: {}".format(grid_search.score(X_test, y_test)))
    print("SVM best parameters: {}".format(grid_search.best_params_))
    print("SVM best score: {}".format(grid_search.best_score_))
    #################################%%
    clf = SVC(kernel='rbf',random_state=0,C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma'],probability = True)
    clf.fit(tmp_featMat, tmp_label)
    
    y_pred = clf.predict(featMat)
    print(accuracy_score(train_label, y_pred))
    #print('SVM Chance level for NULL: ' + str(len(np.where(y_pred=='NULL')[0])/len(y_pred)))
    
    pickle.dump(clf, open(output_file, 'wb'))
    print('SVM trained successfully')
