#!/bin/python 

import numpy as np
import os
import glob
from sklearn.cluster import MiniBatchKMeans
import pickle
import sys
import os
import pickle
import time

# Performs K-means clustering and save the model to a local file
# python3 train_kmeans.py $surf_file '/surf/' $cluster_num 20 $output_file 'kmeans.sav'
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} surf_file cluster_num output_file".format(sys.argv[0]))
        print("surf_file -- path to the surf file")
        print("cluster_num -- number of cluster")
        print("output_file -- path to save the k-means model")
        exit(1)

    surf_file = sys.argv[1]; output_file = sys.argv[3]
    cluster_num = int(sys.argv[2])
    
###### CONCAT DESCRIPTORS VALUE ACROSS KEYFPOITNS AND KEYFRAMES
    # train data list
#     tr_path = '/home/ubuntu/list/train.video'
#     tr_list = []
#     with open(tr_path,'r') as f:
#         for line in f.readlines():
#             tr_list.append(line.replace('\n','') + '.pkl') 


#     def save_pickle(file_list, output_file):
#         n_key = []
#         cnt = 0
#         X = []
#         n3 = 0
#         for file in file_list:
#             path = surf_file + file
#             with open(path, 'rb') as f:
#                 data = pickle.load(f)
#             n1 = len(data)
#             n2 = 0
#             for i in range(0, n1):
#                 tmp = data[i][0]
#                 try :
#                     X.extend(tmp)
#                     n2+= len(tmp)
#                 except :
#                     n3+=1

#             n_key.append(n2)
#             print(cnt, file)

#             with open('train_surf' + output_file + '.pickle', 'wb') as f:
#                 pickle.dump(X, f, pickle.HIGHEST_PROTOCOL)

#             with open('train_count' + output_file + '.pickle', 'wb') as f:
#                 pickle.dump(n_key, f, pickle.HIGHEST_PROTOCOL)
#             cnt+=1


#     N = int(len(tr_list))
#     for i in range(0,N):
#         ind = tr_list[i*N:i*N+N]
#         save_pickle(ind, str(i))
    #########################################
    base = '/home/ubuntu/11775-hws/test/train/'
    model = MiniBatchKMeans(n_clusters=cluster_num, init='k-means++', random_state=0)
    n_file= int(len(glob.glob(base +'*.pickle'))/2)
    for i in range(0, n_file):
        X = []
        st  = time.time()
        path  = base + 'train_surf' + str(i) + '.pickle'
        with open(path, 'rb') as f:
            data = pickle.load(f)
            model = model.partial_fit(data)
        ed = time.time()
        print('kMeans Partial Learning: ' + str(i+1) + '/' + str(n_file) + ', time: ' + str(ed - st))
        pickle.dump(model, open(output_file, 'wb'))
    
    print("K-means trained successfully!")
