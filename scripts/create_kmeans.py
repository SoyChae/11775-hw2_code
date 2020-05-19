#!/bin/python
import numpy as np
import os
import pickle
import sys
import glob
import time
# Generate k-means features for videos; each video is represented by a single vector
# python3 create_kmeans.py $kmeans_model 'kmeans.sav' $cluster_num 20 $file_list '/home/ubuntu/11775-hws/test/train/'
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: {0} kmeans_model, cluster_num, file_list".format(sys.argv[0]))
        print("kmeans_model -- path to the kmeans model")
        print("cluster_num -- number of cluster")
        print("file_list -- the list of surf")
        exit(1)

    kmeans_model = sys.argv[1]; file_list = sys.argv[3]
    cluster_num = int(sys.argv[2])

    # load the kmeans model
    kmeans = pickle.load(open(kmeans_model,"rb"))
    #file_list = '/home/ubuntu/11775-hws/all_video.lst'
    fileName = []
    with open(file_list,'r') as f:
        for line in f.readlines():
            L = line.replace('\n', '')
            fileName.append(L[0:-4])
            
    n_file= len(fileName)
    dir_file = '/home/ubuntu/11775-hws/bow_40/'
    if ~os.path.exists(dir_file):
        os.mkdir(dir_file)
        
    Empty_Surf = []
    for i in range(0, n_file):
        st = time.time()
        x = []
        path_surf = '/home/ubuntu/surf/' + fileName[i] + '.pkl'
        path_bow = dir_file + 'bow' + str(fileName[i]) + '.pkl'
        if os.path.isfile(path_bow) == False:
            print(path_surf)
            
            try:
                with open(path_surf, 'rb') as f:
                    data = pickle.load(f)
                    n1 = len(data)
                    X = []
                    for j in range(0, n1):
                        if data[j][0] is not None:
                            X.extend(kmeans.predict(data[j][0]))
                    hist, bins = np.histogram(X, np.arange(0, cluster_num+1,1))
                    if np.sum(hist) == 0:
                        x = np.ones(cluster_num)*(1/cluster_num)
                    else:
                        x = hist/np.sum(hist)

                with open(path_bow, 'wb') as f:
                    pickle.dump(x, f, pickle.HIGHEST_PROTOCOL)

                ed = time.time()
                print(str(i+1) + '/' + str(n_file) + ', time: '+ str(ed - st))
            except:
                Empty_Surf.append(path_surf)
    
    
    print("K-means features generated successfully!")
