#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print("model_file -- path of the trained svm file")
        print("feat_dir -- dir of feature files")
        print("file_list_path -- path of list file (val.lst or test.lst)")
        print("output_file -- path to save the prediction score")
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    file_list_path = sys.argv[3]
    output_file = sys.argv[4]
    
    file_list = []
    with open(file_list_path) as f:
        for line in f.readlines():
            L = line.replace('\n', ' ').split()
            file_list.append(L[0])
            
    smodel = pickle.load(open(model_file,"rb"))
    possible_results = ['NULL', 'P001','P002','P003']        
    
    pred = []
    conf = []
    print('SVM_MODEL: {}'.format(model_file))
    for file in file_list:
        bow_file = feat_dir + 'bow' + file + '.pkl'
        if os.path.isfile(bow_file):
            with open(bow_file,'rb') as f:
                data = pickle.load(f)
                pred.extend(smodel.predict([data]))
                conf.extend(smodel.decision_function([data]))
        else:
            pred.extend(['NULL'])
            conf.extend([[1, 0, 0, 0]])
           
    print('NUM PREDICTION TO TEST: {}'.format(len(pred)))

    
    with open(output_file,'w') as f:
        for i in range(0, len(file_list)):
            video = file_list[i]
            f.write(str(video) + ' ' + pred[i] + '\n')
            
    for i in range(1,4):
#         tmp = np.asarray(pred)
#         template = np.zeros(np.size(tmp))
#         with open(possible_results[i] +'_val','w') as f:
#             ind = np.where(tmp == possible_results[i])
#             for j in range(0, len(ind)):
#                 template[ind[j]] = 1
#             for j in range(0, len(template)):
#                 f.write(str(int(template[j])) +'\n')
        
        print(output_file[0:-4]+'_'+possible_results[i] +'_val_label')
        with open(output_file[0:-4]+'_'+possible_results[i] +'_val_label','w') as f:
            for j in range(0, len(pred)):
                video = file_list[j]
                if j< len(pred)-1:
                    f.write(str(conf[j][i])+' # confidence for video ' + video + '\n')
                else:
                    f.write(str(conf[j][i])+' # confidence for video ' + video + '\n')

