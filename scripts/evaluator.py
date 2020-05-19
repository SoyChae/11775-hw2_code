import sys
import os
from sklearn.metrics import average_precision_score
import numpy as np
if __name__=="__main__":
    #   load the ground-truth file list
    gt_fn=open(sys.argv[1]).readlines()
    pred_fn=open(sys.argv[2]).readlines()

    print("Evaluating the average precision (AP)")

    y_gt=[]
    y_score=[]
    assert(len(y_gt)==len(y_score))

    tr = []
    possible_result= ['NULL','P001','P002','P003']
        
    for lines in gt_fn:
        L = lines.replace('\n','')
        tmp = possible_result.index(L)
        y_gt.append(float(tmp))
    
    for lines in pred_fn:
        L = lines.replace('\n','').split()
        tmp = possible_result.index(L[1])
        y_score.append(float(tmp))

    assert(len(y_gt) == len(y_score))

    for i in range(0, 4):
        tru = np.zeros(len(y_gt))
        pred = np.zeros(len(y_gt))
        if float(i) in y_gt:
            ind1 = y_gt.index(float(i))
            tru[ind1] = float(1)
        if float(i) in y_score:
            ind2 = y_score.index(float(i))
            pred[ind2]= float(1)
        print(possible_result[i] + " Average precision: ",average_precision_score(tru, pred))
        
