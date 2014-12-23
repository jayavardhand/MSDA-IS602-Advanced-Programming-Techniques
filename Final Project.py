__author__ = 'jayavardhand'

from sklearn.ensemble import RandomForestClassifier
from numpy import genfromtxt, savetxt
import numpy as np
import csv
import pandas as pd
from pandas import DataFrame as df

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt('D:/train.csv', delimiter=',', dtype=np.str_)[1:]

    print ('loaded train')

    target = [x[1] for x in dataset]
    train = [x[2:5].tolist() + x[14:].tolist() for x in dataset]

    #Clear the variable to release the memory.
    dataset = None

    dataset = genfromtxt('D:/test.csv', delimiter=',', dtype=np.str_)[1:]
    test_id = [x[0] for x in dataset]
    test = [x[1:4].tolist() + x[13:].tolist() for x in dataset]

    print ('loaded test')

    #create and train the random forest
    rf = RandomForestClassifier(n_estimators=100, n_jobs=16)
    rf.fit(train, target)

    print ("done fitting")

    columns_obj = ["id", "click"]
    list_obj = list(zip(test_id,rf.predict(test).tolist()))
    df_obj = df(list_obj, columns=columns_obj)

    savetxt('D:/submission.csv', df_obj, delimiter=',', fmt='%s')
    print ("prediction complete")

if __name__=="__main__":
    main()
