import pandas as pd
import numpy as np
import csv
import glob


path = "WalkingPose Data/*.csv"

final_inp = []

for fname in glob.glob(path):
    print(fname)

    data = pd.read_csv(fname,header=None)
    llegs = data.iloc[:,3:9].values
    rlegs = data.iloc[:,12:18].values
    nrows = llegs.shape[0]
    for i in range(nrows-4):
        print(type(llegs[i]))
        print(llegs[i][3:6])

        x = []
        x = np.concatenate((llegs[i],rlegs[i],llegs[i+1],rlegs[i+1],llegs[i+2],rlegs[i+2],llegs[i+3],rlegs[i+3],llegs[i+4][3:6],rlegs[i+4][3:6]),axis=0)
        print(type(x))
        final_inp.append(x)
    #print(final_inp)

myFields = ['lkx','lky','lkz','lx','ly','lz','rkx','rky','rkz','rx','ry','rz','o_lkx','o_lky','o_lkz','o_lx','o_ly','o_lz','o_rkx','o_rky','o_rkz','o_rx','o_ry','o_rz']
myFile = open('train_knee_input_v1.csv', 'w',newline='')
with myFile:
     writer = csv.writer(myFile)
     writer.writerows(final_inp)

"""
    x.append(legs[i])
    x.append(legs[i+1])
    print(x)
    x = np.array(x)
    x.flatten()
    print(x)
    final_inp.append(x)
    break
"""
