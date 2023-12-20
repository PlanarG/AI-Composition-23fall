# author: zz
# work(dataPath, save_name):
# transform every .mid file under dataPath into a .npy file named save_name
import json
import numpy as np
import os
import converter

def mid2json(dataPath):
    fileNames = os.listdir(dataPath)
    for file in fileNames:
        # print(file[-4:])
        if(file[-4:]=='.mid'):
            prefix = file[:-4]
            suffix = '.txt'
            converter.miditodata(dataPath + prefix + '.mid', dataPath + prefix + '.json')

def deal(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    s = []
    for i in range(len(data['pitch'])):
        s.append(data['pitch'][i])
        s.append(data['rhythm'][i])
    return s

def work(dataPath, save_name):
    mid2json(dataPath)
    fileNames = os.listdir(dataPath)
    res = []
    cnt = 0
    for file in fileNames:
        if(file[-5:]=='.json'):
            t = deal(dataPath + file)
            for j in range(1, 1 + len(t)//64):
                s = np.array(t[((j-1)*64):(j*64)])
                s = np.reshape(s, (1, s.shape[0]))
                
                if cnt == 0:
                    res = s
                    cnt = 1
                else:
                    res = np.concatenate((res,s),0)
    print(res.shape)
    np.save(save_name, res)
work(dataPath = "./data/", save_name = "file.npy")