# Author: Answer03  djqrszysy

import cv2 
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, Module
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from music21 import *

np.set_printoptions(threshold = np.inf)

device = torch.device('cpu')
seq_len = 32
lstm_layers = 1
bidirectional_lstm = False
D = 1 
dim = 60
BASE = 40 
hidden_size = 16
prev_h = np.load("./music-source/prev_h.npy")
prev_c = np.load("./music-source/prev_c.npy")
prev_h = torch.FloatTensor(prev_h).to(device)
prev_c = torch.FloatTensor(prev_c).to(device)
harmy = [1.0,0.0,0.25,0.5,0.5,1.0,0.0,1.0,0.5,0.5,0.25,0.0]

    
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    
    def forward(self, x):
        x = self.embedding(x)
        output = self.transformer(x, x)
        output = torch.mean(output, dim = 1)
        sorrowness = self.fc(output)
        return torch.sigmoid(sorrowness)

model = torch.load("./model/transformer.pt")
model.to(device='cpu')

def fitness2 (x) : 
    x=np.expand_dims(x,axis=0)
    x = torch.tensor(x).long()
    return model(x).item()

polularation_size=200

class musicpiece(object):
    def __init__(self):
        self.val=0


def cvt(piece1):
    ret=np.array([])
    for i in range(32):
        ret=np.append(ret,piece1.pcs[i])
        ret=np.append(ret,piece1.dely[i])
    return ret

def fitnes(piece1):
    cnt=0
    sum=0
    nonono=1
    for i in range(31):
        if (piece1.pcs[i] != piece1.pcs[i+1]) or (piece1.dely[i]!=0):
            cnt=cnt+1
            dif=(piece1.pcs[i+1]-piece1.pcs[i]+36)%12
            cjzq = abs((piece1.pcs[i+1]-piece1.pcs[i])/12)
            sum=sum+harmy[dif]*(1-0.2*cjzq)
            if cjzq>=2:
                nonono=0
    if nonono==0:
        return 0
    val1=sum/cnt
    argu1=cvt(piece1)
    val3=fitness2(argu1)
    return 0.05*val1+0.95*val3


mybigmap=np.zeros((12,100))
for i in range(12):
    mybigmap[i][60+i]=1
    mybigmap[i][62+i]=1
    mybigmap[i][64+i]=1
    mybigmap[i][65+i]=1
    mybigmap[i][67+i]=1
    mybigmap[i][69+i]=1
    mybigmap[i][71+i]=1
    for j in range(72+i,90):mybigmap[i][j]=mybigmap[i][j-12]
    for j in range(59+i,30,-1):mybigmap[i][j]=mybigmap[i][j+12]


def turndiao(piece1,whcd):
    retpc=piece1
    for i in range(32):
        if mybigmap[whcd][piece1.pcs[i]]!=1:
            if piece1.pcs[i]>53:
                retpc.pcs[i]-=1
            else:retpc.pcs[i]+=1
    return retpc

def reproduction(a,b):
    ret=musicpiece()
    ret.pcs=np.append(a.pcs[0:16],b.pcs[16:32])
    ret.dely=np.append(a.dely[0:16],b.dely[16:32])


    prob=random.random()
    if prob<=0.05:
        pos=random.randint(0,31)
        ret.pcs[pos]=random.randint(53,79)
        ret.dely[pos] = random.randint(0,1)
    prob = random.random()

    s = stream.Stream()
    for pc, dely in zip(ret.pcs, ret.dely):
        n = note.Note()
        n.pitch.midi = pc
        n.duration.quarterLength = 0.25 
        s.append(n)

    start_pos = random.randint(0, len(s.notes) - 4)
    end_pos = start_pos + 4
    melody_fragment = s.notes[start_pos:end_pos]

    transformation = random.choice(["transpose", "reflect"])

    fragment_notes = [note_obj for note_obj in melody_fragment]

    if transformation == "transpose":
        transpose_amount = random.randint(-3, 3) 
        transformed_fragment = []
        for note_obj in fragment_notes:
            transposed_note = note_obj
            transposed_note.pitch.midi += transpose_amount
            transformed_fragment.append(transposed_note)
    elif transformation == "reflect":
        transformed_fragment = fragment_notes[::-1]  

    transformed_s = s
    transformed_notes = list(transformed_s.notes.stream().elements)
    transformed_notes[start_pos:end_pos] = transformed_fragment
    transformed_s.notes.elements = tuple(transformed_notes)

    ttpcs= [n.pitch.midi for n in transformed_s.notes]


    ret.pcs = np.array(ttpcs)
    ret.pcs = (ret.pcs-53)%27+53
    
    #maxfit=0
    #maxargft=0
    #for i in range(12):
    #    newret=turndiao(ret,i)
    #    thisval=fitnes(newret)
    #    if thisval>maxfit:
    #        maxfit=thisval
    #        maxargft=i
    #ret=turndiao(ret,maxargft)
    for i in range(31):
        if ret.pcs[i]!=ret.pcs[i+1]:
            ret.dely[i]=0
    ret.dely[31]=0
    return ret







def cmp(pc1):
    return pc1.val

populrs = []
presumfit = []
sumfit=0
sumpopulrs=0

def randomselect():
    prob=random.random()*sumfit
    l=0
    r=sumpopulrs-1
    ans=0
    while l<=r:
        mid= (l+r)//2
        if prob <= presumfit[mid]:
            ans=mid
            r=mid-1
        else :
            l=mid+1
    return populrs[ans]


for i in range(polularation_size):
    prt1 = musicpiece()
    prt1.pcs=np.random.randint(53,high=80,size=32)
    prt1.dely=np.random.randint(0,high=2,size=32)
    for i in range(31):
        if prt1.pcs[i]!=prt1.pcs[i+1]:
            prt1.dely[i]=0
    prt1.dely[31]=0
    prt1.val=fitnes(prt1)
    populrs.append(prt1)
    presumfit.append(0)
    sumpopulrs=sumpopulrs+1


Generations = 500
for tms in range(Generations):
    print(tms)
    newpopulrs = []
    sumfit=0
    for i,va in enumerate(populrs):
        sumfit+= va.val
        if i==0:
            presumfit[i]=va.val
        else:
            presumfit[i]=presumfit[i-1]+va.val
    #print(presumfit)
    for i in range(polularation_size):
        father = randomselect()
        mother = randomselect()
        newanm = reproduction(father,mother)
        newanm.val=fitnes(newanm)
        newpopulrs.append(newanm)

    populrs = newpopulrs

for i,va in enumerate(populrs):
    print(va.val)
    print(va.pcs.tolist())
    print(va.dely.tolist())
