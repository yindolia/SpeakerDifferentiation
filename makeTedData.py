#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 17:00:15 2017

@author: yindolia
"""


from tempfile import TemporaryFile
import librosa
import os
import numpy as np


DATA_DIR = "/Users/yindolia/Desktop/data-comedy/TEDLIUM_dataset/TEDLIUM_release2/train/converted/wav"


def gen_data(one_hot):
    all_files = os.listdir(DATA_DIR)
    print(len(all_files))
    if all_files[0]==".DS*":
        all_files.pop(0)
    new_groups = []
    nested_list=[]
    j = 1
    for i in range(1,len(all_files)):
    
        if all_files[i][7:10] == all_files[i-1][7:10]:
            new_groups.append(all_files[i])
        else:
            nested_list.append(new_groups)
            #print(new_groups)
            #print(nested_list)

            new_groups=[]
            j = j+1
    nested_y = np.zeros(j-1)
    nested_y[one_hot] = 1
    nested_list = np.delete(nested_list,73, 0)
    nested_y = np.delete(nested_y, 73,0)
    return nested_list, nested_y

def gen_melspec(add):
    y, sr = librosa.load(add)
    mspect= librosa.feature.melspectrogram(y =y, sr = sr, n_mels = 40)
    return mspect

def gen_speaker(speakerList):
    mspect = np.empty(shape=[40,0])
    #np.zeros(len(speakerList), 1)
    for file in speakerList:
        pa= os.path.join(DATA_DIR, file)
        mspect1 = gen_melspec(pa)
        #print(len(mspect1[1]))
        mspect = np.hstack((mspect, mspect1))
    return mspect

def gen_batch(x,y):    
    batchx = gen_speaker(x)
    extra = len(batchx[0])%40
    print(len(batchx[0]), extra, y)
    siz= len(batchx[0]) - extra
    to_del = np.random.randint(len(batchx[0]-2), size =extra)
    batchx =np.delete(batchx, to_del, 1)
    dm= divmod(len(batchx[0])/40,1)
    print(len(to_del), batchx.shape, dm )
    if dm[1]!=0:
        batchx= np.delete(batchx[0],siz-1,0)
    batchx = np.hsplit(batchx, dm[0])
    batchy = np.repeat(y, len(batchx))
    print(batchx[0].shape, batchy.shape, batchy[0])
    part = int(len(batchx)*0.80)
    testx = batchx[part:]
    testy = batchy[part:]
    batchx = batchx[:part]
    batchy = batchy[:part]
    return batchx,batchy, testx, testy

def feed_data(hot):
    dataX, dataY= gen_data(hot)
    all_batch_train_x = [0]
    all_batch_train_y = [0]
    all_batch_test_x = [0]
    all_batch_test_y = [0]
    for dx, dy in zip(dataX, dataY):
        gbatchx, gbatchy, gtestx, gtesty = gen_batch(dx,dy)
        print(len(gbatchx), len(gbatchy))
        all_batch_train_x.append(gbatchx)
        all_batch_train_y.append(np.repeat(dy, len(gbatchx)))
        all_batch_test_x.append(gtestx)
        all_batch_test_y.append(np.repeat(dy, len(gtestx)))
    
    all_batch_train_x.pop(0)
    all_batch_train_y.pop(0)
    all_batch_test_x.pop(0)
    all_batch_test_y.pop(0)
    all_train_flat_y = [val for sublist in all_batch_train_y for val in sublist]
    all_train_flat_x = [val for sublist in all_batch_train_x for val in sublist]
    all_test_flat_x = [val for sublist in all_batch_test_x for val in sublist]
    all_test_flat_y = [val for sublist in all_batch_test_y for val in sublist]

    return np.array(all_train_flat_x), np.array(all_train_flat_y), np.array(all_test_flat_x), np.array(all_test_flat_y)

def cal_y(Y):
    
    Y = Y.reshape(len(Y),1)
    Y_ = Y-1
    Y_ = np.absolute(Y_)
    Y_ = Y_.flatten()
    Y_ = np.array([Y_])
    Y = np.concatenate((Y, Y_.T), axis = 1)
    return Y

def sav_file(X1, Y1, testX1, testY1):
    outfileX1 = TemporaryFile()
    outfileY1 = TemporaryFile()
    outfileXtest = TemporaryFile()
    outfileYtest = TemporaryFile()
    np.save(outfileX1, X1)
    np.save(outfileY1, Y1)
    np.save(outfileXtest, testX1)
    np.save(outfileYtest, testY1)
    

X1, Y1, testX1, testY1 = feed_data(1)
Y1 = cal_y(Y1)
testY1 = cal_y(testY1)

sav_file(X1, Y1, testX1, testY1)

print(X1.shape, Y1.shape, testX1.shape, testY1.shape)
