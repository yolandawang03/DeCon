# -*- coding: utf-8 -*-
import numpy as np
# from pygraph.classes.digraph import digraph
import os
from search import ReadAllTriples
import pickle
import json
import re
import math
import random

def get_index(file):

    source_vob = {}
    sourc_idex_word = {}

    f = open(file,'r')
    fr = f.readlines()
    for line in fr:
        sourc = line.strip('\r\n').rstrip('\n').split('\t')
        if not source_vob.__contains__(sourc[0]):
            source_vob[sourc[0]] = int(sourc[1])
            sourc_idex_word[int(sourc[1])] = sourc[0]
    f.close()

    return source_vob, sourc_idex_word

def load_vec_txt(fname, vocab, k=300):
    f = open(fname)
    w2v={}
    W = np.zeros(shape=(vocab.__len__() + 2, k))
    unknowtoken = 0
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        w2v[word] = coefs
    f.close()
    w2v["**UNK**"] = np.random.uniform(-0.25, 0.25, k)
    for word in vocab:
        # print(word)

        if not w2v.__contains__(word):
            w2v[word] = w2v["**UNK**"]
            unknowtoken +=1
            W[vocab[word]] = w2v[word]
        else:
            W[vocab[word]] = w2v[word]

    print('!!!!!! UnKnown tokens in w2v', unknowtoken)
    # for sss in W:
    #     print(sss)
    return k, W

def get_data_txt(trainfile):
    train_triple = []
    train_confidence = []

    f = open(trainfile, "r")
    lines = f.readlines()
    for line in lines:
        tri = line.rstrip('\r\n').rstrip('\n').rstrip('\r').split('\t')
        new_tri = [int(x) for x in tri]
        train_triple.append(tuple(new_tri))
        if tri[len(tri)-1] == '1':
            train_confidence.append([0, 1])
        else:
            train_confidence.append([1, 0])
    f.close()
    
    return train_triple, train_confidence

def get_data(entity2idfile, relation2idfile,
             entity2vecfile, relation2vecfile, w2v_k,
             trainfile, testfile,
             max_p,
             datafile):

    ent_vocab, ent_idex_word = get_index(entity2idfile)
    rel_vocab, rel_idex_word = get_index(relation2idfile)
    print("entity vocab size: ", str(len(ent_vocab)), str(len(ent_idex_word)))
    print("relation vocab size: ", str(len(rel_vocab)), str(len(rel_idex_word)))


    entvec_k, entity2vec = load_vec_txt(entity2vecfile, ent_vocab, k=w2v_k)
    print("word2vec loaded!")
    print("entity2vec  size: " + str(len(entity2vec)))

    relvec_k, relation2vec = load_vec_txt(relation2vecfile, rel_vocab, k=w2v_k)
    print("word2vec loaded!")
    print("relation2vec  size: " + str(len(relation2vec)))

    train_triple, train_confidence = get_data_txt(trainfile)
    test_triple, test_confidence = get_data_txt(testfile)
    print('train_triple size: ', len(train_triple), 'train_confidence size: ', len(train_confidence))
    print('test_triple size: ', len(test_triple), 'test_confidence size: ', len(test_confidence))
    
    print ("dataset created!")
    out = open(datafile,'wb')
    pickle.dump([ent_vocab, ent_idex_word, rel_vocab, rel_idex_word,
                 entity2vec, entvec_k,
                 relation2vec, relvec_k,
                 train_triple, train_confidence,
                 test_triple, test_confidence,
                 max_p], out)
    out.close()