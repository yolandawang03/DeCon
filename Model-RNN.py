import pickle
import os.path
import numpy as np
from PrecessData import get_data
from keras.layers.core import Dropout,Flatten
# from keras.layers.merge import concatenate
from keras.layers import concatenate, Concatenate, GlobalAveragePooling1D
from keras.layers import Lambda, TimeDistributed, Input, Bidirectional, Dense, Embedding, LSTM, SimpleRNN, RepeatVector,add,subtract,dot
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import optimizers
# from keras.layers.normalization import BatchNormalization
from keras.layers.normalization.batch_normalization_v1 import BatchNormalization
from keras.callbacks import Callback
from keras import regularizers
# from keras.losses import my_cross_entropy_withWeight
from keras.losses import hinge, binary_crossentropy, squared_hinge
import matplotlib.pyplot as plt
import keras.backend as K
import tensorflow as tf

def creat_Model_BiLSTM_BP(entvocabsize, relvocabsize, ent2vec, rel2vec, input_path_lenth,
                          ent_emd_dim, rel_emd_dim, max_r_len = 6):
    def repeat_and_flatten(x, n_times):
        repeated = K.repeat_elements(x, n_times, axis=1)
        return K.flatten(repeated)

    ent_h_input = Input(shape=(1,), dtype='int32')
    ent_h_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_h_input)
    
    ent_t_input = Input(shape=(1,), dtype='int32')
    ent_t_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=1,
                                   mask_zero=False, trainable=False, weights=[ent2vec])(ent_t_input)

    rel_r_input = Input(shape=(None,), dtype='int32')
    rel_r_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=None,
                           mask_zero=True, trainable=False, weights=[ent2vec])(rel_r_input)

    # 忽略padding的位置

    ent_h_embedding = Flatten()(ent_h_embedding)
    ent_t_embedding = Flatten()(ent_t_embedding)
    rel_r_input = Input(shape=(max_r_len,), dtype='int32')
    rel_r_embedding = Embedding(input_dim=entvocabsize+2, output_dim=ent_emd_dim, input_length=None,
                           mask_zero=True, trainable=False, weights=[ent2vec])(rel_r_input)
    rel_r_embedding = GlobalAveragePooling1D()(rel_r_embedding)

    ent_h_embedding = RepeatVector(2 + max_r_len)(ent_h_embedding)
    ent_t_embedding = RepeatVector(2 + max_r_len)(ent_t_embedding)
    rel_r_embedding = RepeatVector(2 + max_r_len)(rel_r_embedding)
    # ent_h_embedding_repeated = Lambda(lambda x: repeat_and_flatten(x, 2 + max_r_len))(ent_h_embedding)
    # ent_t_embedding_repeated = Lambda(lambda x: repeat_and_flatten(x, 2 + max_r_len))(ent_t_embedding)
    # rel_r_embedding_repeated = [Lambda(lambda x: repeat_and_flatten(x, 2 + max_r_len))(embedding) for embedding in rel_r_embedding]

    path_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding], axis=-1)
    path_embedding = Dropout(0.5)(path_embedding)

    path_LSTM = SimpleRNN(50, return_sequences=False)(path_embedding)
    path_LSTM = BatchNormalization()(path_LSTM)
    path_LSTM = Dropout(0.5)(path_LSTM)

    path_value = Dense(1, activation='sigmoid')(path_LSTM)

    # ------------------
    path2_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding], axis=-1)
    path2_embedding = Dropout(0.5)(path2_embedding)

    path2_LSTM = SimpleRNN(80, return_sequences=False)(path2_embedding)
    path2_LSTM = BatchNormalization()(path2_LSTM)
    path2_LSTM = Dropout(0.5)(path2_LSTM)
    path2_value = Dense(1, activation='sigmoid')(path2_LSTM)

    # ------------------
    path3_embedding = concatenate([ent_h_embedding, rel_r_embedding, ent_t_embedding], axis=-1)
    path3_embedding = Dropout(0.5)(path3_embedding)

    path3_LSTM = SimpleRNN(100, return_sequences=False)(path3_embedding)
    path3_LSTM = BatchNormalization()(path3_LSTM)
    path3_LSTM = Dropout(0.5)(path3_LSTM)
    path3_value = Dense(1, activation='sigmoid')(path3_LSTM)

    # ------------------
    BP_input = concatenate([path_value, path2_value, path3_value], axis=-1)
    BP_hidden = Dense(50)(BP_input)
    BP_hidden = Dropout(0.5)(BP_hidden)
    model = Dense(2, activation='softmax')(BP_hidden)

    Models = Model([ent_h_input, ent_t_input, rel_r_input], model)
    
    # TODO: Loss需要自己定义一下

    # def custom_loss(y_true, y_pred):
    #     print(y_true, y_pred)

    #     loss = tf.reduce_mean(tf.square(y_true - y_pred))  # 计算损失
    #     return loss

        
    # Models.compile(loss=hinge, optimizer='adam', metrics=['acc'])
    # Models.compile(loss=binary_crossentropy, optimizer='adam', metrics=['acc'])
    Models.compile(loss=squared_hinge, optimizer='adam', metrics=['acc'])

    return Models

def SelectModel(modelname, entvocabsize, relvocabsize, ent2vec, rel2vec,
                input_path_lenth,
                ent_emd_dim, rel_emd_dim):
    nn_model = None
    if modelname == 'creat_Model_BiLSTM_BP':
        nn_model = creat_Model_BiLSTM_BP(entvocabsize = entvocabsize,
                                         relvocabsize = relvocabsize,
                                         ent2vec = ent2vec, rel2vec =rel2vec,
                                         input_path_lenth = input_path_lenth,
                ent_emd_dim = ent_emd_dim, rel_emd_dim =rel_emd_dim)


    return nn_model

def train_model(modelname, datafile, modelfile, resultdir, npochos=100, batch_size=50, retrain=False, max_r = 5):
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    max_p = pickle.load(open(datafile, 'rb'))
    
    max_r = max(len(s)//2 for s in train_triple)
    
    input_train_h = np.zeros((len(train_triple),1)).astype('int32')
    input_train_t = np.zeros((len(train_triple),1)).astype('int32')
    input_train_r = np.zeros((len(train_triple),max_r)).astype('int32')
    for idx, s in enumerate(train_triple):
        input_train_h[idx,] = train_triple[idx][0]
        input_train_t[idx,] = train_triple[idx][-3]
        # input_train_r[idx,] = train_triple[idx][1::2]
        for i, r in enumerate(s[1::2]):
            input_train_r[idx, i] = r

    input_test_h = np.zeros((len(test_triple),1)).astype('int32')
    input_test_t = np.zeros((len(test_triple),1)).astype('int32')
    input_test_r = np.zeros((len(test_triple),max_r)).astype('int32')
    for idx, tri in enumerate(test_triple):
        input_test_h[idx,] = tri[0]
        input_test_t[idx,] = tri[-3]
        for i, r in enumerate(s[1::2]):
            input_test_r[idx, i] = r
    
    # TODO:max_p的取值需要关注一下
    nn_model = SelectModel(modelname, entvocabsize = len(ent_vocab), relvocabsize = len(rel_vocab),
                           ent2vec = entity2vec, rel2vec = relation2vec,
                input_path_lenth = max_p,
                ent_emd_dim = entity2vec_dim, rel_emd_dim = relation2vec_dim)
    
    if retrain:
        nn_model.load_weights(modelfile)

    nn_model.summary()
    
    epoch = 0
    save_inter = 1
    saveepoch = save_inter
    maxF = 0
    earlystopping =0
    train_acc = []
    val_acc = []
    loss = []
    val_loss = []
    while (epoch < npochos):
        epoch = epoch + 1
        
        history = nn_model.fit([np.array(input_train_h), np.array(input_train_t), np.array(input_train_r)],
            np.array(train_confidence),
                         batch_size=batch_size,
                         epochs=1,
                         validation_split=0.2,
                         shuffle=True,
                         verbose=0)
        
        train_acc.append(history.history['acc'])
        val_acc.append(history.history['val_acc'])
        loss.append(history.history['loss'])
        val_loss.append(history.history['val_loss'])

        if epoch >= saveepoch:
            # if epoch >=0:
            saveepoch += save_inter
            resultfile = resultdir+"result-"+str(saveepoch)

            print('the test result-----------------------')
            
            acc = test_model(nn_model,
                             input_test_h, input_test_t, input_test_r,
                             test_confidence, resultfile)
            
            if acc > maxF:
                earlystopping = 0
                maxF=acc
                save_model(nn_model, modelfile)
            else:
                earlystopping += 1

            print(epoch, acc, '  maxF=', maxF)

        if earlystopping >= 30:
            break
    
    plot_acc(train_acc, val_acc, save_path="./result/acc.png")
    plot_loss(loss, val_loss, save_path="./result/loss.png")
    print("train_acc: ", train_acc)
    print("val_acc: ", val_acc)
    print("loss: ", loss)
    print("val_loss: ", val_loss)
    return nn_model

def save_model(nn_model, NN_MODEL_PATH):
    nn_model.save_weights(NN_MODEL_PATH, overwrite=True)

def test_model(model,input_test_h, input_test_t, input_test_r, test_confidence, resultfile):
    total_predict_right = 0.
    total_predict = 0.
    total_right = 0.
    results = model.predict([np.array(input_test_h), np.array(input_test_t), np.array(input_test_r)], batch_size=40, verbose=0)
    
    fin0 = open(resultfile + 'train_conf0.txt','w')
    fin1 = open(resultfile + 'train_conf1.txt', 'w')
    
    for i, res in enumerate(results):
        tag = np.argmax(res)
        
        if test_confidence[i][1] == 1:
            fin1.write(str(res[1]) + '\n')
            if tag == 1:
                total_predict_right += 1.0
        else:
            fin0.write(str(res[1]) + '\n')
            if tag == 0:
                total_predict_right += 1.0
    fin0.close()
    fin1.close()
    print('total_predict_right', total_predict_right, 'len(test_confidence)', float(len(test_confidence)))
    acc = total_predict_right / float(len(test_confidence))
    return acc

def infer_model(modelname, datafile, modelfile, resultfile, batch_size=50, max_r = 5):
    ent_vocab, ent_idex_word, rel_vocab, rel_idex_word, \
    entity2vec, entity2vec_dim, \
    relation2vec, relation2vec_dim, \
    train_triple, train_confidence, \
    test_triple, test_confidence, \
    max_p = pickle.load(open(datafile, 'rb'))
    
    max_r = max(len(s)//2 for s in train_triple)
    
    input_train_h = np.zeros((len(train_triple),1)).astype('int32')
    input_train_t = np.zeros((len(train_triple),1)).astype('int32')
    input_train_r = np.zeros((len(train_triple),max_r)).astype('int32')
    for idx, s in enumerate(train_triple):
        input_train_h[idx,] = train_triple[idx][0]
        input_train_t[idx,] = train_triple[idx][-3]
        # input_train_r[idx,] = train_triple[idx][1::2]
        for i, r in enumerate(s[1::2]):
            input_train_r[idx, i] = r

    input_test_h = np.zeros((len(test_triple),1)).astype('int32')
    input_test_t = np.zeros((len(test_triple),1)).astype('int32')
    input_test_r = np.zeros((len(test_triple),max_r)).astype('int32')
    for idx, tri in enumerate(test_triple):
        input_test_h[idx,] = tri[0]
        input_test_t[idx,] = tri[-3]
        # input_test_r[idx,] = tri[1::2]
        for i, r in enumerate(s[1::2]):
            input_test_r[idx, i] = r

    model = SelectModel(modelname, entvocabsize = len(ent_vocab), relvocabsize = len(rel_vocab),
                           ent2vec = entity2vec, rel2vec = relation2vec,
                input_path_lenth = max_p,
                ent_emd_dim = entity2vec_dim, rel_emd_dim = relation2vec_dim)

    model.load_weights(modelfile)

    acc = test_model(model,
                    input_train_h, input_train_t, input_train_r,
                    train_confidence, resultfile)
    print(acc)
    

def plot_acc(acc, val_acc, savefile=True, save_path="./result/"):
    linewidth = 4
    markersize = 5
    labelsize = 20
    legendsize = 10
    pic_type = "png"  # pdf, png
    ticksize = 18
    bwith = 2
    lwith = 2.5

    # 画布背景设置
    fig, ax = plt.subplots(1, 1)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.grid(which="major", ls="--", lw=lwith, c="gray")

    # 2. 坐标轴设置
    # x_ticks = list(range(len(acc)))
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels([f"{bs}" for bs in range(1, len(acc), 10)], fontsize=ticksize)  # 设置刻度标签

    my_acc = min(acc)
    my_val_acc = min(val_acc)
    ax.set_ylim(min(my_acc[0], my_val_acc[0])-0.05, 1.05)
    print(my_acc[0], my_val_acc[0])
    y_ticks = np.arange(min(my_acc[0], my_val_acc[0]), 1.0, 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y :.1f}" for y in y_ticks], fontsize=ticksize)  # 设置刻度标签

    ax1 = ax.secondary_yaxis('right')

    # 3. 画图
    # marker 类型: o: 圆形, *: 五角星, v: 三角形, s: 正方形
    # google 配色: #f4433c 红色, #ffbc32 黄色, #0aa858 绿色, #2d85f0 蓝色
    # #e21e26 红色，#34a047, 3e9243 绿色， #ebbc80 淡黄， #eea5a5 粉色
    ax.plot(
        list(range(len(acc))),
        acc,
        label="train",
        linewidth=linewidth,
        # c="#C53A32",
        marker="o",
        ls = "-",
        markersize=markersize,
    )
    ax.plot(
        list(range(len(val_acc))),
        val_acc,
        label="test",
        linewidth=linewidth,
        # c="#518DEC",
        marker="v",
        ls = "-",
        markersize=markersize,
    )

    ax.set_xlabel("Batch Num", fontsize=labelsize)
    ax.set_ylabel("Train Acc", fontsize=labelsize)
    ax1.set_ylabel('Test Acc', fontsize=labelsize)
    ax.legend(fontsize=legendsize, loc="lower left")
    plt.savefig(save_path)
    
def plot_loss(acc, val_acc, savefile=True, save_path="./result/"):
    linewidth = 4
    markersize = 5
    labelsize = 20
    legendsize = 15
    pic_type = "png"  # pdf, png
    ticksize = 18
    bwith = 2
    lwith = 2.5

    # 画布背景设置
    fig, ax = plt.subplots(1, 1)
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)
    ax.grid(which="major", ls="--", lw=lwith, c="gray")

    # 2. 坐标轴设置
    # x_ticks = list(range(len(acc)))
    # ax.set_xticks(x_ticks)
    # ax.set_xticklabels([f"{bs}" for bs in range(len(acc))], fontsize=ticksize)  # 设置刻度标签

    my_acc = max(acc)
    my_val_acc = max(val_acc)
    ax.set_ylim(0., max(my_acc[0], my_val_acc[0])+0.05)
    print(my_acc[0], my_val_acc[0], max(my_acc[0], my_val_acc[0]))
    y_ticks = np.arange(0., max(my_acc[0], my_val_acc[0]), 0.2)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{y :.1f}" for y in y_ticks], fontsize=ticksize)  # 设置刻度标签

    ax1 = ax.secondary_yaxis('right')

    # 3. 画图
    # marker 类型: o: 圆形, *: 五角星, v: 三角形, s: 正方形
    # google 配色: #f4433c 红色, #ffbc32 黄色, #0aa858 绿色, #2d85f0 蓝色
    # #e21e26 红色，#34a047, 3e9243 绿色， #ebbc80 淡黄， #eea5a5 粉色
    ax.plot(
        list(range(len(acc))),
        acc,
        label="train",
        linewidth=linewidth,
        # c="#C53A32",
        marker="o",
        ls = "-",
        markersize=markersize,
    )
    ax.plot(
        list(range(len(val_acc))),
        val_acc,
        label="test",
        linewidth=linewidth,
        # c="#518DEC",
        marker="v",
        ls = "-",
        markersize=markersize,
    )

    ax.set_xlabel("Batch Num", fontsize=labelsize)
    ax.set_ylabel("Train loss", fontsize=labelsize)
    ax1.set_ylabel('Test loss', fontsize=labelsize)
    ax.legend(fontsize=legendsize, loc="lower left")
    plt.savefig(save_path)

if __name__ == "__main__":

    modelname = 'creat_Model_BiLSTM_BP'
    
    file_mydata = "../0may_use"
    
    # entity2idfile = file_mydata + "/WN18RR/entity2id.txt"
    # relation2idfile = file_mydata + "/WN18RR/relation2id.txt"

    # entity2vecfile =file_mydata + "/WN18RR/WN18RR_TransE_Entity2Vec_100.txt"
    # relation2vecfile = file_mydata + "/WN18RR/WN18RR_TransE_Relation2Vec_100.txt"
    
    # trainfile = file_mydata + "/WN18RR/1conflict/WN18RR_train_conflict_random.txt"
    # testfile = file_mydata + "/WN18RR/1conflict/WN18RR_test_conflict_random.txt"
    # trainfile = file_mydata + "/WN18RR/3conflic't/WN18RR_train_random_3conflict.txt"
    # testfile = file_mydata + "/WN18RR/3conflict/WN18RR_test_random_3conflict.txt"

    # entity2idfile = file_mydata + "/FB15K/entity2id.txt"
    # relation2idfile = file_mydata + "/FB15K/relation2id.txt"
    
    # entity2vecfile =file_mydata + "/FB15K/FB15K_TransE_Entity2Vec_100.txt"
    # relation2vecfile = file_mydata + "/FB15K/FB15K_TransE_Relation2Vec_100.txt"
    
    # # trainfile = file_mydata + "/FB15K/FB15K_train_conflict_random.txt"
    # # testfile = file_mydata + "/FB15K/FB15K_test_conflict_random.txt"
    # trainfile = file_mydata + "/FB15K/3conflict/FB15K_train_conflict_random.txt"
    # testfile = file_mydata + "/FB15K/3conflict/FB15K_test_conflict_random.txt"
    
    entity2idfile = file_mydata + "/YAGO3-10/entity2id.txt"
    relation2idfile = file_mydata + "/YAGO3-10/relation2id.txt"
    
    entity2vecfile =file_mydata + "/YAGO3-10/YAGO3-10_TransE_Entity2Vec_100.txt"
    relation2vecfile = file_mydata + "/YAGO3-10/YAGO3-10_TransE_Relation2Vec_100.txt"
    
    trainfile = file_mydata + "/YAGO3-10/YAGO3-10_train_conflict_random_small.txt"
    testfile = file_mydata + "/YAGO3-10/YAGO3-10_test_conflict_random_small.txt"
    # trainfile = file_mydata + "/YAGO3-10/3conflict/YAGO3-10_train_random_3conflict.txt"
    # testfile = file_mydata + "/YAGO3-10/3conflict/YAGO3-10_test_random_3conflict.txt"
    
    datafile = "./RNN-model/YAGO3-10_test_TransE.pkl"
    modelfile = "./RNN-model/YAGO3-10_ALL_model1_TransE_20231111175717.h5"
    resultdir = "./result/"
    resultdir = "./result/Model1_model_YAGO3-10_3conflict_TransE_---"
    
    # path_file = file_data + "/Path_4/"
    # entityRank = file_data + "/ResourceRank_4/"
    
    batch_size = 64
    retrain = False
    Test = True
    valid = False
    Label = False
    if not os.path.exists(datafile):
        print("Precess data....")
        
        get_data(entity2idfile=entity2idfile, relation2idfile=relation2idfile,
             entity2vecfile=entity2vecfile, relation2vecfile=relation2vecfile, w2v_k=100,
             trainfile=trainfile, testfile=testfile,
             max_p=5,
             datafile=datafile)
        
    if not os.path.exists(modelfile):
        print("data has extisted: " + datafile)
        print("Training model....")
        print(modelfile)
        train_model(modelname, datafile, modelfile, resultdir,
                        npochos=200, batch_size=batch_size, retrain=False)
    else:
        if retrain:
            print("ReTraining EE model....")
            train_model(modelname, datafile, modelfile, resultdir,
                            npochos=200, batch_size=batch_size, retrain=retrain)
    if Test:
        print("test EE model....")
        print(modelfile)
        infer_model(modelname, datafile, modelfile, resultdir, batch_size=batch_size)