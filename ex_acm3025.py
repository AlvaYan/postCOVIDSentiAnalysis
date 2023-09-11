import time
import numpy as np
import tensorflow as tf

from gat import GAT, HeteGAT, HeteGAT_multi
import process

# 禁用gpu
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

dataset = 'acm'
featype = 'fea'
checkpt_file = os.getcwd()+'\\tranfer learning\model.ckpt'
#checkpt_file = 'pre_trained/{}/{}_allMP_multi_{}_.ckpt'.format(dataset, dataset, featype)
print('model: {}'.format(checkpt_file))
# training params
batch_size = 1
nb_epochs = 200
patience = 100
lr = 0.005  # learning rate
l2_coef = 0.001  # weight decay
# numbers of hidden units per each attention head in each layer
hid_units = [8]
n_heads = [8, 1]  # additional entry for the output layer
residual = False
nonlinearity = tf.nn.elu
model = HeteGAT_multi

print('Dataset: ' + dataset)
print('----- Opt. hyperparams -----')
print('lr: ' + str(lr))
print('l2_coef: ' + str(l2_coef))
print('----- Archi. hyperparams -----')
print('nb. layers: ' + str(len(hid_units)))
print('nb. units per layer: ' + str(hid_units))
print('nb. attention heads: ' + str(n_heads))
print('residual: ' + str(residual))
print('nonlinearity: ' + str(nonlinearity))
print('model: ' + str(model))

# jhy data
import scipy.io as sio
import scipy.sparse as sp


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

path_cur=os.getcwd()
path_data=os.getcwd()+"\\tranfer learning\\"

def load_data_dblp(path=path_data):
    import pandas as pd
    import pickle
    import scipy.sparse
    from scipy.sparse import csc_matrix
    from scipy import sparse
    import numpy as np
    import os
    import csv
    import numpy as np
    schools = ["notredame","uofm","columbia","dartmouth","UCSD","berkeley","Harvard","ucla"]
    s=schools[4]#use this index to control which network we are going to run.
    t="comment"
   
    os.chdir(path)
    name="label_emt3_"+s+"_cm.csv"
    truelabels=pd.read_csv(name,skip_blank_lines=True,header=None)
    name="feature_"+s+t+".npz"
    truefeatures=scipy.sparse.load_npz(name)
    truefeatures=truefeatures.toarray()

    N=truefeatures.shape[0]
    name="CCsym_"+s+".npz"
    dat_cc=scipy.sparse.load_npz(name);dat_cc=dat_cc.toarray();
    name="CSCsy_"+s+".npz"
    dat_csc=scipy.sparse.load_npz(name);dat_csc=dat_csc.toarray();
    name="CACsy_"+s+".npz"
    dat_cac=scipy.sparse.load_npz(name);dat_cac=dat_cac.toarray();
    rownetworks = [dat_csc,dat_cac,dat_cc]
   
    y=truelabels
    name="train_index_"+s+"_cm.p"
    train_idx = pickle.load(open(name,"rb"));train_idx=np.array(train_idx)
    name="to_be_labeled_index_"+s+"_cm.p"
    val_idx = pickle.load(open(name,"rb"));val_idx=np.array(val_idx)
    name="test_index_"+s+"_cm.p"
    test_idx = pickle.load(open(name,"rb"));test_idx=np.array(test_idx)

    train_mask = sample_mask(train_idx, y.shape[0])
    val_mask = sample_mask(val_idx, y.shape[0])
    test_mask = sample_mask(test_idx, y.shape[0])
   
    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y=y.values
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    # return selected_idx, selected_idx_2
    print('y_train:{}, y_val:{}, y_test:{}, train_idx:{}, val_idx:{}, test_idx:{}'.format(y_train.shape,
                                                                                          y_val.shape,
                                                                                          y_test.shape,
                                                                                          train_idx.shape,
                                                                                          val_idx.shape,
                                                                                          test_idx.shape))
    truefeatures_list = [truefeatures, truefeatures, truefeatures, truefeatures]# ?? why copy three times? First for center node, 2 for each metapath.    
    os.chdir(path_cur)
    return rownetworks, truefeatures_list, y_train, y_val, y_test, train_mask, val_mask, test_mask


# use adj_list as fea_list, have a try~
adj_list, fea_list, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data_dblp()


if featype == 'adj':
    fea_list = adj_list



import scipy.sparse as sp




nb_nodes = fea_list[0].shape[0]
ft_size = fea_list[0].shape[1]
nb_classes = y_train.shape[1]

# adj = adj.todense()

# features = features[np.newaxis]  # [1, nb_node, ft_size]
fea_list = [fea[np.newaxis] for fea in fea_list]
adj_list = [adj[np.newaxis] for adj in adj_list]
y_train = y_train[np.newaxis]
y_val = y_val[np.newaxis]
y_test = y_test[np.newaxis]
train_mask = train_mask[np.newaxis]
val_mask = val_mask[np.newaxis]
test_mask = test_mask[np.newaxis]

biases_list = [process.adj_to_bias(adj, [nb_nodes], nhood=1) for adj in adj_list]

print('build graph...')
#checkpt_file = 'C:\data\HAN-modified\\tranfer learning\model.ckpt'
with tf.Graph().as_default():
    with tf.name_scope('input'):
        ftr_in_list = [tf.placeholder(dtype=tf.float32,
                                      shape=(batch_size, nb_nodes, ft_size),
                                      name='ftr_in_{}'.format(i))
                       for i in range(len(fea_list))]
        bias_in_list = [tf.placeholder(dtype=tf.float32,
                                       shape=(batch_size, nb_nodes, nb_nodes),
                                       name='bias_in_{}'.format(i))
                        for i in range(len(biases_list))]
        lbl_in = tf.placeholder(dtype=tf.int32, shape=(
            batch_size, nb_nodes, nb_classes), name='lbl_in')
        msk_in = tf.placeholder(dtype=tf.int32, shape=(batch_size, nb_nodes),
                                name='msk_in')
        attn_drop = tf.placeholder(dtype=tf.float32, shape=(), name='attn_drop')
        ffd_drop = tf.placeholder(dtype=tf.float32, shape=(), name='ffd_drop')
        is_train = tf.placeholder(dtype=tf.bool, shape=(), name='is_train')
    # forward
    logits, final_embedding, att_val = model.inference(ftr_in_list, nb_classes, nb_nodes, is_train,
                                                       attn_drop, ffd_drop,
                                                       bias_mat_list=bias_in_list,
                                                       hid_units=hid_units, n_heads=n_heads,
                                                       residual=residual, activation=nonlinearity)

    # cal masked_loss
    log_resh = tf.reshape(logits, [-1, nb_classes])
    lab_resh = tf.reshape(lbl_in, [-1, nb_classes])
    msk_resh = tf.reshape(msk_in, [-1])
    loss = model.masked_softmax_cross_entropy(log_resh, lab_resh, msk_resh)
    accuracy = model.masked_accuracy(log_resh, lab_resh, msk_resh)
    # optimzie
    train_op = model.training(loss, lr, l2_coef)
    vlss_mn = np.inf
    vacc_mx = 0.0
    curr_step = 0
    saver = tf.train.Saver()
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    #new_saver = tf.train.import_meta_graph('C:\data\HAN-modified\\tranfer learning\model.ckpt.meta')


    with tf.Session(config=config) as sess:
        #sess.run(init_op)

        train_loss_avg = 0
        train_acc_avg = 0
        val_loss_avg = 0
        val_acc_avg = 0
        #os.chdir(path_cur)
        saver.restore(sess,checkpt_file)
        #new_saver.restore(sess, tf.train.latest_checkpoint('./'))
        print("Model restored.")
        print("Model restored.")
        
        #saver.restore(sess, checkpt_file)
        #print('load model from : {}'.format(checkpt_file))
        ts_size = fea_list[0].shape[0]
        ts_step = 0
        ts_loss = 0.0
        ts_acc = 0.0

        while ts_step * batch_size < ts_size:
            # fd1 = {ftr_in: features[ts_step * batch_size:(ts_step + 1) * batch_size]}
            fd1 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(ftr_in_list, fea_list)}
            fd2 = {i: d[ts_step * batch_size:(ts_step + 1) * batch_size]
                   for i, d in zip(bias_in_list, biases_list)}
            fd3 = {lbl_in: y_test[ts_step * batch_size:(ts_step + 1) * batch_size],
                   msk_in: test_mask[ts_step * batch_size:(ts_step + 1) * batch_size],
            
                   is_train: False,
                   attn_drop: 0.0,
                   ffd_drop: 0.0}
        
            fd = fd1
            fd.update(fd2)
            fd.update(fd3)
            print('q')
            loss_value_ts, acc_ts, jhy_final_embedding = sess.run([loss, accuracy, final_embedding],
                                                                  feed_dict=fd)
            ts_loss += loss_value_ts
            ts_acc += acc_ts
            ts_step += 1

        print('Test loss:', ts_loss / ts_step,
              '; Test accuracy:', ts_acc / ts_step)

        
        sess.close()
