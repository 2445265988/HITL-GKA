import csv
import math
import time

import numpy as np

from .Load import *
import psutil
import wandb
import matplotlib.pyplot as plt
from .Init import *
from .Test import get_hits
import scipy
import json
from .util import no_weighted_adj
import scipy.spatial as spatial
import os
from .preprocess import generate_2steps_path
import tensorflow as tf
# import keras.backend as K
# import wandb



def rfunc(KG, e):
    head = {}
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
            tail[tri[1]].add(tri[2])
    r_num = len(head)
    head_r = np.zeros((e, r_num))
    tail_r = np.zeros((e, r_num))
    r_mat_ind = []
    r_mat_val = []
    for tri in KG:
        # print("this kg")
        # print(tri[0],tri[1],tri[2]) Fh3.1415926
        head_r[tri[0]][tri[1]] = 1
        tail_r[tri[2]][tri[1]] = 1
        r_mat_ind.append([tri[0], tri[2]])
        r_mat_val.append(tri[1])
    r_mat = tf.SparseTensor(
        indices=r_mat_ind, values=r_mat_val, dense_shape=[e, e])

    return head, tail, head_r, tail_r, r_mat


def get_mat(e, KG):
    du = [{e_id} for e_id in range(e)]
    for tri in KG:
        if tri[0] != tri[2]:
            du[tri[0]].add(tri[2])
            du[tri[2]].add(tri[0])
    du = [len(d) for d in du]
    M = {}
    for tri in KG:
        if tri[0] == tri[2]:
            continue
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = 1
        else:
            pass
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = 1
        else:
            pass

    for i in range(e):
        M[(i, i)] = 1
    return M, du


# get a sparse tensor based on relational triples
def get_sparse_tensor(e, KG):
    print('getting a sparse tensor...')
    M, du = get_mat(e, KG)
    ind = []
    val = []
    M_arr = np.zeros((e, e))
    for fir, sec in M:
        ind.append((sec, fir))
        val.append(M[(fir, sec)] / math.sqrt(du[fir]) / math.sqrt(du[sec]))
        M_arr[fir][sec] = 1.0
    M = tf.SparseTensor(indices=ind, values=val, dense_shape=[e, e])

    return M, M_arr


# add a layer
# def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
#     inlayer = tf.nn.dropout(inlayer, 1 - dropout)
#     dimension_in = inlayer.get_shape().as_list()[-1]
#     W = init([dimension_in, dimension])
#     print('adding a diag layer...')
#
#     node_features = tf.matmul(inlayer, W)
#     aggregated_features = tf.sparse_tensor_dense_matmul(M, node_features)
#     if act_func is None:
#         return aggregated_features
#     else:
#         return act_func(aggregated_features)

def add_diag_layer(inlayer, dimension, M, act_func, dropout=0.0, init=ones):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a diag layer...')
    w0 = init([1, dimension])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.multiply(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_full_layer(inlayer, dimension_in, dimension_out, M, act_func, dropout=0.0, init=glorot):
    inlayer = tf.nn.dropout(inlayer, 1 - dropout)
    print('adding a full layer...')
    w0 = init([dimension_in, dimension_out])
    tosum = tf.sparse_tensor_dense_matmul(M, tf.matmul(inlayer, w0))
    if act_func is None:
        return tosum
    else:
        return act_func(tosum)


def add_sparse_att_layer(inlayer, dual_layer, r_mat, act_func, e):
    dual_transform = tf.reshape(tf.layers.conv1d(
        tf.expand_dims(dual_layer, 0), 1, 1), (-1, 1))
    logits = tf.reshape(tf.nn.embedding_lookup(
        dual_transform, r_mat.values), [-1])
    print('adding sparse attention layer...')
    lrelu = tf.SparseTensor(indices=r_mat.indices,
                            values=tf.nn.leaky_relu(logits),
                            dense_shape=(r_mat.dense_shape))
    coefs = tf.sparse_softmax(lrelu)
    vals = tf.sparse_tensor_dense_matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def add_dual_att_layer(inlayer, inlayer2, adj_mat, act_func, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(inlayer2, 0), hid_dim, 1)
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    print('adding dual attention layer...')
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    logits = tf.multiply(adj_tensor, logits)
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    vals = tf.matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def add_self_att_layer(inlayer, adj_mat, act_func, hid_dim):
    in_fts = tf.layers.conv1d(tf.expand_dims(
        inlayer, 0), hid_dim, 1, use_bias=False)
    f_1 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    f_2 = tf.reshape(tf.layers.conv1d(in_fts, 1, 1), (-1, 1))
    logits = f_1 + tf.transpose(f_2)
    print('adding self attention layer...')
    adj_tensor = tf.constant(adj_mat, dtype=tf.float32)
    logits = tf.multiply(adj_tensor, logits)
    bias_mat = -1e9 * (1.0 - (adj_mat > 0))
    coefs = tf.nn.softmax(tf.nn.leaky_relu(logits) + bias_mat)

    vals = tf.matmul(coefs, inlayer)
    if act_func is None:
        return vals
    else:
        return act_func(vals)


def highway(layer1, layer2, dimension):
    kernel_gate = glorot([dimension,dimension])
    bias_gate = zeros([dimension])
    transform_gate = tf.matmul(layer1, kernel_gate) + bias_gate
    transform_gate = tf.nn.sigmoid(transform_gate)
    carry_gate = 1.0 - transform_gate
    return transform_gate*layer2 + carry_gate * layer1


def compute_r(inlayer, head_r, tail_r, dimension):
    head_l = tf.transpose(tf.constant(head_r, dtype=tf.float32))
    tail_l = tf.transpose(tf.constant(tail_r, dtype=tf.float32))
    L = tf.matmul(head_l, inlayer) / \
        tf.expand_dims(tf.reduce_sum(head_l, axis=-1), -1)
    R = tf.matmul(tail_l, inlayer) / \
        tf.expand_dims(tf.reduce_sum(tail_l, axis=-1), -1)
    r_embeddings = tf.concat([L, R], axis=-1)
    return r_embeddings


def get_dual_input(inlayer, head, tail, head_r, tail_r, dimension):
    dual_X = compute_r(inlayer, head_r, tail_r, dimension)
    print('computing the dual input...')
    count_r = len(head)
    dual_A = np.zeros((count_r, count_r))
    for i in range(count_r):
        for j in range(count_r):
            a_h = len(head[i] & head[j]) / len(head[i] | head[j])
            a_t = len(tail[i] & tail[j]) / len(tail[i] | tail[j])
            dual_A[i][j] = a_h + a_t
    return dual_X, dual_A


def get_input_layer(e, dimension, lang):
    print('adding the primal input layer...')
    with open(file='data/' + lang + '_en/' + lang + '_vectorList.json', mode='r', encoding='utf-8') as f:
        embedding_list = json.load(f)
        print(len(embedding_list), 'rows,', len(embedding_list[0]), 'columns.')
    input_embeddings = tf.convert_to_tensor(embedding_list)
    ent_embeddings = tf.Variable(input_embeddings)
    return tf.nn.l2_normalize(ent_embeddings, 1)

def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = tf.reduce_sum(y_pred ** 2, axis=1, keepdims=True)
    dot_product = tf.matmul(y_true, y_pred, transpose_b=True)
    square_distance = tf.sqrt(square_pred + tf.transpose(square_pred) - 2 * dot_product)
    loss = y_true * square_distance + (1 - y_true) * tf.square(tf.maximum(margin - square_distance, 0))
    return tf.reduce_mean(loss)


"""对比损失"""
def get_loss_compare(outlayer, ILL, k):
    margin=1.0
    gamma=1.0
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 对比损失
    positive_distance = tf.reduce_sum(tf.square(left_x - right_x), 1)

    # 正则化项，这里使用L2正则化
    positive_loss = tf.reduce_sum(tf.nn.relu(positive_distance))

    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1)
    negative_loss_1 = tf.reduce_mean(tf.nn.relu(B - gamma))
    neg2_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg2_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg2_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    B2 = tf.reduce_sum(tf.square(neg2_l_x - neg2_r_x), 1)
    negative_loss_2 = tf.reduce_mean(tf.nn.relu(B2 - gamma))

    # 总损失是正样本损失和两组负样本损失的和
    loss = (positive_loss + negative_loss_1 + negative_loss_2) / (3 * k * t)
    return loss

def get_loss_dualaloss(outlayer, ILL, k, node_size):
    def squared_dist(x):
        A, B = x
        row_norms_A = tf.reduce_sum(tf.square(A), axis=1)
        row_norms_A = tf.reshape(row_norms_A, [-1, 1])  # Column vector.
        row_norms_B = tf.reduce_sum(tf.square(B), axis=1)
        row_norms_B = tf.reshape(row_norms_B, [1, -1])  # Row vector.
        return row_norms_A + row_norms_B - 2 * tf.matmul(A, B, transpose_b=True)

    def reduce_std(tensor, axis=None, keepdims=False):
        """
        计算张量在指定维度上的元素的标准偏差。

        参数:
        - tensor: 要计算标准偏差的张量。
        - axis: 沿着哪个维度计算标准偏差，默认为None，表示计算所有元素的标准偏差。
        - keepdims: 是否保留维度。

        返回:
        - 每个指定维度的标准偏差张量。
        """
        # 计算均值
        mean = tf.reduce_mean(tensor, axis=axis, keepdims=keepdims)

        # 计算方差
        squared_diff = tf.square(tensor - mean)

        # 计算方差
        variance = tf.reduce_mean(squared_diff, axis=axis, keepdims=keepdims)

        # 计算标准偏差
        std_dev = tf.sqrt(variance)

        return std_dev

    gamma=1.0
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 维度为【9000，300】
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)

    # 计算绝对差之和，度量正样本对之间的距离
    # 度为【9000，1】
    positive_distance = tf.reduce_sum(tf.square(left_x - right_x), 1,keepdims=True)
    # with tf.Session() as sess:
    #     print("实际维度 of positive_distance:", sess.run(tf.shape(positive_distance)))
    # 计算左侧的负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    with tf.Session() as sess:
        print("实际维度 of neg_l_x:", sess.run(tf.shape(neg_l_x)))
    with tf.Session() as sess:
        print("实际维度 of neg_r_x:", sess.run(tf.shape(neg_r_x)))
    # negative_distance_1 = squared_dist([neg_l_x,neg_r_x])
    negative_distance_1 = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1, keepdims=True)
    # 归一化副样本1损失
    neg1_dist_mean = tf.reduce_mean(negative_distance_1)
    neg1_dist_std = reduce_std(negative_distance_1)
    normalized_neg1_dist = (negative_distance_1-neg1_dist_mean)/(neg1_dist_std+1e-8)
    left_loss=positive_distance-normalized_neg1_dist+gamma
    # negative_distance_1 = tf.reduce_sum(negative_distance_1,1,keepdims=True)
    # with tf.Session() as sess:
    #     print("实际维度 of negative_distance_1:", sess.run(tf.shape(negative_distance_1)))
    # negative_distance_1 = tf.reshape(negative_distance_1, [t, k])
    neg2_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg2_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg2_l_x = tf.nn.embedding_lookup(outlayer, neg2_left)
    neg2_r_x = tf.nn.embedding_lookup(outlayer, neg2_right)
    negative_distance_2 = tf.reduce_sum(tf.square(neg2_l_x - neg2_r_x), 1, keepdims=True)

    # 归一化副样本1损失
    neg2_dist_mean = tf.reduce_mean(negative_distance_2)
    neg2_dist_std = reduce_std(negative_distance_2)
    normalized_neg2_dist = (negative_distance_2 - neg2_dist_mean) / (neg2_dist_std + 1e-8)
    right_loss = positive_distance - normalized_neg2_dist + gamma

    # negative_distance_2 = squared_dist([neg2_l_x,neg2_r_x])
    negative_distance_2 = tf.reduce_sum(tf.square(neg2_l_x - neg2_r_x), 1, keepdims=True)
    # negative_distance_2 = tf.reshape(negative_distance_2, [t, k])
    # 边界损失函数，使用relu激活函数和边界gamma
    # left_loss=positive_distance-negative_distance_1+gamma
    # right_loss=positive_distance-negative_distance_2+gamma
    # # 创建独热编码矩阵
    # l_one_hot = tf.one_hot(indices=left, depth=k)
    # r_one_hot = tf.one_hot(indices=right, depth=k)
    # left_loss = left_loss *(1 - l_one_hot- r_one_hot)
    # #
    # right_loss = right_loss * (1 - l_one_hot- r_one_hot)
    # right_loss = (right_loss - tf.stop_gradient(tf.reduce_mean(right_loss, axis=-1, keepdims=True))) / tf.stop_gradient(
    #     right_loss)
    # left_loss = (left_loss - tf.stop_gradient(tf.reduce_mean(left_loss, axis=-1, keepdims=True))) / tf.stop_gradient(
    #     left_loss)

    lamb, tau = 30, 10
    neg1_loss = tf.reduce_logsumexp(lamb * left_loss + tau, axis=-1)
    neg2_loss = tf.reduce_logsumexp(lamb * right_loss + tau, axis=-1)
    # pos_loss = tf.reduce_logsumexp(lamb * positive_distance + tau, axis=-1)
    # positive_loss=K.asd

    # 总损失是正样本损失和两组负样本损失的和
    # loss = (l_loss + r_loss) / (2 * k * t)
    return tf.reduce_mean(neg1_loss + neg2_loss )

"""温度缩放"""
def get_loss_wen(outlayer, ILL, gamma, k, temperature=1):
    print('getting loss...')
    # 提取实体ID
    left_ids = ILL[:, 0]
    right_ids = ILL[:, 1]
    t = len(ILL)

    # 获取正样本嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left_ids)
    right_x = tf.nn.embedding_lookup(outlayer, right_ids)

    # 计算正样本对之间的平方距离
    pos_dist = tf.reduce_sum(tf.abs(left_x - right_x),1)

    # 计算损失的辅助函数
    def calc_loss_with_neg_samples(pos_dist, neg_left, neg_right, temperature):
        # 获取负样本嵌入向量
        neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
        neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)

        # 计算负样本对之间的平方距离
        neg_dist = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), axis=1)
        neg_dist = -tf.reshape(neg_dist, [t, k])

        # 温度缩放和平移，防止数值过小
        lamb, tau = 30, 10
        scaled_pos_dist = lamb*pos_dist+tau
        scaled_neg_dist = lamb*neg_dist+tau
        # 使用logsumexp技巧来稳定损失计算
        loss = tf.reduce_logsumexp(tf.nn.leaky_relu(tf.add(scaled_neg_dist, tf.reshape(scaled_pos_dist, [t, 1]))))
        return loss
    # 计算第一种负样本的损失
    neg_left1 = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right1 = tf.placeholder(tf.int32, [t * k], "neg_right")
    loss1 = calc_loss_with_neg_samples(pos_dist, neg_left1, neg_right1, temperature)
    # 计算第二种负样本的损失
    neg_left2 = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right2 = tf.placeholder(tf.int32, [t * k], "neg2_right")
    loss2 = calc_loss_with_neg_samples(pos_dist, neg_left2, neg_right2, temperature)
    # 合并两种负样本的损失
    total_loss = (loss1 +loss2) / (2.0 * k * t)
    # # 平均损失
    # final_loss = total_loss / (k * t) + gamma
    return total_loss

"""平方损失"""
def get_loss_square(outlayer, ILL, gamma, k):
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 计算绝对差之和，度量正样本对之间的距离
    A = tf.reduce_sum(tf.square(left_x - right_x), 1)
    # A = tf.nn.softplus(-A))
    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    neg_distance = neg_l_x - neg_r_x
    B = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1)
    # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
    C = - tf.reshape(B, [t, k])
    # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
    D = A + gamma
    # L1 计算正样本对的损失，L2 计算负样本对的损失
    L1 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.square(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))

    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


# def get_loss(outlayer, ILL, gamma, k):
#     print('getting loss...')
#     # 提取实体ID
#     left = ILL[:, 0]
#     right = ILL[:, 1]
#     t = len(ILL)
#     # 对应的嵌入向量
#     left_x = tf.nn.embedding_lookup(outlayer, left)
#     right_x = tf.nn.embedding_lookup(outlayer, right)
#     # 计算绝对差之和，度量正样本对之间的距离
#     A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
#     # 计算左负样本对之间的距离
#     neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
#     neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
#     neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
#     neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
#     with tf.Session() as sess:
#         print("实际维度 of negative_distance_1:", sess.run(tf.shape(neg_r_x)))
#
#     shape_of_neg_r_x = tf.shape(neg_r_x)
#     with tf.Session() as sess:
#         print("实际维度 of neg_r_x:", sess.run(shape_of_neg_r_x))
#     B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
#     # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
#     C = - tf.reshape(B, [t, k])
#     # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
#     D = A + gamma
#     # L1 计算正样本对的损失，L2 计算负样本对的损失
#     L1 = tf.reduce_logsumexp(tf.add(C, tf.reshape(D, [t, 1])), axis=-1)
#     neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
#     neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
#     neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
#     neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
#     B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
#     C = - tf.reshape(B, [t, k])
#     L2 = tf.reduce_logsumexp(tf.add(C, tf.reshape(D, [t, 1])), axis=-1)
#     return (tf.reduce_mean(L1) + tf.reduce_mean(L2)) / (2.0 * t)
"""边际排序损失"""
def get_loss_rank(outlayer, ILL, gamma, k):
    print('getting loss...')
    # 提取实体ID
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 计算绝对差之和，度量正样本对之间的距离
    A = tf.reduce_sum(tf.multiply(left_x,right_x), 1)
    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.multiply(neg_l_x,neg_r_x), 1)
    # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
    C = tf.reshape(B, [t, k])
    # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
    D = -A+gamma
    # L1 计算正样本对的损失，L2 计算负样本对的损失
    L1 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.multiply(neg_l_x,neg_r_x), 1)
    C = tf.reshape(B, [t, k])
    L2 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)

def get_loss(outlayer, ILL, gamma, k):
    print('getting loss...')
    # 提取实体ID
    # left = ILL[:, 0]
    left = tf.placeholder(tf.int32, [None], "pos_left")
    # right = ILL[:, 1]
    right = tf.placeholder(tf.int32, [None], "pos_right")

    t = tf.placeholder(tf.float32, [], "t")  # 假设 t 是浮点数
    t = tf.cast(t, tf.int32)  # 将 t 转换为整数类型
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    # 计算绝对差之和，度量正样本对之间的距离
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    # 计算左负样本对之间的距离
    print("t*k in build :",t,k,t*k)
    neg_left = tf.placeholder(tf.int32, [None], "neg_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)

    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    # C 通过 B 计算负样本对的距离并重塑为 [t, k] 形状
    C = - tf.reshape(B, [t, k])
    # D 是一个常数项，等于 A 加上 gamma，重塑为 [t, 1] 形状。
    D = A + gamma
    # L1 计算正样本对的损失，L2 计算负样本对的损失
    L1 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = tf.placeholder(tf.int32, [None], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = - tf.reshape(B, [t, k])
    L2 = tf.nn.leaky_relu(tf.add(C, tf.reshape(D, [t, 1])))
    # 假设 k 和 t 是整数类型的张量
    k_float = tf.cast(k, tf.float32)
    t_float = tf.cast(t, tf.float32)

    # 将整数 2 转换为浮点数
    two_float = tf.constant(2.0, dtype=tf.float32)
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (two_float * k_float * t_float)


def get_loss_act2(outlayer, ILL, gamma):
    print('getting loss...')
    # 提取实体ID
    left = tf.placeholder(tf.int32, [None], "pos_left")
    right = tf.placeholder(tf.int32, [None], "pos_right")
    pos_len = tf.placeholder(tf.float32, [], "pos_len")
    neg_len = tf.placeholder(tf.float32, [], "neg_len")
    total_samples = pos_len+neg_len
    pos_wight = total_samples/pos_len
    neg_wight = total_samples / neg_len
    # pos_wight = pos_len / total_samples
    # neg_wight = neg_len / total_samples
    print("pos_wight:",pos_wight)
    print("neg_wight:",neg_wight)
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    pos_distance = tf.reduce_sum(tf.abs(left_x - right_x), 1)

    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [None], "neg_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    neg_distance = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)

    pos_loss = tf.nn.leaky_relu(pos_distance)
    neg_loss = tf.nn.leaky_relu(gamma-neg_distance)
    weighted_pos_loss = tf.reduce_sum(pos_loss) * pos_wight
    weighted_neg_loss = tf.reduce_sum(neg_loss) * neg_wight
    loss = (weighted_neg_loss+weighted_pos_loss)/(pos_len+neg_len)
    print("loss:", loss)
    return loss

def get_loss_act3(outlayer, ILL, gamma,k):
    print('getting loss...')
    # 提取实体ID
    left = tf.placeholder(tf.int32, [None], "pos_left")
    right = tf.placeholder(tf.int32, [None], "pos_right")
    pos_len = tf.placeholder(tf.int32, [], "pos_len")
    neg_len = tf.placeholder(tf.int32, [], "neg_len")
    total_samples = pos_len+neg_len
    pos_wight = total_samples/pos_len
    neg_wight = total_samples / neg_len
    # pos_wight = pos_len / total_samples
    # neg_wight = neg_len / total_samples
    print("pos_wight:",tf.cast(pos_wight,tf.float32))
    print("neg_wight:",tf.cast(neg_wight,tf.float32))
    # 对应的嵌入向量
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    pos_distance = tf.reduce_sum(tf.abs(left_x - right_x), 1)

    # 计算左负样本对之间的距离
    neg_left = tf.placeholder(tf.int32, [None], "neg_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    neg_distance = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = -tf.reshape(neg_distance, [pos_len, k])
    D = pos_distance+gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [pos_len, 1])))
    neg_left = tf.placeholder(tf.int32, [None], "neg2_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    neg_distance = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = -tf.reshape(neg_distance, [pos_len, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [pos_len, 1])))
    # pos_loss = tf.nn.leaky_relu(pos_distance)
    # neg_loss = tf.nn.leaky_relu(gamma-neg_distance)
    # weighted_pos_loss = tf.reduce_sum(pos_loss) * pos_wight
    # weighted_neg_loss = tf.reduce_sum(neg_loss) * neg_wight
    # loss = (weighted_neg_loss+weighted_pos_loss)/(pos_len+neg_len)
    # print("loss:", loss)
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2))/(2.0*tf.cast(pos_len,tf.float32)*tf.cast(k,tf.float32))

def get_loss_act(outlayer, ILL, gamma):
    print('getting loss...')
    # 创建占位符
    pos_left = tf.placeholder(tf.int32, [None], "pos_left")
    pos_right = tf.placeholder(tf.int32, [None], "pos_right")
    neg_left = tf.placeholder(tf.int32, [None], "neg_left")
    neg_right = tf.placeholder(tf.int32, [None], "neg_right")
    pos_len = tf.placeholder(tf.float32, [], "pos_len")
    neg_len = tf.placeholder(tf.float32, [], "neg_len")

    # 定义计算距离的函数
    def compute_distance(positive):
        left_x = tf.nn.embedding_lookup(outlayer, positive[0])
        right_x = tf.nn.embedding_lookup(outlayer, positive[1])
        dis = tf.reduce_sum(tf.nn.leaky_relu((tf.reduce_sum(tf.abs(left_x - right_x), 1))))
        return dis
    # 计算正样本对之间的距离
    A = tf.cond(
        tf.reduce_any(tf.not_equal(pos_left, 0)),
        lambda: compute_distance([pos_left, pos_right]),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )
    # 计算负样本对之间的距离
    B = tf.cond(
        tf.reduce_any(tf.not_equal(neg_left, 0)),
        lambda: compute_distance([neg_left, neg_right]),
        lambda: tf.constant(0.0, dtype=tf.float32)
    )
    # 归一化最终的损失
    loss = A + gamma * B
    return loss/(pos_len + neg_len)

def merge_adj_matrices(adj1, adj2):
    # 确保两个邻接矩阵的形状相同


    # 将两个邻接矩阵相加
    merged_adj = adj1 + adj2

    # 将所有大于 1 的元素置为 1，确保最终的邻接矩阵中每个元素的值不会大于 1
    merged_adj.data = np.minimum(merged_adj.data, np.ones_like(merged_adj.data))

    return merged_adj



def build(dimension, act_func, alpha, beta, gamma, k, lang, e, ILL, KG):
    tf.reset_default_graph()
    # 从输入数据中获取原始节点的特征表示
    primal_X_0 = get_input_layer(e, dimension, lang)
    heads = set([triple[0] for triple in KG])
    tails = set([triple[2] for triple in KG])
    ents = heads | tails


    one_adj,_ = no_weighted_adj(len(ents), KG, is_two_adj= False)
    two_hop_triples = generate_2steps_path(KG)
    two_adj,_ = no_weighted_adj(len(ents), two_hop_triples, is_two_adj= False)

    # 创建 SparseTensor 对象
    sparse_tensor_one = tf.SparseTensor(indices=one_adj[0], values=one_adj[1], dense_shape=one_adj[2])
    # sparse_tensor_two = tf.SparseTensor(indices=two_adj[0], values=two_adj[1], dense_shape=two_adj[2])
    # M_sparse = tf.sparse_add(sparse_tensor_one, sparse_tensor_two)
    M_sparse = sparse_tensor_one
    M = tf.cast(M_sparse, tf.float32)
    node_size = len(ents)


    # M, M_arr = get_sparse_tensor(e, KG)
    # 生成头实体、尾实体、头关系、尾关系以及关系矩阵。
    head, tail, head_r, tail_r, r_mat = rfunc(KG, e)
    print('first interaction...')
    # 获取双图输入层和双图邻接矩阵。
    dual_X_1, dual_A_1 = get_dual_input(
        primal_X_0, head, tail, head_r, tail_r, dimension)
    # 对双图进行自注意力操作，得到双图的隐藏表示。
    dual_H_1 = add_self_att_layer(dual_X_1, dual_A_1, tf.nn.relu, 600)
    # 对原始图进行稀疏注意力操作，得到原始图的隐藏表示。
    primal_H_1 = add_sparse_att_layer(
        primal_X_0, dual_H_1, r_mat, tf.nn.relu, e)
    # 使用残差连接更新原始节点特征表示。
    primal_X_1 = primal_X_0 + alpha * primal_H_1

    print('second interaction...')
    # 与第一个交互阶段类似，对更新后的原始节点特征再次进行交互操作。
    dual_X_2, dual_A_2 = get_dual_input(
        primal_X_1, head, tail, head_r, tail_r, dimension)
    dual_H_2 = add_dual_att_layer(
        dual_H_1, dual_X_2, dual_A_2, tf.nn.relu, 600)
    primal_H_2 = add_sparse_att_layer(
        primal_X_1, dual_H_2, r_mat, tf.nn.relu, e)
    primal_X_2 = primal_X_0 + beta * primal_H_2

    print('gcn layers...')
    # shape = tf.shape(primal_X_2)
    # with tf.Session() as sess:
    #     dimension1 = sess.run(shape)
    # print("dimension维度")
    # print(dimension1)
    gcn_layer_1 = add_diag_layer(
        primal_X_2, dimension, M, act_func, dropout=0.1)
    gcn_layer_1 = highway(primal_X_2, gcn_layer_1, dimension)
    gcn_layer_2 = add_diag_layer(
        gcn_layer_1, dimension, M, act_func, dropout=0.1)
    # dimension2 = tf.shape(gcn_layer_1)
    # with tf.Session() as sess:
    #     dimension2 = sess.run(shape)
    #
    # print("dimension维度:", dimension2)
    output_layer = highway(gcn_layer_1, gcn_layer_2, dimension)
    # loss = get_loss_rank(output_layer, ILL, gamma, k)
    loss = get_loss_act3(output_layer, ILL, gamma,k)
    # loss = get_loss_compare(output_layer, ILL, k)
    # loss = get_loss_wen(output_layer, ILL, gamma, k)
    # loss = get_loss_dualaloss(output_layer, ILL, k, node_size)
    print(loss)
    return output_layer, loss


# get negative samples
# 负样本生成-可改进
def get_neg(ILL, output_layer, k):
    neg = []
    t = len(ILL)
    ILL_vec = np.array([output_layer[e1] for e1 in ILL])
    KG_vec = np.array(output_layer)
    # 得到每个实体与字典中所有实体的曼哈顿距离
    sim = spatial.distance.cdist(ILL_vec, KG_vec, metric='cityblock')
    for i in range(t):
        # 得到每个实体与字典中所有实体的前k个相似实体
        rank = sim[i, :].argsort()
        # rank = sim[i, :].argsort()[:: , -1]
        neg.append(rank[0:k])
    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg

def compute_validation_loss_old(session, output_layer, ILL_valid, gamma, k):
    # 假设 ILL_valid 是类似 ILL 的结构，其中包含需要验证的实体对
    t_valid = len(ILL_valid)
    ILL_valid = np.array(ILL_valid)

    left_ids = ILL_valid[:, 0]
    right_ids = ILL_valid[:, 1]

    left_embeddings = tf.nn.embedding_lookup(output_layer, left_ids)
    right_embeddings = tf.nn.embedding_lookup(output_layer, right_ids)

    # 计算基本损失，这里简单使用 L1 距离
    basic_loss = tf.reduce_mean(tf.abs(left_embeddings - right_embeddings),1)
    # 计算负采样损失，这部分可能需要根据你的具体情况调整
    # 确保 ILL_valid[:, 0] 是整数类型
    ILL_valid_left_ids = ILL_valid[:, 0].astype(np.int32)
    # 生成负采样的左实体ID
    L = np.ones((t_valid, k)) * ILL_valid_left_ids.reshape((t_valid, 1))
    neg_left_ids = L.reshape((t_valid * k,)).astype(np.int32)
    outvec = session.run(output_layer)
    # 将 output_layer 保存到文本文件
    # 生成唯一的文件名，例如通过添加时间戳
    # timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    # filename = f'output_layer_{timestamp}.txt'
    # # 将 output_layer 保存到文本文件
    # np.savetxt(filename, outvec, fmt='%f')
    neg_right_ids = get_neg(ILL_valid[:, 0], outvec, k).astype(np.int32)

    # neg_left_ids = np.random.randint(0, output_layer.shape[0], size=(t_valid * k,))
    # neg_right_ids = np.random.randint(0, output_layer.shape[0], size=(t_valid * k,))
    neg_left_embeddings = tf.nn.embedding_lookup(output_layer, neg_left_ids)
    neg_right_embeddings = tf.nn.embedding_lookup(output_layer, neg_right_ids)
    neg_distance = tf.reduce_mean(tf.abs(neg_left_embeddings - neg_right_embeddings),1)
    # change 2
    C = - tf.reshape(neg_distance, [t_valid, k])
    D = basic_loss + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t_valid, 1])))

    # pos_loss = tf.nn.leaky_relu(basic_loss)
    # neg_loss = tf.nn.leaky_relu(gamma - neg_distance)
    # pos_len = len(left_ids)
    # neg_len = len(neg_left_ids)
    # print("验证集正负样本长度：",pos_len, neg_len)
    # total_samples = pos_len + neg_len
    # pos_wight = total_samples / pos_len
    # neg_wight = total_samples / neg_len
    # weighted_pos_loss = tf.reduce_sum(pos_loss)*pos_wight
    # weighted_neg_loss = tf.reduce_sum(neg_loss)*neg_wight
    # loss = (weighted_neg_loss + weighted_pos_loss) / (pos_len+neg_len)
    # 组合损失并计算总损失
    # total_loss = basic_loss + gamma * neg_loss
    # total_loss_val = session.run(loss)  # 计算具体的损失值
    total_loss_val = (tf.reduce_sum(L1)) / (t_valid * k)
    print("验证集loss:", session.run(total_loss_val))
    return session.run(total_loss_val)

def compute_validation_loss_old2(session, output_layer, ILL_valid, gamma, k):
    # 假设 ILL_valid 是类似 ILL 的结构，其中包含需要验证的实体对
    t_valid = len(ILL_valid)
    ILL_valid = np.array(ILL_valid)

    left_ids = ILL_valid[:, 0]
    right_ids = ILL_valid[:, 1]

    left_embeddings = tf.nn.embedding_lookup(output_layer, left_ids)
    right_embeddings = tf.nn.embedding_lookup(output_layer, right_ids)

    # 计算基本损失，这里简单使用 L1 距离
    basic_loss = tf.reduce_mean(tf.abs(left_embeddings - right_embeddings))

    # 计算负采样损失，这部分可能需要根据你的具体情况调整
    neg_left_ids = np.random.randint(0, output_layer.shape[0], size=(t_valid * k,))
    neg_right_ids = np.random.randint(0, output_layer.shape[0], size=(t_valid * k,))
    neg_left_embeddings = tf.nn.embedding_lookup(output_layer, neg_left_ids)
    neg_right_embeddings = tf.nn.embedding_lookup(output_layer, neg_right_ids)
    neg_loss = tf.reduce_mean(tf.abs(neg_left_embeddings - neg_right_embeddings))
    # 组合损失并计算总损失
    total_loss = basic_loss + gamma * neg_loss
    total_loss_val = session.run(total_loss)  # 计算具体的损失值
    print("验证集loss:", total_loss_val)
    return total_loss_val


def compute_validation_loss(sess, output_layer, ILL_valid, gamma, k):
    # 假设 ILL_valid 是类似 ILL 的结构，其中包含需要验证的实体对
    t_valid = len(ILL_valid)
    ILL_valid = np.array(ILL_valid)
    left_ids = ILL_valid[:, 0]
    right_ids = ILL_valid[:, 1]
    # 计算负采样损失
    t_valid = len(ILL_valid)
    ILL_valid = np.array(ILL_valid)
    v_L = np.ones((t_valid, k)) * (ILL_valid[:, 0].reshape((t_valid, 1)))
    v_neg_left = v_L.reshape((t_valid * k,))
    v_neg_left = tf.convert_to_tensor(v_neg_left, dtype=tf.int32)
    v_L = np.ones((t_valid, k)) * (ILL_valid[:, 1].reshape((t_valid, 1)))
    v_neg2_right = v_L.reshape((t_valid * k,))
    v_neg2_right = tf.convert_to_tensor(v_neg2_right, dtype=tf.int32)
    out = sess.run(output_layer)
    v_neg2_left = get_neg(ILL_valid[:, 1], out, k)
    v_neg_right = get_neg(ILL_valid[:, 0], out, k)
    # 计算基本损失，这里简单使用 L1 距离
    left_embeddings = tf.nn.embedding_lookup(output_layer, left_ids)
    right_embeddings = tf.nn.embedding_lookup(output_layer, right_ids)
    A = tf.reduce_mean(tf.square(left_embeddings - right_embeddings),1)


    neg_left_embeddings = tf.nn.embedding_lookup(output_layer,v_neg_left)
    neg_right_embeddings = tf.nn.embedding_lookup(output_layer, v_neg_right)
    neg2_left_embeddings = tf.nn.embedding_lookup(output_layer, v_neg2_left)
    neg2_right_embeddings = tf.nn.embedding_lookup(output_layer, v_neg2_right)

    B = tf.reduce_sum(tf.square(neg_left_embeddings - neg_right_embeddings),1)
    C = -tf.reshape(B, [t_valid, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t_valid, 1])))
    B = tf.reduce_sum(tf.square(neg2_left_embeddings - neg2_right_embeddings),1)
    C = -tf.reshape(B, [t_valid, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t_valid, 1])))
    # 计算总损失
    total_loss = (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t_valid)
    total_loss_val = sess.run(total_loss)
    return total_loss_val


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.01):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.wait = 0
        self.stopped = False

    def check(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < (self.best_loss - self.min_delta):
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped = True



def training(output_layer, loss, learning_rate, epochs, ILL, e, k, test,validation, gamma, save_path, restore=False):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    print('initializing training...')
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    saver = tf.train.Saver()  # 创建 Saver 对象
    print('running...')
    J = []
    t = len(ILL)
    ILL = np.array(ILL)
    L = np.ones((t, k)) * (ILL[:, 0].reshape((t, 1)))
    neg_left = L.reshape((t * k,))
    L = np.ones((t, k)) * (ILL[:, 1].reshape((t, 1)))
    neg2_right = L.reshape((t * k,))
    out_test = sess.run(output_layer)
    print("建图结束,开始评估初始测试集")
    get_hits(out_test, test)
    print("--------------------")
    for i in range(600):
        if i % 10 == 0:
            out = sess.run(output_layer)
            neg2_left = get_neg(ILL[:, 1], out, k)
            neg_right = get_neg(ILL[:, 0], out, k)
            feeddict = {"neg_left:0": neg_left,
                        "neg_right:0": neg_right,
                        "neg2_left:0": neg2_left,
                        "neg2_right:0": neg2_right,
                        "pos_left:0": ILL[:, 0],
                        "pos_right:0": ILL[:, 1],
                        "pos_len:0": t,
                        "neg_len:0": t * k,
                        }
        _, th = sess.run([train_step, loss], feed_dict=feeddict)
        if i % 10 == 0:
            th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
            J.append(th)
            get_hits(outvec, test)
            # val_loss = compute_validation_loss_old(sess, output_layer, validation, gamma, k)
            # early_stopping.check(val_loss)
            # if early_stopping.stopped:
            #     print("Early stopping triggered at epoch:", i + 1)
            #     break
        print('%d/%d' % (i + 1, epochs), 'epochs...', th)
    try:
        print(f"正在保存模型到{save_path}...............")
        print("大约train中的图结构：")
        # print(tf.get_default_graph().as_graph_def())
        print("----------------------")
        saver.save(sess, os.path.join(save_path, 'model_DRNA.ckpt'))  # 尝试保存模型参数
        print(f"成功保存模型到{save_path}")
    except Exception as e:
        print(f"保存模型时发生错误: {e}")
    # saver.save(sess, os.path.join(save_path, 'model_DRNA_bw.ckpt'))  # 保存模型参数
    outvec = sess.run(output_layer)
    sess.close()
    print(f"成功保存模型到{save_path}")
    print("训练完成，开始评估测试集...............")

    # 保存vec到文件
    return outvec, J


def select_qurey_fun1(X , vec, uncertainty_threshold):
    Lvec = np.array([vec[e1] for e1, e2 in X])
    Rvec = np.array([vec[e2] for e1, e2 in X])
    simL2R = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
    # print('simL2R:', simL2R)
    simR2L = spatial.distance.cdist(Rvec, Lvec, metric='cityblock')
    # print('simR2L:', simR2L)
    rankingsL2R = np.argsort(simL2R, axis=1)
    rankingsR2L = np.argsort(simR2L, axis=1)
    # print('rankingsL2R:', rankingsL2R)
    # print('rankingsR2L:', rankingsR2L)
    # 初始化不确定性度量字典
    uncertainty_measures = {}
    post = {}
    neg = {}
    for i, (e1, e2) in enumerate(X):
        # 计算 e1 和 e2 的相似度排名
        rank_e1 = np.where(rankingsL2R[i] == i)[0][0]+1
        # print('rank_e1:', rank_e1)
        rank_e2 = np.where(rankingsR2L[i] == i)[0][0]+1
        # print('rank_e2:', rank_e2)
        # 计算 e1 和 e2 的距离
        # sim1 = 1-1 / (1 + np.exp(-(simL2R[i][i])))
        sim1 = simL2R[i][i]
        # print('sim1:', sim1)
        # sim2 = 1-1 / (1 + np.exp(-(simR2L[i][i])))
        sim2 = simR2L[i][i]
        # print('sim2:', sim2)
        # rank越接近小,则不确定性越小
        rank = np.mean([rank_e1*sim1, rank_e2*sim2])
        # print('rank:', rank)
        rank_inverse = 1 / (rank + 1e-10)
        # print('rank_inverse:', rank_inverse)

        prob_dist = np.array([1-rank_inverse, rank_inverse])
        # print('prob_dist:', prob_dist)
        # 计算熵
        uncertainty = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        uncertainty = 1 / (1 + np.exp(-uncertainty))

        if uncertainty>uncertainty_threshold:
            uncertainty_measures[(e1,e2)] = [uncertainty,simL2R[i][i]]
        else:
            if rank_e1 < 10 or rank_e2 < 10:
                post[(e1,e2)] = [uncertainty,simL2R[i][i]]
            else:
                neg[(e1,e2)] = [uncertainty,simL2R[i][i]]
        # 将不确定性度量限制在 0 到 1 之间
    return uncertainty_measures,post,neg

def ranking_normalization_function(x,k=-np.log(2)**4):
    return 1/(1+np.exp(-k*(x-20)))
# 分配权重
def select_qurey_fun2(X , vec, uncertainty_threshold,weight_sim,weight_rank):
    Lvec = np.array([vec[e1] for e1,e2 in X])
    Rvec = np.array([vec[e2] for e1,e2 in X])
    simL2R = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')

    # print('simL2R:', simL2R)
    simR2L = spatial.distance.cdist(Rvec, Lvec, metric='cityblock')
    # print('simR2L:', simR2L)
    rankingsL2R = np.argsort(simL2R, axis=1)
    rankingsR2L = np.argsort(simR2L, axis=1)
    # print('rankingsL2R:', rankingsL2R)
    # print('rankingsR2L:', rankingsR2L)
    # 初始化不确定性度量字典
    uncertainty_measures = {}
    post = {}
    neg = {}
    for i,(e1,e2) in enumerate(X):
        # 计算 e1 和 e2 的相似度排名
        # print("(e1,e2):", e1, e2)
        # print("vec[e1]:", vec[e1])
        rank_e1 = np.where(rankingsL2R[i] == i)[0][0]+1
        rank_e1_nor = ranking_normalization_function(rank_e1)
        # print("rank_e1:", rank_e1)
        # print('rank_e1_nor:', rank_e1_nor)
        rank_e2 = np.where(rankingsR2L[i] == i)[0][0]+1
        rank_e2_nor = ranking_normalization_function(rank_e2)
        # print("rank_e2:", rank_e2)
        # print('rank_e2_nor:', rank_e2_nor)
        # 归一化排名 (1-0)，排名越接近小,约接近1
        rank = (rank_e1_nor+rank_e2_nor)/2
        # print('rank:', rank)
        # 计算 e1 和 e2 的距离
        # sim1 = 1-1 / (1 + np.exp(-(simL2R[i][i])))
        dist = simL2R[i][i]
        # print("dist:",dist)

        # 归一化距离,距离(1-0)
        s =-np.log(0.5)/4.5
        dist =np.exp(-s*dist)
        # print('np.exp(-s*dist) :', dist)
        # # sim2 = 1-1 / (1 + np.exp(-(simR2L[i][i])))
        # sim2 = simR2L[i][i]
        # sim2 = sim2
        # print('sim2:', sim2)
        rank_inverse = rank*weight_rank+dist*weight_sim
        # print('rank_inverse:', rank_inverse)

        prob_dist = np.array([1-rank_inverse, rank_inverse])
        # print('prob_dist:', prob_dist)
        # 计算熵
        # uncertainty的值域为0-1,0为最佳，1为最不确定
        uncertainty = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
        # uncertainty = 1 / (1 + np.exp(-uncertainty))

        if uncertainty>uncertainty_threshold:
            # print("uncertainty:", uncertainty)
            uncertainty_measures[(e1,e2)] = [uncertainty,simL2R[i][i]]
        else:
            if rank_e1 < 10 and rank_e2 < 10:
                post[(e1,e2)] = [uncertainty,simL2R[i][i]]
            else:
                neg[(e1,e2)] = [uncertainty,simL2R[i][i]]
        # 将不确定性度量限制在 0 到 1 之间
    return uncertainty_measures,post,neg


# 基于newfile创建一个新文件，该文件的前缀为newfile
def create_new_file(nowfile,newfilename,filetype):
    parent_dir = os.path.dirname(nowfile)
    # 获取 X_pool_file 的当前文件名
    file_name = os.path.basename(nowfile)
    # csv_file_path = X_pool_file+'标注文件_id.csv'  # 创建一个 CSV 文件
    csv_file_path = os.path.join(parent_dir, f'{os.path.splitext(file_name)[0]}{newfilename}.{filetype}')
    return csv_file_path


def tipswrite2csv(X_pool_file,dict1,pos_left,pos_right,neg_left,neg_right):
    # X_pool_parent_dir = os.path.dirname(X_pool_file)
    # # 获取 X_pool_file 的当前文件名
    # X_pool_file_name = os.path.basename(X_pool_file)
    pos = list(zip(pos_left,pos_right))
    neg = list(zip(neg_left,neg_right))
    # csv_file_path = X_pool_file+'标注文件_id.csv'  # 创建一个 CSV 文件
    # csv_file_path = os.path.join(X_pool_parent_dir,f'{os.path.splitext(X_pool_file_name)[0]}标注文件_id.csv')
    # csv_file_path_name = os.path.join(X_pool_parent_dir,f'{os.path.splitext(X_pool_file_name)[0]}标注文件_name.csv') # 创建一个 CSV 文件
    csv_file_path = create_new_file(X_pool_file,'标注文件_id','csv')
    csv_file_path_name = create_new_file(X_pool_file,'标注文件_name','csv') # 创建一个 CSV 文件
    # 打开文件并写入
    with open(csv_file_path, 'a', newline='') as csvfile,open(csv_file_path_name, 'a', newline='') as csvfile_name:
        writer_id = csv.writer(csvfile)
        writer_name = csv.writer(csvfile_name)
        # 写入标题行
        writer_id.writerow(['entity1', 'entity2', 'label'])
        writer_name.writerow(['实体1', '实体2', '标签'])
        # 写入正样本对
        if len(pos)>0:
            for left_id, right_id in pos:
                writer_id.writerow([left_id, right_id, 1])
                writer_name.writerow([dict1[left_id], dict1[right_id], 1])
        # 写入负样本对
        if len(neg)>0:
            for left_id, right_id in neg:
                writer_id.writerow([left_id, right_id, 0])
                writer_name.writerow([dict1[left_id], dict1[right_id], 0])


def write_data_to_file(key_pos_pair, data_dict, file_name, pairtype):
    # 获取要写入文件的数据
    data_to_write0 = data_dict[key_pos_pair[0]]
    data_to_write1 = data_dict[key_pos_pair[1]]
    # 打开文件并写入数据
    with open(file_name, 'a') as file:  # 使用'a'模式以追加的方式写入文件
       file.write(f"{data_to_write0}, {data_to_write1},{pairtype}\n")  # 写入位置和数据

# 查看样本分类精度
def sample_classify_acc(pool,pairs,dict,filename,pooltype):
    acc=0
    wrong = 0
    if pooltype == 'pos':
        for key_pair in pool.keys():
            if key_pair in pairs:
                acc = acc + 1
            else:
                # print(key_pair, pool[key_pair], "实体1：", dict[key_pair[0]], "实体2：",
                #       dict[key_pair[1]])
                wrong = wrong + 1
                write_data_to_file(key_pair, dict,filename,'1')
    elif pooltype == 'neg':
        for key_pair in pool.keys():
            if key_pair not in pairs:
                acc = acc + 1
            else:
                # print(key_pair, pool[key_pair], "实体1：",dict[key_pair[0]], "实体2：",
                #       dict[key_pair[1]])
                wrong = wrong + 1
                write_data_to_file(key_pair, dict,filename,'0')

    return acc,wrong

def funt_training(X_pool_file,output_layer, loss, learning_rate, epochs, ILL,dict1, e, k, test,validation, gamma, save_path,modle_name, restore=False):
    test_data = test
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    print('initializing training...')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  # 创建 Saver 对象
    sess = tf.Session()
    sess.run(init)
    print('running...')
    if restore and os.path.exists(save_path):
        saver.restore(sess, os.path.join(save_path, modle_name))  # 恢复模型参数
        print("Model restored from", save_path,modle_name)
    else:
        print('starting fresh...')
    J = []
    t = len(ILL)
    outvec = sess.run(output_layer)
    # 初始化列表
    # neg_left = []
    # neg_right = []
    # pos_left = []
    # pos_right = []
    uncertainty_threshold = 0.75
    # query_pool,pos_pool,neg_pool = select_qurey_fun1(ILL, out,uncertainty_threshold)
    while(1):
        # 初始化列表
        neg_left = []
        neg_right = []
        pos_left = []
        pos_right = []
        query_pool, pos_pool, neg_pool = select_qurey_fun2(ILL, outvec, uncertainty_threshold,0.3,0.7)
        if(len(query_pool) == 0):
            print("---------------已无待标注数据，退出程序---------------")
            break
        print("query_pool:", len(query_pool),query_pool)
        print("pos_pool:", len(pos_pool),pos_pool)
        print("neg_pool:", len(neg_pool),neg_pool)
        # 计算置信度精度start
        print("开始计算置信度精度...............")
        print("当前置信度阈值：", uncertainty_threshold)
        print("小样本数据池总数：", len(ILL))
        pairs = loadfile("./data/j3_en/sup_ent_ids", 2)
        print("成功加载对齐字典")
        pos_acc, neg_acc = 0,0
        print("正在查找分类错误的正样本")
        for key_pos_pair in pos_pool.keys():
            if key_pos_pair in pairs:
                pos_acc = pos_acc + 1
            else:
                print(key_pos_pair,pos_pool[key_pos_pair],"实体1：",dict1[key_pos_pair[0]],"实体2：",dict1[key_pos_pair[1]])
        print("----------正在查找分类错误的负样本-----------------")
        for key_neg_pair in neg_pool.keys():
            if key_neg_pair not in pairs:
                neg_acc = neg_acc + 1
            else:
                print(key_neg_pair,neg_pool[key_neg_pair],"实体1：",dict1[key_neg_pair[0]],"实体2：",dict1[key_neg_pair[1]])
        print("------------开始置信度评估--------------")
        print("置信度阈值：", uncertainty_threshold)
        print("正确分类的正样本：", pos_acc)
        print("正确分类的负样本：", neg_acc)
        if len(pos_pool) > 0:
            print("正样本正确分类精度：", pos_acc / len(pos_pool))
        else:
            print("正样本池为空，无法计算精度")
        if len(neg_pool) > 0:
            print("负样本正确分类精度：", neg_acc / len(neg_pool))
        else:
            print("负样本池为空，无法计算精度")
        if len(pos_pool) + len(neg_pool) > 0:
            print("置信度精度：", (pos_acc + neg_acc) / (len(pos_pool) + len(neg_pool)))
        else:
            print("小样本池为空，无法计算置信度精度")
        print("------------结束计算置信度精度...............")

        # 计算置信度精度end
        for pos, value in pos_pool.items():
            pos_left.append(pos[0])
            pos_right.append(pos[1])
            ILL.remove(pos)
        for neg, value in neg_pool.items():
            neg_left.append(neg[0])
            neg_right.append(neg[1])
            ILL.remove(neg)
        for query, value in query_pool.items():
            print(query)
            print(f'不确定样本：')
            print(f"实体1: {dict1[query[0]]}, 实体2: {dict1[query[1]]}")
            print(f"模型预测实体距离：{value[1]}")

            while True:
                human_label = input("请输入这个样本的正确标签（0不匹配或1匹配）或暂停标记exit，然后按回车键： ")
                if human_label not in ["0", "1","exit"]:
                    print("输入无效，请输入0（不匹配）或1（匹配）。")
                else:

                    human_label = human_label  # 转换为整数
                    break  # 输入有效，退出循环
            print("已人工矫正样本")
            if human_label == '0':
                neg_left.append(query[0])
                neg_right.append(query[1])
                print("已人工矫fu样本")
            elif human_label == '1':
                pos_left.append(query[0])
                pos_right.append(query[1])
                print("已人工矫正样本")
            elif human_label == "exit":
                break
            ILL.remove(query)

        tipswrite2csv(X_pool_file,dict1,pos_left,pos_right,neg_left,neg_right)

        # 如果负样本过少，则进行负样本增强
        # if len(neg_left)/(len(pos_left)+len(neg_left))< 0.5:
        # print("进行负样本增强---------------")
        # need_neg_num = len(pos_left)
        # # neg_left = pos_left[:need_neg_num]
        # # neg_right = get_neg(neg_left, out, 1)
        # L = np.ones((need_neg_num, k)) * (np.array(pos_left).reshape((need_neg_num, 1)))
        # new_neg_left = L.reshape((need_neg_num * k,))
        # new_neg_right = get_neg(pos_left, outvec, k)
        # # neg_left = new_neg_left
        # # neg_right = new_neg_right
        # L = np.ones((need_neg_num, k)) * (np.array(pos_right).reshape((need_neg_num, 1)))
        # new_neg_right2 = L.reshape((need_neg_num * k,))
        # new_neg_left2 = get_neg(pos_right, outvec, k)
        # # neg_left2 = new_neg_left2
        # # neg_right2 = new_neg_right2
        #
        # # 将新的负样本添加到原有的neg_left中
        # # neg_left.extend(new_neg_left)
        # # neg_right.extend(new_neg_right)
        # print("增强后的负1样本数量：", len(new_neg_left))
        # print("增强后的负2样本数量：", len(new_neg_left2))
        #
        #
        #
        # feeddict = {"neg_left:0": np.array(new_neg_left),
        #             "neg_right:0": np.array(new_neg_right),
        #             "neg2_left:0": np.array(new_neg_left2),
        #             "neg2_right:0": np.array(new_neg_right2),
        #             "pos_left:0": np.array(pos_left),
        #             "pos_right:0": np.array(pos_right),
        #             "pos_len:0": len(pos_left),
        #             "neg_len:0": len(neg_left),
        #             }
        # print("本轮迭代训练样本数量(正样本):", len(pos_left))
        # print("本轮迭代训练样本数量(负样本1):", len(new_neg_left))
        # print("本轮迭代训练样本数量(负样本2):", len(new_neg_left2))
        epo = int(input("请输入本轮迭代次数："))
        for i in range(epo):
            if i % 10 == 0:
                print("进行本轮负样本增强---------------")
                need_neg_num = len(pos_left)
                # neg_left = pos_left[:need_neg_num]
                # neg_right = get_neg(neg_left, out, 1)
                L = np.ones((need_neg_num, k)) * (np.array(pos_left).reshape((need_neg_num, 1)))
                new_neg_left = L.reshape((need_neg_num * k,))
                new_neg_right = get_neg(pos_left, outvec, k)
                # neg_left = new_neg_left
                # neg_right = new_neg_right
                L = np.ones((need_neg_num, k)) * (np.array(pos_right).reshape((need_neg_num, 1)))
                new_neg_right2 = L.reshape((need_neg_num * k,))
                new_neg_left2 = get_neg(pos_right, outvec, k)
                # neg_left2 = new_neg_left2
                # neg_right2 = new_neg_right2

                # 将新的负样本添加到原有的neg_left中
                # neg_left.extend(new_neg_left)
                # neg_right.extend(new_neg_right)
                print("增强后的负1样本数量：", len(new_neg_left))
                print("增强后的负2样本数量：", len(new_neg_left2))

                feeddict = {"neg_left:0": np.array(new_neg_left),
                            "neg_right:0": np.array(new_neg_right),
                            "neg2_left:0": np.array(new_neg_left2),
                            "neg2_right:0": np.array(new_neg_right2),
                            "pos_left:0": np.array(pos_left),
                            "pos_right:0": np.array(pos_right),
                            "pos_len:0": len(pos_left),
                            "neg_len:0": len(neg_left),
                            }
                print("本轮迭代训练样本数量(正样本):", len(pos_left))
                print("本轮迭代训练样本数量(负样本1):", len(new_neg_left))
                print("本轮迭代训练样本数量(负样本2):", len(new_neg_left2))
            _, th = sess.run([train_step, loss], feed_dict=feeddict)
            if i % 10 == 0:
                th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
                J.append(th)
                get_hits(outvec, test)
                # val_loss = compute_validation_loss_old(sess, output_layer, validation, gamma, k)
                # early_stopping.check(val_loss)
                # if early_stopping.stopped:
                #     print("Early stopping triggered at epoch:", i + 1)
                #     # saver.save(sess, os.path.join(save_path, 'model_DRNA.ckpt'))  # 保存模型参数
                #     # print(f"成功保存模型到{save_path}")
                #     break
            print('%d/%d' % (i + 1, epochs), 'epochs...', th)
        outvec = sess.run(output_layer)
        print("训练完成，开始评估测试集...............")
        print(test)
        get_hits(outvec, test)
        choose2save = input("是否保存模型？(y/n)")
        if choose2save == 'y':
            new_model_name = input("请输入新模型名字（model_DRNA.ckpt）：")
            save_path_full=os.path.join(save_path, new_model_name)
            saver.save(sess,save_path_full )  # 保存模型参数
            # # 确保文件被写入磁盘
            # for ext in ['.meta', '.data-00000-of-00001', '.index']:
            #     file_path = save_path_full + ext
            #     if os.path.exists(file_path):
            #         with open(file_path, 'a') as f:
            #             # 打开文件并写入空字符串，这会触发flush
            #             f.write('')
            #             # 在文件关闭前确保数据被写入磁盘
            #             os.fsync(f.fileno())
            print(f"成功保存模型到{save_path}")
        else:
            print("模型未保存")
    # 保存vec到文件
    return outvec, J ,saver


def test_uncertain_acc(uncertainty_threshold,weight1,weight2,output_layer, loss, learning_rate, epochs, ILL,dict1, e, k, test,validation, gamma, save_path,modle_name, restore=False):
    pairs = loadfile("./data/j3_en/sup_ent_ids", 2)
    print('initializing training...')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  # 创建 Saver 对象
    sess = tf.Session()
    sess.run(init)
    print('running...')
    if restore and os.path.exists(save_path):
        saver.restore(sess, os.path.join(save_path, modle_name))  # 恢复模型参数
        print("Model restored from", save_path, modle_name)
    else:
        print('starting fresh...')
    out = sess.run(output_layer)
    # uncertainty_threshold = 0.5
    query_pool, pos_pool, neg_pool = select_qurey_fun2(ILL, out, uncertainty_threshold, weight1, weight2)
    # print("query_pool:", len(query_pool))
    # print("pos_pool:", len(pos_pool))
    # print("neg_pool:", len(neg_pool))
    # 计算置信度精度start
    print("开始计算置信度精度...............")

    pairs = loadfile("./data/j3_en/sup_ent_ids", 2)
    print("成功加载对齐字典")
    pos_acc, neg_acc = 0, 0
    print("正在查找分类错误的正样本")
    for key_pos_pair in pos_pool.keys():
        if key_pos_pair in pairs:
            pos_acc = pos_acc + 1
        else:
            print(key_pos_pair, pos_pool[key_pos_pair], "实体1：", dict1[key_pos_pair[0]], "实体2：",
                  dict1[key_pos_pair[1]])
    print("正在查找分类错误的负样本")
    for key_neg_pair in neg_pool.keys():
        if key_neg_pair not in pairs:
            neg_acc = neg_acc + 1
        else:
            print(key_neg_pair, neg_pool[key_neg_pair], "实体1：", dict1[key_neg_pair[0]], "实体2：",
                  dict1[key_neg_pair[1]])
    print("--------------------------")
    print("小样本数据池总数：", len(ILL))
    print("置信度阈值：", uncertainty_threshold)
    print("query_pool:", len(query_pool))
    print("pos_pool:", len(pos_pool))
    print("neg_pool:", len(neg_pool))
    print("正确分类的正样本：", pos_acc)
    print("正确分类的负样本：", neg_acc)
    print("正样本正确分类精度：", pos_acc / len(pos_pool))
    if len(neg_pool) is not 0:
        print("负样本正确分类精度：", neg_acc / len(neg_pool))
    print("置信度精度：", (pos_acc + neg_acc) / (len(pos_pool) + len(neg_pool)))
    print("结束计算置信度精度...............")
    # 计算置信度精度end
    return (pos_acc + neg_acc) / (len(pos_pool) + len(neg_pool))


# 只有标注数据训练
def only_noting(X_pool_file,output_layer, loss, learning_rate, epochs, ILL,dict1, e, k, test,validation, gamma, save_path,modle_name, restore=False):
    ILL = set(ILL)
    test_data = test
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    early_stopping = EarlyStopping(patience=5, min_delta=0.01)
    print('initializing training...')
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()  # 创建 Saver 对象
    sess = tf.Session()
    sess.run(init)
    print('running...')
    if restore and os.path.exists(save_path):
        saver.restore(sess, os.path.join(save_path, modle_name))  # 恢复模型参数
        print("Model restored from", save_path,modle_name)
    else:
        print('starting fresh...')
    J = []
    t = len(ILL)
    outvec = sess.run(output_layer)
    # 初始化列表
    neg_left = []
    neg_right = []
    pos_left = []
    pos_right = []
    uncertainty_threshold = 0.75
    wrong_file = create_new_file(X_pool_file, "_wrong_data",'csv')
    # query_pool,pos_pool,neg_pool = select_qurey_fun1(ILL, out,uncertainty_threshold)
    while(1):
        neg_left = []
        neg_right = []
        pos_left = []
        pos_right = []
        if len(ILL) == 0:
            print("---------------数据池无数据，退出程序---------------")
            break
        query_pool, pos_pool, neg_pool = select_qurey_fun2(ILL, outvec, uncertainty_threshold,0.3,0.7)
        if(len(query_pool) == 0):
            print("---------------已无待标注数据，退出程序---------------")
            break
        print("query_pool:", len(query_pool))
        print("pos_pool:", len(pos_pool))
        print("neg_pool:", len(neg_pool))
        # 计算置信度精度start
        print("开始计算置信度精度...............")
        print("当前置信度阈值：", uncertainty_threshold)
        print("小样本数据池总数：", len(ILL))
        pairs = loadfile("./data/j3_en/sup_ent_ids", 2)
        print("成功加载对齐字典")
        pos_acc, neg_acc,pos_wrong,neg_wrong = 0,0,0,0
        print("正在查找分类错误的正样本:")
        pos_acc,pos_wrong = sample_classify_acc(pos_pool, pairs, dict1,wrong_file,'pos')
        print(pos_wrong)
        print("正在查找分类错误的负样本")
        neg_acc,neg_wrong = sample_classify_acc(neg_pool, pairs, dict1, wrong_file,'neg')
        print(neg_wrong)
        print("本轮所有错误分类的样本数量：",pos_wrong+neg_wrong)
        # for key_neg_pair in neg_pool.keys():
        #     if key_neg_pair not in pairs:
        #         neg_acc = neg_acc + 1
        #     else:
        #         print(key_neg_pair,neg_pool[key_neg_pair],"实体1：",dict1[key_neg_pair[0]],"实体2：",dict1[key_neg_pair[1]])
        print("------------开始置信度评估--------------")
        print("置信度阈值：", uncertainty_threshold)
        print("正确分类的正样本：", pos_acc)
        print("正确分类的负样本：", neg_acc)
        if len(pos_pool) > 0:
            print("正样本正确分类精度：", pos_acc / len(pos_pool))
        else:
            print("正样本池为空，无法计算精度")
        if len(neg_pool) > 0:
            print("负样本正确分类精度：", neg_acc / len(neg_pool))
        else:
            print("负样本池为空，无法计算精度")
        if len(pos_pool) + len(neg_pool) > 0:
            print("置信度精度：", (pos_acc + neg_acc) / (len(pos_pool) + len(neg_pool)))
        else:
            print("小样本池为空，无法计算置信度精度")
        print("------------结束计算置信度精度...............")

        # 计算置信度精度end
        for pos, value in pos_pool.items():
            pos_left.append(pos[0])
            pos_right.append(pos[1])
            ILL.remove(pos)
        for neg, value in neg_pool.items():
            neg_left.append(neg[0])
            neg_right.append(neg[1])
            ILL.remove(neg)
        for query,value in query_pool.items():
            print(query)
            print(f'不确定样本：')
            print(f"实体1: {dict1[query[0]]},实体2: {dict1[query[1]]}")
            print(f"模型预测实体距离：{value[1]}")

            while True:
                human_label = input("请输入这个样本的正确标签（0不匹配或1匹配）或暂停标记exit，然后按回车键： ")
                if human_label not in ["0", "1","exit"]:
                    print("输入无效，请输入0（不匹配）或1（匹配）。")
                else:

                    human_label = human_label  # 转换为整数
                    break  # 输入有效，退出循环
            print("已人工矫正样本")
            if human_label == '0':
                neg_left.append(query[0])
                neg_right.append(query[1])
                print("已人工矫fu样本")
            elif human_label == '1':
                pos_left.append(query[0])
                pos_right.append(query[1])
                print("已人工矫正样本")
            elif human_label == "exit":
                break
            ILL.remove(query)

        tipswrite2csv(X_pool_file,dict1,pos_left,pos_right,neg_left,neg_right)
        chooseiftrain = input("是否继续训练模型？(y/n): ")
        if chooseiftrain == 'y':

            # 如果负样本过少，则进行负样本增强
            # if len(neg_left)/(len(pos_left)+len(neg_left))< 0.5:
            print("进行负样本增强---------------")
            need_neg_num = len(pos_left)
            # neg_left = pos_left[:need_neg_num]
            # neg_right = get_neg(neg_left, out, 1)
            L = np.ones((need_neg_num, k)) * (np.array(pos_left).reshape((need_neg_num, 1)))
            new_neg_left = L.reshape((need_neg_num * k,))
            new_neg_right = get_neg(pos_left, outvec, k)
            # neg_left = new_neg_left
            # neg_right = new_neg_right
            L = np.ones((need_neg_num, k)) * (np.array(pos_right).reshape((need_neg_num, 1)))
            new_neg_right2 = L.reshape((need_neg_num * k,))
            new_neg_left2 = get_neg(pos_right, outvec, k)
            # neg_left2 = new_neg_left2
            # neg_right2 = new_neg_right2

            # 将新的负样本添加到原有的neg_left中
            # neg_left.extend(new_neg_left)
            # neg_right.extend(new_neg_right)
            print("增强后的负1样本数量：", len(new_neg_left))
            print("增强后的负2样本数量：", len(new_neg_left2))

            feeddict = {"neg_left:0": np.array(new_neg_left),
                        "neg_right:0": np.array(new_neg_right),
                        "neg2_left:0": np.array(new_neg_left2),
                        "neg2_right:0": np.array(new_neg_right2),
                        "pos_left:0": np.array(pos_left),
                        "pos_right:0": np.array(pos_right),
                        "pos_len:0": len(pos_left),
                        "neg_len:0": len(neg_left),
                        }

            for i in range(epochs):
                _, th = sess.run([train_step, loss], feed_dict=feeddict)
                if i % 10 == 0:
                    th, outvec = sess.run([loss, output_layer], feed_dict=feeddict)
                    J.append(th)
                    get_hits(outvec, validation)
                    val_loss = compute_validation_loss_old(sess, output_layer, validation, gamma, k)
                    early_stopping.check(val_loss)
                    if early_stopping.stopped:
                        print("Early stopping triggered at epoch:", i + 1)
                        # saver.save(sess, os.path.join(save_path, 'model_DRNA.ckpt'))  # 保存模型参数
                        # print(f"成功保存模型到{save_path}")
                        break
                print('%d/%d' % (i + 1, epochs), 'epochs...', th)
            outvec = sess.run(output_layer)
            print("训练完成，开始评估测试集...............")
            print(test)
            get_hits(outvec, test)
            choose2save = input("是否保存模型？(y/n)")
            if choose2save == 'y':
                new_model_name = input("请输入新模型名字（model_DRNA.ckpt）：")
                save_path_full=os.path.join(save_path, new_model_name)
                saver.save(sess,save_path_full )  # 保存模型参数
                # # 确保文件被写入磁盘
                # for ext in ['.meta', '.data-00000-of-00001', '.index']:
                #     file_path = save_path_full + ext
                #     if os.path.exists(file_path):
                #         with open(file_path, 'a') as f:
                #             # 打开文件并写入空字符串，这会触发flush
                #             f.write('')
                #             # 在文件关闭前确保数据被写入磁盘
                #             os.fsync(f.fileno())
                print(f"成功保存模型到{save_path}")
            else:
                print("模型未保存")
        else:
            print("本轮标注结束，剩余数据量：", len(ILL))
    # 保存vec到文件
    return outvec, J ,saver