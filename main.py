import os

import numpy as np
import psutil
import scipy
import scipy.spatial as spatial
import tensorflow as tf
from include.Config import Config
# 改进的模型
from include.Model_human import training,build,get_neg,funt_training, test_uncertain_acc, only_noting
from include.Test import get_hits
from include.Load import *

import warnings
warnings.filterwarnings("ignore")

'''
Follow the code style of GCN-Align:
https://github.com/1049451037/GCN-Align
'''




seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

def get_float_input(prompt):
    while True:
        try:
            value = float(input(prompt))
            return value
        except ValueError:
            print("输入无效，请输入一个小数。")
def load_and_test_model(output_layer, loss, test, gamma, save_path, modle_name):

    saver = tf.train.Saver()  # 创建 Saver 对象
    with tf.Session() as sess:
        if os.path.exists(save_path):
            saver.restore(sess, os.path.join(save_path, modle_name))  # 恢复模型参数
            print("Model restored from", save_path)
        else:
            print("No saved model found at", save_path)
            return
        # 测试模型
        print("Testing model on test set...")
        outvec = sess.run(output_layer)
        get_hits(outvec, test)

if __name__ == '__main__':
    e = len(set(loadfile(Config.e1, 1)) | set(loadfile(Config.e2, 1)))
    # ILL = loadfile(Config.ill, 2)
    train_data = loadfile(Config.train, 2)
    test_data = loadfile(Config.test, 2)
    val_data = loadfile(Config.val, 2)
    # illL = len(ILL)
    # np.random.shuffle(ILL)
    dict1 = loadfile2dic(Config.e1)
    dict2 = loadfile2dic(Config.e2)
    # 使用字典解包合并字典
    dict1.update(dict2)
    ent_dic = dict1
    train = np.array(train_data)
    print("训练集大小为：", len(train))
    test = np.array(test_data)
    print("测试集大小为：", len(test))
    val = np.array(val_data)
    print("验证集大小为：", len(val))
    KG1 = loadfile(Config.kg1, 3)
    KG2 = loadfile(Config.kg2, 3)
    tf.reset_default_graph()
    output_layer, loss = build(
        Config.dim, Config.act_func, Config.alpha, Config.beta, Config.gamma, Config.k, Config.language[0:2], e,
        train, KG1 + KG2)

    while(1):
        mode = input("请选择要进行的操作：1.测试当前模型精度 2.训练新模型 3.小样本场景训练并测试 4.测试置信度精度（阈值调整）5.打标：")

        if mode == '1':
            # 调用函数进行测试
            test_file = input("请输入需要加载的测试文件名称data/j3_en/entity_pairs：")
            test_data = loadfile(test_file, 2)
            save_path = input("请输入模型保存目录./model/model_DRNA_1021：")
            modle_name = input("请输入需要测试的模型文件名称：")
            load_and_test_model(output_layer, loss, test_data, Config.gamma, save_path,modle_name)
        elif mode == '2':
            save_path = input("请输入模型保存目录./model/model_DRNA_1021：")
            restore = False  # 设置为 True 来恢复模型
            vec, J = training(output_layer, loss, 0.001,
                              Config.epochs, train, e, Config.k, test, val, Config.gamma,save_path,restore)
            get_hits(vec, test)
        elif mode == '3':
            # try:
            model_dir = input("请输入模型目录./model/model_DRNA_1021：")
            modle_name = input("请输入需要加载的模型文件名称model_DRNA.ckpt：")
            X_pool_file = input("请输入需要加载的X_pool文件名称data/j3_en/entity_pairs：")
            X_pool = loadfile(X_pool_file, 2)
            # modle_dir = os.path.join(model_dir, modle_name)
            if os.path.exists(model_dir):
                restore = True  # 设置为 True 来恢复模型
                vec, J, saver = funt_training(X_pool_file,output_layer, loss, 0.001,
                                         Config.epochs, X_pool,dict1, e, Config.k, test, val, Config.gamma, model_dir,modle_name, restore)
                print("success in retrain models")
            else:
                print("模型文件不存在")
            # except ValueError:
            #     print("输入有误，请重试")
        elif mode == '4':
            try:
                # model_dir = input("请输入模型目录./model/model_DRNA_1021：")
                # modle_name = input("请输入需要加载的模型文件名称model_DRNA.ckpt：")
                # X_pool_file = input("请输入需要加载的X_pool文件名称data/j3_en/entity_pairs：")
                model_dir = './model/model_DRNA_1029'
                modle_name = 'model_DRNA.ckpt'
                X_pool_file = 'data/j3_en/entity_pairs_200'
                X_pool = loadfile(X_pool_file, 2)
                if os.path.exists(model_dir):
                    restore = True  # 设置为 True 来恢复模型
                    uncertainty_threshold = get_float_input("请输入置信度阈值：")
                    weight1 = get_float_input("请输入权重1：")
                    weight2 = get_float_input("请输入权重2：")
                    result = test_uncertain_acc(uncertainty_threshold,weight1,weight2,output_layer, loss, 0.001,
                                             Config.epochs, X_pool,dict1, e, Config.k, test, val, Config.gamma, model_dir,modle_name, restore)
            except ValueError:
                print("输入有误，请重试")
        elif mode == '5':
            # try:
            model_dir = input("请输入模型目录：")
            modle_name = input("请输入需要加载的模型文件名称：")
            X_pool_file = input("请输入需要加载的X_pool文件名称：")
            X_pool = loadfile(X_pool_file, 2)

            # modle_dir = os.path.join(model_dir, modle_name)
            if os.path.exists(model_dir):
                restore = True  # 设置为 True 来恢复模型
                vec, J, saver = only_noting(X_pool_file, output_layer, loss, 0.001,
                                              Config.epochs, X_pool, dict1, e, Config.k, test, val, Config.gamma,
                                              model_dir, modle_name, restore)
                print("success in retrain models")
            else:
                print("模型文件不存在")


