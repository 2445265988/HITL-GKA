import time

from scipy import spatial
from sklearn import svm
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestNeighbors
import numpy as np
from Load import *
from tqdm import tqdm

# 使用余弦距离
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

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
        neg.append(rank[1:k+1])
    neg = np.array(neg)
    neg = neg.reshape((t * k,))
    return neg

def dis_normalization_function(dist):
    s = -np.log(0.5) / 4.5
    dist = np.exp(-s * dist)
    return dist

def normalize_distance_matrix(distance_matrix):
    # 将归一化函数应用于距离矩阵的每个元素
    normalized_matrix = dis_normalization_function(distance_matrix)
    return normalized_matrix

def calculate_entropy(prob_dist):
    # 计算信息熵
    entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))
    return entropy

def calculate_entropy_matrix(normalized_distance_matrix):
    # 将信息熵计算函数应用于归一化距离矩阵的每个元素
    entropy_matrix = np.zeros_like(normalized_distance_matrix)
    for i in range(normalized_distance_matrix.shape[0]):
        distance = normalized_distance_matrix[i]
        prob_dist = np.array([1 - distance, distance])
        entropy_matrix[i] = calculate_entropy(prob_dist)
    return entropy_matrix

def find_top_similar_pairs_2(entity_vectors_list1, entity_vectors_list2,dict1,dict2, top_k=10):
    # 将列表转换为 NumPy 数组，以便进行矩阵运算
    entity_vectors_array1 = np.array(entity_vectors_list1)
    entity_vectors_array2 = np.array(entity_vectors_list2)

    # 计算两个数组中所有向量对之间的余弦距离
    distance_matrix = spatial.distance.cdist(entity_vectors_array1, entity_vectors_array2, metric='cityblock')
    # 越接近0越不匹配
    normalized_sim_matrix = normalize_distance_matrix(distance_matrix)
    # 计算信息熵矩阵
    entropy_matrix = calculate_entropy_matrix(normalized_sim_matrix)
    # 获取最相似的top_k个索引对
    # 找到信息熵和归一化距离都在阈值范围内的 k 个实体对
    # 信息熵为0表示最确定
    threshold_entropy_min = 0.2  # 信息熵阈值
    threshold_entropy_max = 0.8  # 信息熵阈值
    threshold_sim_max = 0.6  # 归一化距离阈值
    threshold_sim_min = 0.4  # 归一化距离阈值
    top_similar_pairs = []
    top_unsimilar_pairs = []
    for i in range(len(entity_vectors_list1)):
        for j in range(len(entity_vectors_list2)):
            if entropy_matrix[i, j] >= threshold_entropy_min and  entropy_matrix[i, j] <= threshold_entropy_max and normalized_sim_matrix[i, j] >= threshold_sim_max:
                print( dict1[i], dict2[j], "置信度:", entropy_matrix[i, j], "相似度:", normalized_sim_matrix[i, j])
                top_similar_pairs.append((i, j))
                if len(top_similar_pairs) == top_k:
                    break
            if entropy_matrix[i, j] >= threshold_entropy_min and  entropy_matrix[i, j] <= threshold_entropy_max and normalized_sim_matrix[i, j] <= threshold_sim_min:
                print( dict1[i], dict2[j], "置信度:", entropy_matrix[i, j], "相似度:", normalized_sim_matrix[i, j])
                top_unsimilar_pairs.append((i, j))
                if len(top_unsimilar_pairs) == top_k:
                    break

    return top_similar_pairs,top_unsimilar_pairs

def find_similar_entities_cos(entity_list1, entity_list2, dict1, dict2, k, threshold, m):
    """
    找到两个实体列表中相似度最高的k个实体对
    :param entity_list1: 第一个实体列表
    :param entity_list2: 第二个实体列表
    :param dict1: 第一个实体列表对应的字典
    :param dict2: 第二个实体列表对应的字典
    :param k: 需要找到的相似实体对的数量
    :param threshold: 相似度阈值
    :param m: 排除自身和列表1中的实体，只考虑列表2中的实体
    :return: None
    """
    # 将两个列表合并为一个numpy数组，以便使用NearestNeighbors
    results = set()
    results_dissimilar = set()
    all_entities = np.vstack((entity_list1, entity_list2))

    # 初始化NearestNeighbors模型，使用余弦距离
    neigh = NearestNeighbors(n_neighbors=min(k + 1, len(all_entities)), metric='cosine')

    # 训练模型
    neigh.fit(all_entities)

    # 对于entity_list1中的每个实体，找到相似度最高的k个实体对
    for i, entity in enumerate(entity_list1):
        # 计算距离
        distances, indices = neigh.kneighbors([entity], n_neighbors=k + 1)
        # 排除自身，因为kneighbors包括自身
        # print("distances_old:", distances)
        # print("indices_old:", indices)
        distances = distances[0][np.array(indices[0]) >= m]
        indices = indices[0][np.array(indices[0]) >= m]
        # print("distances:",distances)
        # print("indices:",indices)
        # 计算每个实体对的相似度概率分布
        prob_dist = 1 - distances  # 将距离转换为相似度
        # print("prob_dist:",prob_dist)
        entropy = calculate_entropy_matrix(prob_dist)  # 计算信息熵
        # print("entropy:",entropy)
        # 根据信息熵和相似度阈值筛选实体对
        # 根据信息熵和距离阈值筛选实体对
        threshold_entropy_min = 0.2  # 信息熵阈值
        threshold_entropy_max = 0.8  # 信息熵阈值
        threshold_sim_max = 0.2  # 归一化距离阈值
        threshold_sim_min = 0.1  # 归一化距离阈值

        for j, (distance, index) in enumerate(zip(distances, indices)):
            if entropy[j] < threshold_entropy_max and entropy[j] > threshold_entropy_min:
                # print(dict1[i], dict2[int(index)], "置信度:", entropy[j], "距离:", distance)
                if distance < threshold_sim_min and len(results) <= k :
                    # print("找到相似实体对")
                    # print(dict1[i], dict2[int(index)], "置信度:", entropy[j], "距离:", distance)
                    results.add((i, index))
                if distance > threshold_sim_max and len(results_dissimilar)<=k:
                    # print("找到不相似实体对")
                    # print(dict1[i], dict2[int(index)], "置信度:", entropy[j], "距离:", distance)
                    results_dissimilar.add((i, index))
                if len(results) >= k and len(results_dissimilar) >= k:
                    break
        if len(results) >= k and len(results_dissimilar) >= k:
            break



        # # 过滤相似度低于阈值的结果
        # filtered_indices = indices[distances < threshold]
        # filtered_distances = distances[distances < threshold]
        # # print("filtered_indices:",filtered_indices)
        # # print("filtered_distances:",filtered_distances)
        # if len(filtered_indices) > 0:
        #     # print(f"{dict1[i]},{dict2[int(filtered_indices[0])]}")
        #     results.add((i, int(filtered_indices[0])))
        #     # 找到不相似的实体对
        # dissimilar_indices = indices[distances > 0.5]
        # if len(dissimilar_indices) > 0:
        #     results_dissimilar.add((i, int(dissimilar_indices[0])))
    return results,results_dissimilar

def find_similar_entities(entity_list1, entity_list2,dict1, dict2, k, threshold,m):
    """
    使用欧式距离
    找到两个实体列表中相似度最高的k个实体对
    找到两个实体列表中相似度最高的k个实体对
    :param entity_list1: 第一个实体列表
    :param entity_list2: 第二个实体列表
    :param k: 需要找到的相似实体对的数量
    :return: None
    """
    # 将两个列表合并为一个numpy数组，以便使用NearestNeighbors
    results = set()
    all_entities = np.vstack((entity_list1, entity_list2))

    # 初始化NearestNeighbors模型
    neigh = NearestNeighbors(n_neighbors=min(k + 1, len(all_entities)))  # n_neighbors包括自身，所以设置为k+1

    # 训练模型
    neigh.fit(all_entities)

    # 对于entity_list1中的每个实体，找到相似度最高的k个实体对
    for i, entity in tqdm(enumerate(entity_list1),total=len(entity_list1)):
        # 计算距离
        distances, indices = neigh.kneighbors([entity], n_neighbors=k + 1)
        # 排除自身，因为kneighbors包括自身
        distances = distances[0][np.array(indices[0]) >= m]
        indices = indices[0][np.array(indices[0]) >= m]
        # 过滤相似度低于阈值的结果
        filtered_indices = indices[distances < threshold]
        filtered_distances = distances[distances < threshold]
        # print("filtered_indices:",filtered_indices)
        # print("filtered_distances:",filtered_distances)
        if len(filtered_indices) > 0:
            # print(f"{dict1[i]},{dict2[int(filtered_indices[0])]}")
            results.add((i, int(filtered_indices[0])))
        # print("results:",results)
    return results



def cut_vec(vecfile,k):
    # 加载.npy文件
    all_vectors = np.loadtxt(vecfile)
    # 分割数组
    kg1_vec = all_vectors[:k]
    kg2_vec = all_vectors[k:]
    return kg1_vec,kg2_vec




if __name__ == '__main__':
    language = 'j3_en'  # zh_en | ja_en | fr_en
    e1 = '../data/' + language + '/ent_ids_1'
    e2 = '../data/' + language + '/ent_ids_2'
    dict1 = loadfile2dic(e1)
    dict2 = loadfile2dic(e2)
    # print("字典1大小：", len(dict1))
    # print("字典2大小：", len(dict2))
    # print("字典1：", dict1)
    # print("字典2：", dict2)
    vectors_kg1, vectors_kg2 = cut_vec('../data/j3_en/out_vec.txt', len(dict1))
    outvec = np.loadtxt('../data/j3_en/out_vec.txt')
    # print("向量1大小：", vectors_kg1.shape)
    # print("向量2大小：", vectors_kg2.shape)
    start_time = time.time()  # 记录开始时间
    # results = find_similar_entities(vectors_kg1, vectors_kg2, dict1, dict2, 20, 0.3,len(dict1))
    results,results_dissim = find_similar_entities_cos(vectors_kg1, vectors_kg2, dict1, dict2, 20, 0.2, len(dict1))
    results = list(results)
    print("相似实体对:",results)
    # results,results_dissim = find_top_similar_pairs_2(vectors_kg1, vectors_kg2, dict1, dict2, 20)
    # print("进行负样本增强---------------")
    # head_entities = [result[0] for result in results]
    # tail_entities = [result[1] for result in results]
    # need_neg_num = len(results)
    # # neg_left = pos_left[:need_neg_num]
    # # neg_right = get_neg(neg_left, out, 1)
    # L = np.ones((need_neg_num, 1)) * (np.array(head_entities).reshape((need_neg_num, 1)))
    # new_neg_left = L.reshape((need_neg_num * 1,))
    # new_neg_right = get_neg(head_entities, outvec, 1)
    # # neg_left = new_neg_left
    # # neg_right = new_neg_right
    # L = np.ones((need_neg_num, 1)) * (np.array(tail_entities).reshape((need_neg_num, 1)))
    # new_neg_right2 = L.reshape((need_neg_num * 1,))
    # new_neg_left2 = get_neg(tail_entities, outvec, 1)
    # # 创建负样本对，保持与 results 一致的格式
    # results_dissim = [(int(new_neg_left[i]), new_neg_right[i]) for i in range(need_neg_num)] + \
    #                  [(new_neg_left2[i], int(new_neg_right2[i])) for i in range(need_neg_num)]
    # # 打印结果
    # print("负样本对:", results_dissim)
    # results_dissim_l = get_neg
    end_time = time.time()  # 记录结束时间
    print("Elapsed time: {:.2f} seconds".format(end_time - start_time))
    # with open('similar_entities_name_cos.txt', 'w') as file:
    #     for result in results:
    #         file.write(f"{dict1[result[0]]},{dict2[result[1]]}\n")
    # with open('similar_entities_id_cos.txt', 'w') as file:
    #     for result in results:
    #         file.write(f"{result[0]},{result[1]}\n")

    results_dissim = list(results_dissim)
    print("不相似实体对:",results_dissim)
#   进行K折交叉验证找出可能错误的项
    y = ([1] * len(results)) + ([0] * len(results_dissim))
    print("y:",y)
    # x_correct = ([vectors_kg1[results[i][0]], vectors_kg2[results[i][1]-len(vectors_kg1)]] for i in range(len(results)))
    # x_dissimilar = ([vectors_kg1[results_dissim[i][0]], vectors_kg2[results_dissim[i][1]-len(vectors_kg1)]] for i in range(len(results_dissim)))
    # x = list(x_correct) + list(x_dissimilar)
    # x = np.array([np.concatenate([vectors_kg1[results[i][0]], vectors_kg2[results[i][1]-len(vectors_kg1)]]) for i in range(len(results))]+
    #               [np.concatenate([vectors_kg1[results_dissim[i][0]], vectors_kg2[results_dissim[i][1]-len(vectors_kg1)]]) for i in range(len(results_dissim))])
    # 计算相似实体对的向量差
    similar_diffs = np.array([
        np.subtract(vectors_kg1[results[i][0]], vectors_kg2[results[i][1] - len(vectors_kg1)])
        for i in range(len(results))
    ])

    # 计算不相似实体对的向量差
    dissimilar_diffs = np.array([
        np.subtract(vectors_kg1[results_dissim[i][0]], vectors_kg2[results_dissim[i][1] - len(vectors_kg1)])
        for i in range(len(results_dissim))
    ])

    # 将相似和不相似的向量差合并到一个数组中
    x = np.concatenate((similar_diffs, dissimilar_diffs))
    # x = np.array([np.subtract([vectors_kg1[results[i][0]], vectors_kg2[results[i][1]-len(vectors_kg1)]]) for i in range(len(results))]+
    #               [np.subtract([vectors_kg1[results_dissim[i][0]], vectors_kg2[results_dissim[i][1]-len(vectors_kg1)]]) for i in range(len(results_dissim))])

    result_all = list(results + results_dissim)
    y = np.array(y)
    print("result_all:",result_all)
    print("x:",x)
#     创建一个SVM模型
    svm_model= svm.SVC(kernel='linear', probability=True)
    kf= KFold(n_splits=5, shuffle=True, random_state=42)
    incorrect_indices = []
    recorrect_indices = []
    for fold ,(train_idx,val_idx) in enumerate(kf.split(x)):
        print(f"Fold {fold+1}---------------------")
        x_train,x_val = x[train_idx],x[val_idx]
        y_train,y_val = y[train_idx],y[val_idx]
        print("train_idx:",train_idx)
        print("val_idx:",val_idx)
        print("x_train:",x_train)
        print("x_val:",x_val)
        # 训练模型
        svm_model.fit(x_train, y_train)
        # 预测验证集
        preds = svm_model.predict(x_val)
    #    获取错误标注的实体对
        incorrect_indices = np.where(preds != y_val)[0]
        print('incorrect_indices:',incorrect_indices)
        if len(incorrect_indices) > 0:
            print(f"Found {len(incorrect_indices)} incorrect indices in fold {fold+1}")
            for idx in incorrect_indices:
                entity1_idx = result_all[idx][0]
                entity2_idx = result_all[idx][1]
                print(f"Incorrect : {entity1_idx,entity2_idx}, Predicted: {preds[idx]}, True: {y_val[idx]}")
                print(f"Incorrect_pairs: {entity1_idx,dict1[entity1_idx],dict2[entity2_idx]}, Predicted: {preds[idx]}, True: {y_val[idx]}")
                hunman_label = input("是否匹配? (y/n): ")
                if hunman_label == 'n':
                    recorrect_indices.append(val_idx[idx])
    # 遍历 recorrect_indices 中的每个索引
    for idx in recorrect_indices:
        if idx < len(results):  # 如果索引小于 results_correct 的长度
            # 将 results_correct 中的对应实体对移动到 results_dissim
            results_dissim.append(results.pop(idx))
        # else:  # 如果索引大于或等于 results_correct 的长度
        #     # 将 results_dissim 中的对应实体对移动到 results_correct
        #     # 需要调整索引以适应 results_dissim 的实际长度
        #     adjusted_idx = idx - len(results)
        #     results.append(results_dissim.pop(adjusted_idx))

    # 打印更新后的结果
    print("Updated results_correct:", results)
    print("Updated results_dissim:", results_dissim)
    with open('similar_entities_name_cos.txt', 'w') as file:
        for result in results:
            file.write(f"{dict1[result[0]]},{dict2[result[1]]}\n")
    with open('similar_entities_id_cos.txt', 'w') as file:
        for result in results:
            file.write(f"{result[0]},{result[1]}\n")
    with open('unsimilar_entities_id_cos.txt', 'w') as file:
        for result in results_dissim:
            file.write(f"{result[0]},{result[1]}\n")
    with open('unsimilar_entities_name_cos.txt', 'w') as file:
        for result in results_dissim:
            file.write(f"{dict1[result[0]]},{dict2[result[1]]}\n")
    print("pos_letf:")
    first_elements = [result[0] for result in results]
    print(first_elements)



