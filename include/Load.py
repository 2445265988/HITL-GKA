import numpy as np


# load a file and return a list of tuple containing $num integers in each line
def loadfile(fn, num=1):
    print('loading a file...' + fn)
    ret = []
    with open(fn, encoding='utf-8') as f:
        for line in f:
            # th = line[:-1].split('\t')
            th = line[:-1].split(',')
            x = []
            for i in range(num):
                # print(th[i])
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def loadfile2dic(fn):
    print('loading a file...' + fn)
    # 初始化一个空字典来存储结果
    entity_dict = {}

    # 打开文件并读取内容
    with open(fn, 'r', encoding='utf-8') as file:
        for line in file:
            # 去除每行的首尾空白字符（包括换行符）
            line = line.strip()
            # 分割行，假设行的格式是 "编号 URL"
            if line:  # 确保行不为空
                # 分割编号和URL，假设它们之间用制表符或空格分隔

                parts = line.split(',')
                if len(parts) == 2:
                    # 将编号和URL添加到字典中
                    entity_dict[int(parts[0])] = parts[1]
    return entity_dict

# dic = loadfile("../data/zh_en/ref_ent_ids", 2)
# for i in dic:
#     print(i)