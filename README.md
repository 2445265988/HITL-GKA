# Cross-domain geographic knowledge graph alignment via augmented human-in-the-loop learning 

## Repository Overview
This model is an extension of the previous DRNA-GCNE model.
This repository contains two main components:
1. **Sample Generation based on Clustering Algorithms in Human-Machine Collaboration**
2. **Geographic Knowledge Graph Alignment Guided by Human-in-the-Loop**

## Environment Setup
The runtime dependencies are listed in the `requirements.txt` file. Please install the dependencies using the following command:
```bash
pip install -r requirements.txt
```
## Dataset
- **The dataset is located in the `GeoEA2024` folder.**

## Code Execution
### Sample Generation based on Clustering Algorithms in Human-Machine Collaboration
- **The relevant code is located in `include/SGC_HMC.py`.**

### Geographic Knowledge Graph Alignment Guided by Human-in-the-Loop
- **The main program entry is `main.py`.**  
  The program interacts with users through the command line and provides the following operational options:
  1. **Test Current Model Accuracy**: Load the saved model and evaluate it on the test data.
  2. **Train a New Model**: Train a new model based on user-specified parameters and save the training results.
  3. **Iterative Training**: Further optimize the training based on an existing model.
  4. **Test Confidence Accuracy (Threshold Adjustment)**: Evaluate the model's performance at different confidence levels by adjusting the confidence threshold.
  5. **Labeling**: Apply the model for human-machine interactive data labeling.

## 仓库简介
本模型为上一个研究DRNA-GCNE模型的进一步研究
本仓库包含两个主要内容：
1. **基于聚类算法的人机协同样本生成**
2. **人在回路引导下的地理知识图谱对齐**

## 环境配置
运行环境依赖已列出在 `requirements.txt` 文件中。请使用以下命令安装依赖：
```bash
pip install -r requirements.txt
```

## 数据集
- 数据集位于 GeoEA2024 文件夹中。该数据集包含用于地理知识图谱对齐的训练、测试和验证数据。

## 代码运行
### 基于聚类算法的人机协同样本生成  
- 相关代码位于 include/SGC_HMC.py。
### 人在回路引导下的地理知识图谱   
- 主程序入口为 main.py。
- 程序通过命令行与用户交互，提供以下操作选项：
1. **测试当前模型精度**：加载已保存的模型，对测试数据进行评估。
2. **训练新模型**：根据用户指定的参数训练新模型，并保存训练结果。
3. **迭代训练**：在已有模型的基础上进行进一步优化训练。
4. **测试置信度精度（阈值调整）**：通过调整置信度阈值，评估模型在不同置信度下的性能。
5. **打标**：模型的应用，使用该模型进行人机交互式的数据标注




