# Cross-domain geographic knowledge graph alignment via augmented human-in-the-loop learning 

## Repository Introduction
This model is a further study based on the previous DRNA-GCNE model.

This repository contains two main components:
1. **Human-Machine Collaborative Sample Generation Based on Clustering Algorithm**
2. **Geographic Knowledge Graph Alignment Guided by Human-in-the-Loop**

## Environment Setup
The dependencies required for running this project are listed in the `requirements.txt` file. Install them using the following command:
```bash
pip install -r requirements.txt
```

## Dataset
- The dataset is located in the `GeoEA2024` folder. It includes training, testing, and validation data for geographic knowledge graph alignment.

## Code Execution
### Human-Machine Collaborative Sample Generation Based on Clustering Algorithm  
- The relevant code is located in `include/SGC_HMC.py`.

### Geographic Knowledge Graph Alignment Guided by Human-in-the-Loop  
- The main entry point is `main.py`.
- The program interacts with users via the command line and provides the following options:
  1. **Test current model accuracy**: Load a pre-trained model and evaluate it on the test dataset.
  2. **Train a new model**: Train a new model based on user-specified parameters and save the training results.
  3. **Iterative training**: Further optimize the model based on an existing one.
  4. **Uncertainty sensitivity testing (threshold adjustment)**: Adjust the uncertainty threshold to evaluate the model's performance under different uncertainty conditions.
  5. **Annotation**: Apply the model for interactive human-in-the-loop data annotation.

## Experimental Process  

### Analysis of Human-Machine Collaborative Sample Generation Effectiveness  
1. Run `include/SGC_HMC.py` to generate potential alignment data of 8 different scales (stored in `GeoEA2024_small`) and manually correct potential errors returned in the console.  
2. Use the LLaMa2 large model to automatically evaluate the generated potential entity pairs.  
3. Record the number of erroneous annotations from both methods and compare their accuracy.  

### Optimization Experiment for Few-Shot Human-in-the-Loop Alignment  
1. Run `main.py` using the `train` and `test` datasets in `GeoEA2024_small` for model training and testing.  
2. Train and test multiple baseline models using the same dataset. The baseline model codes can be accessed from the following links:  
   - **RDGCN** (Wu et al., 2019b) - [GitHub](https://github.com/StephanieWyt/RDGCN)  
   - **Dual-AMN** (Mao et al., 2021) - [GitHub](https://github.com/MaoXinn/Dual-AMN)  
   - **AliNet** (Sun et al., 2020b) - [GitHub](https://github.com/nju-websoft/AliNet)  
   - **BootEA** (Sun et al., 2018) - [GitHub](https://github.com/nju-websoft/BootEA)  
   - **Aligne** (Sun et al., 2020a) - [GitHub](https://github.com/nju-websoft/OpenEA)  
   - **SEA** (Pei et al., 2019) - [GitHub](https://github.com/nju-websoft/OpenEA)  
   - **GCN-Align** (Wang et al., 2018) - [GitHub](https://github.com/1049451037/GCN-Align)  
   - **MTransE** (Chen et al., 2017) - [GitHub](https://github.com/muhaochen/MTransE)  
   - **PEEA** (Tang et al., 2023) - [GitHub](https://github.com/OceanTangWei/PEEA)  
   - **RNM** (Zhu et al., 2022) - [GitHub](https://github.com/Peter7Yao/RNM)  

### Parameter Sensitivity Experiment for Uncertainty Sampling  
1. Run `main.py` and perform **uncertainty sensitivity testing (threshold adjustment)** operations.  
2. Set the **uncertainty threshold testing range** to **0.6 - 0.8**.  
3. Set the **weight testing range** to **0.4 - 0.8**, and analyze the impact of different parameter configurations on model performance.  

### Experiment on the Impact of Annotation Quantity on Model Performance  
1. Run `main.py` to conduct the experiment.  
2. To compare with a fully annotated model, we randomly select **8000 unannotated data samples**, with **6000 used for training and 2000 for testing**.  
3. Conduct experiments using two training approaches for comparison:  
   - **One-time training**: Train the model directly using manually annotated alignment data, execute the **Train a new model** operation, and record the test results.  
   - **Iterative training**: Input unannotated data into the model for **active learning-based iterative training**, execute the **Iterative training** operation, manually correct potential errors returned in the console, and finally record the test results after **five training rounds**.  



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
4. **不确定度敏感度测试（阈值调整）**：通过调整不确定度阈值，评估模型在不同不确定度下的性能。
5. **打标**：模型的应用，使用该模型进行人机交互式的数据标注。

## 实验过程  

### 人机协同样本生成效果分析  
1. 运行 `include/SGC_HMC.py` 生成 8 个量级的潜在对齐数据（存储于 `GeoEA2024_small`），并对控制台返回的潜在错误进行人工修正。  
2. 使用 LLaMa2 大模型对生成的潜在实体对进行自动判断。  
3. 记录两种方法的错误标记数量，并计算正确率进行对比分析。  

### 面向小样本的人在回路对齐结果优化实验  
1. 运行 `main.py`，使用 `GeoEA2024_small` 数据集中的 `train` 和 `test` 数据进行模型训练和测试。  
2. 使用相同的数据训练并测试多个对比模型，对比模型代码可从以下链接获取：  
   - **RDGCN** (Wu et al., 2019b) - [GitHub](https://github.com/StephanieWyt/RDGCN)  
   - **Dual-AMN** (Mao et al., 2021) - [GitHub](https://github.com/MaoXinn/Dual-AMN)  
   - **AliNet** (Sun et al., 2020b) - [GitHub](https://github.com/nju-websoft/AliNet)  
   - **BootEA** (Sun et al., 2018) - [GitHub](https://github.com/nju-websoft/BootEA)  
   - **Aligne** (Sun et al., 2020a) - [GitHub](https://github.com/nju-websoft/OpenEA)  
   - **SEA** (Pei et al., 2019) - [GitHub](https://github.com/nju-websoft/OpenEA)  
   - **GCN-Align** (Wang et al., 2018) - [GitHub](https://github.com/1049451037/GCN-Align)  
   - **MTransE** (Chen et al., 2017) - [GitHub](https://github.com/muhaochen/MTransE)  
   - **PEEA** (Tang et al., 2023) - [GitHub](https://github.com/OceanTangWei/PEEA)  
   - **RNM** (Zhu et al., 2022) - [GitHub](https://github.com/Peter7Yao/RNM)  

### 不确定性抽样方法参数敏感性实验  
1. 运行 `main.py` 并执行 **不确定度敏感度测试（阈值调整）** 相关操作。  
2. 设定 **不确定度阈值测试范围** 为 **0.6 - 0.8**。  
3. 设定 **权重测试范围** 为 **0.4 - 0.8**，并分析不同参数配置对模型性能的影响。  

### 标注量对模型性能提升实验  
1. 运行 `main.py` 进行实验。  
2. 为了与全标注模型进行对比，我们随机抽取 **8000 条未标注数据**，其中 **6000 条用于训练，2000 条用于测试**。  
3. 采用两种训练方式进行实验对比：  
   - **一次性训练**：使用人工标注后的对齐数据直接训练模型，执行 **训练新模型** 操作，并记录测试结果。  
   - **迭代训练**：将未标注数据输入模型进行 **主动学习式迭代训练**，执行 **迭代训练** 操作，并对控制台返回的潜在错误进行人工修正，最终记录 **五轮训练后** 的测试结果。  




