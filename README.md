# SRPNet: Stroke Risk Prediction Model Based on Multi-layer Feature Selection and Deep Fusion Network

We propose a novel prediction model based on multi-layer feature selection and deep fusion network (SRPNet) to alleviate the above problems. First, the proposed multi-layer feature selection method is used to screen comprehensive features related to stroke risk, enabling accurate identification of significant risk factors while eliminating redundant information. Subsequently, we utilize the deep fusion network integrating Transformer and fully connected network (FCN) to establish a risk prediction model for stroke patients. Finally, we evaluate the performance of the SRPNet using screening data from the China Stroke Data Center (CSDC), and further validate its effectiveness with census data on stroke provided by Affiliated Hospital of Jining Medical University. Numerous experimental results demonstrate that the proposed SRPNet model outperforms a series of baselines, and can accurately predict stroke risk levels of patients based on risk factors.

Framework

![image](https://github.com/zhangdaoliang/SRPNet/blob/main/SRPNet.png)


The code is licensed under the MIT license. 

# 1. Requirements 

## 1.1 Operating systems:

The code in python has been tested on both Linux (Ubuntu 20.04.6 LTS) and windows 10 system.

## 1.2 Required packages in python: 

torch~=1.13.0+cu116

numpy~=1.25.0

pandas~=2.0.2

matplotlib~=3.7.1

sklearn~=0.0.post5

scikit-learn~=1.2.2

scipy~=1.11.0

xgboost~=1.7.6

joblib~=1.2.0

stlearn==0.3.2


# 2. Run SRPNet


## 2.1 Overall

The SRPNet model is implemented in ***SRPNet***.The program is divided into three parts: Multi-layer feature selection 1, Multi-layer feature selection 2, Deep fusion network.


## 2.2 Usage

 If you want to run Deep fusion network on in-house dataset, run 

 'python run_DFN.py --dataset inhouse'

We also provide examples of feature selection Multi-layer feature selection 1 & 2, run

'python run_mutilslect1.py'

'python run_mutilslect2.py'

All results are saved in the results folder. We provide our results in the folder ***result*** for taking further analysis. 



# 3. Download all datasets used in SRPNet:

CSDC: https://service.chinasdc.cn/healthcare/

in-house dataset: We have provided in this project.

