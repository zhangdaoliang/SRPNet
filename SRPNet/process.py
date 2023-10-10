from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
import os
import torch
import random
import numpy as np
from torch.backends import cudnn

class StrokeDataset(Dataset):  # 继承Dataset类
    def __init__(self, x,y,device):
        # 把数据和标签拿出来
        self.x_data = torch.tensor(x).float().to(device)
        self.y_data = torch.tensor(y).long().to(device)

        # 数据集的长度
        self.length = len(self.y_data)

    # 下面两个魔术方法比较好写，直接照着这个格式写就行了
    def __getitem__(self, index):  # 参数index必写
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length  # 只需返回数据集的长度即可

def featureselect(data_x):
    data_x['marragenum'] = 1
    data_x.loc[data_x["aMarriage"] == 10, "marragenum"] = 0
    data_x.loc[data_x["aMarriage"] == 23, "marragenum"] = 2
    data_x.loc[data_x["aMarriage"] == 22, "marragenum"] = 2
    data_x['marragestatus'] = 1
    data_x.loc[data_x["aMarriage"] == 10, "marragestatus"] = 0
    data_x.loc[data_x["aMarriage"] == 30, "marragestatus"] = 0
    data_x.loc[data_x["aMarriage"] == 40, "marragestatus"] = 0
    data_x['marrage_other'] = 0
    data_x.loc[data_x["aMarriage"] == 90, "marrage_other"] = 1

    data_x.loc[data_x["aNation"] != 1, "aNation"] = 0

    data_x.loc[data_x["aCategory"] == 50, "aCategory"] = 0
    data_x.loc[data_x["aCategory"] == 60, "aCategory"] = 1
    # 1北京 2山东 3河南 4山西 5陕西 6四川
    data_x['pgdp'] = 0
    data_x.loc[data_x["Province"] == 1, "pgdp"] = 80394
    data_x.loc[data_x["Province"] == 2, "pgdp"] = 46976
    data_x.loc[data_x["Province"] == 3, "pgdp"] = 28716
    data_x.loc[data_x["Province"] == 4, "pgdp"] = 30802
    data_x.loc[data_x["Province"] == 5, "pgdp"] = 33197
    data_x.loc[data_x["Province"] == 6, "pgdp"] = 26147
    data_x['pn'] = 0  # 纬度
    data_x.loc[data_x["Province"] == 1, "pn"] = 39.9
    data_x.loc[data_x["Province"] == 2, "pn"] = 36.4
    data_x.loc[data_x["Province"] == 3, "pn"] = 33.9
    data_x.loc[data_x["Province"] == 4, "pn"] = 37.5
    data_x.loc[data_x["Province"] == 5, "pn"] = 35.6
    data_x.loc[data_x["Province"] == 6, "pn"] = 30.1
    data_x['pe'] = 0  # 经度
    data_x.loc[data_x["Province"] == 1, "pe"] = 116.4
    data_x.loc[data_x["Province"] == 2, "pe"] = 118.5
    data_x.loc[data_x["Province"] == 3, "pe"] = 113.5
    data_x.loc[data_x["Province"] == 4, "pe"] = 112.3
    data_x.loc[data_x["Province"] == 5, "pe"] = 118.3
    data_x.loc[data_x["Province"] == 6, "pe"] = 100.4

    data_x['pp'] = 0  # 降水量
    data_x.loc[data_x["Province"] == 1, "pp"] = 500
    data_x.loc[data_x["Province"] == 2, "pp"] = 710
    data_x.loc[data_x["Province"] == 3, "pp"] = 700
    data_x.loc[data_x["Province"] == 4, "pp"] = 520
    data_x.loc[data_x["Province"] == 5, "pp"] = 600
    data_x.loc[data_x["Province"] == 6, "pp"] = 1100

    data_x['te_max'] = 0  # 最高温
    data_x.loc[data_x["Province"] == 1, "te_max"] = 27.3  # 北京
    data_x.loc[data_x["Province"] == 2, "te_max"] = 27.1  # 山东
    data_x.loc[data_x["Province"] == 3, "te_max"] = 27.1  # 河南
    data_x.loc[data_x["Province"] == 4, "te_max"] = 24  # 4山西
    data_x.loc[data_x["Province"] == 5, "te_max"] = 31.6  # 陕西
    data_x.loc[data_x["Province"] == 6, "te_max"] = 25.4  # 四川

    data_x['te_min'] = 0  # 最低温
    data_x.loc[data_x["Province"] == 1, "te_min"] = -3  # 北京
    data_x.loc[data_x["Province"] == 2, "te_min"] = -0.6  # 山东
    data_x.loc[data_x["Province"] == 3, "te_min"] = 0.5  # 河南
    data_x.loc[data_x["Province"] == 4, "te_min"] = -5  # 4山西
    data_x.loc[data_x["Province"] == 5, "te_min"] = -0.3  # 陕西
    data_x.loc[data_x["Province"] == 6, "te_min"] = 5.6  # 四川

    data_x['pp_max'] = 0  # 最高湿度
    data_x.loc[data_x["Province"] == 1, "pp_max"] = 71  # 北京
    data_x.loc[data_x["Province"] == 2, "pp_max"] = 77  # 山东
    data_x.loc[data_x["Province"] == 3, "pp_max"] = 79  # 河南
    data_x.loc[data_x["Province"] == 4, "pp_max"] = 74  # 4山西
    data_x.loc[data_x["Province"] == 5, "pp_max"] = 82  # 陕西
    data_x.loc[data_x["Province"] == 6, "pp_max"] = 86  # 四川

    data_x['pp_min'] = 0  # 最低湿度
    data_x.loc[data_x["Province"] == 1, "pp_min"] = 42  # 北京
    data_x.loc[data_x["Province"] == 2, "pp_min"] = 45  # 山东
    data_x.loc[data_x["Province"] == 3, "pp_min"] = 59  # 河南
    data_x.loc[data_x["Province"] == 4, "pp_min"] = 45  # 4山西
    data_x.loc[data_x["Province"] == 5, "pp_min"] = 63  # 陕西
    data_x.loc[data_x["Province"] == 6, "pp_min"] = 77  # 四川

    data_x = data_x.drop('Province', axis=1)

    #
    # data_x["aJob"]=data_x["aJob"].apply(str)
    # res = pd.get_dummies(data_x[["aJob"]])
    # data_x= data_x.join(res)
    # data_x =data_x.drop('aJob', axis=1)
    data_x = data_x.drop('aMarriage', axis=1)
    return data_x

def set_seed(seed=3):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    torch.backends.cudnn.deterministic = True
    cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
