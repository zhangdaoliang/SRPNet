import torch                        # 导入 PyTorch 库
from torch import nn
import torch.nn.functional as F


# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TransformerModel, self).__init__()
        # 定义 Transformer 编码器，并指定输入维数和头数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        # 定义全连接层，将 Transformer 编码器的输出映射到分类空间
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        # 在序列的第2个维度（也就是时间步或帧）上添加一维以适应 Transformer 的输入格式
        x = x.unsqueeze(1)
        # 将输入数据流经 Transformer 编码器进行特征提取
        x = self.encoder(x)
        # 通过压缩第2个维度将编码器的输出恢复到原来的形状
        x = x.squeeze(1)
        # 将编码器的输出传入全连接层，获得最终的输出结果
        x = self.fc(x)
        return x

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self,input_size, num_classes,kernal_num=32,kernel_size=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, kernal_num, kernel_size=3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((input_size-kernel_size+1)*kernal_num, num_classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)  # 添加一个通道维度
        # x = x.permute(0, 2, 1)  # 转换维度顺序，将通道维度放在最后
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        # x = self.relu(x)
        # x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self,in_size,hidden_size=16):
        super(Attention, self).__init__()
        self.project=nn.Linear(in_size, 1, bias=False)

    def forward(self,z):
        w = self.project(z)
        beta = torch.softmax(w,dim=1)
        return (beta*z).sum(1),beta
# 定义 Transformer 模型
class SRPNetModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(SRPNetModel, self).__init__()
        # 定义 Transformer 编码器，并指定输入维数和头数
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=1)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)
        # 定义全连接层，将 Transformer 编码器的输出映射到分类空间
        self.fc = nn.Linear(input_size,20)

        self.layer = nn.Sequential(
            nn.Linear(input_size, 100),nn.BatchNorm1d(100),nn.ReLU(True),
            nn.Linear(100, 50), nn.BatchNorm1d(50), nn.ReLU(True),
            nn.Linear(50, 20), nn.ReLU(True),
        )
        self.attention = Attention(20)
        self.fnn = nn.Linear(20, num_classes)


    def forward(self, x):
        # 在序列的第2个维度（也就是时间步或帧）上添加一维以适应 Transformer 的输入格式
        t = x.unsqueeze(1)
        # 将输入数据流经 Transformer 编码器进行特征提取
        t = self.encoder(t)
        # 通过压缩第2个维度将编码器的输出恢复到原来的形状
        t = t.squeeze(1)
        # 将编码器的输出传入全连接层，获得最终的输出结果
        t = F.relu(self.fc(t))
        l=self.layer(x)
        tl_emb = torch.stack([t, l], dim=1)
        tl, _ = self.attention(tl_emb)

        return tl


# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1) # 添加时间步维度
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 只使用最后一个时间步的输出
        return out