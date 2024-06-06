import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from scipy.stats import spearmanr



# 定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_length=237):
        self.data = dataframe

        # 筛选 Protein_Length 为 max_length 的数据
        self.data = self.data[self.data['Protein_Length'] == max_length]

        # 提取输入和输出
        self.mutations = self.data['Mutation']
        self.log_fluorescence = self.data['Log_Fluorescence']

        # 标签编码
        self.le = LabelEncoder()
        self.le.fit([char for mutation in self.mutations for char in mutation.replace('-', '')])
        self.mutations_encoded = [self.encode_mutation(mutation) for mutation in self.mutations]
        self.max_length = max_length
        self.mutations_padded = [self.pad_sequence(mutation, self.max_length) for mutation in self.mutations_encoded]

    def encode_mutation(self, mutation):
        return self.le.transform(list(mutation.replace('-', '')))

    def pad_sequence(self, sequence, max_length):
        return list(sequence) + [0] * (max_length - len(sequence))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (torch.tensor(self.mutations_padded[idx], dtype=torch.long),
                torch.tensor(self.log_fluorescence.iloc[idx], dtype=torch.float32),
                self.mutations.iloc[idx])  # 添加原始突变信息

# 数据加载
test_df = pd.read_csv(r'../data/compare/flu/flu_test_compared.csv')

max_length = 237  # 固定序列长度为 237

# 创建数据集和数据加载器
test_dataset = ProteinDataset(test_df, max_length=max_length)
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义Xception模块
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(SeparableConv1d, self).__init__()
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(XceptionBlock, self).__init__()
        self.separable_conv1 = SeparableConv1d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.separable_conv2 = SeparableConv1d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.separable_conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.separable_conv2(x)
        x = self.bn2(x)
        x += shortcut
        x = self.relu2(x)
        return x

# 定义Xception网络
class Xception1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_length):
        super(Xception1D, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.block1 = XceptionBlock(hidden_dim, hidden_dim)
        self.block2 = XceptionBlock(hidden_dim, hidden_dim)
        self.block3 = XceptionBlock(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * max_length, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, max_length)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 初始化模型、损失函数和优化器
input_dim = len(test_dataset.le.classes_) + 1  # 包含填充的0
hidden_dim = 256
output_dim = 1

model = Xception1D(input_dim, hidden_dim, output_dim, max_length)
model.to('cuda')  # 将模型移动到 GPU



# 加载Spearman指标最高的模型权重
model.load_state_dict(torch.load('../output/Xception/wt_best_xception1d_model.pth'))
model.eval()
# 测试模型
criterion = nn.MSELoss()
total_test_loss = 0
all_mutations = []
all_outputs = []
all_targets = []
with torch.no_grad():
    for inputs, targets, mutations in tqdm(test_loader, desc='Testing'):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        total_test_loss += loss.item()
        all_mutations.extend(mutations)
        all_outputs.extend(outputs.squeeze().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
avg_test_loss = total_test_loss / len(test_loader)
spearman_corr, _ = spearmanr(all_outputs, all_targets)
print(f'Test Loss: {avg_test_loss:.4f}, Spearman: {spearman_corr:.4f}')

# 保存预测值与实际值到CSV文件
results_df = pd.DataFrame({
    'Mutation': all_mutations,
    'Actual': all_targets,
    'Predicted': all_outputs
})
results_df.to_csv('flu_Xception_pre_load.csv', index=False)
print("预测值与实际值已保存为 flu_Xception_pre_load.csv")
