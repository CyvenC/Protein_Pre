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

# 超参数
MAX_LENGTH = 237  # 固定序列长度
BATCH_SIZE = 64  # 批次大小
INPUT_DIM = None  # 输入的特征维度，稍后在读取数据集时确定
HIDDEN_DIM = 256  # 隐藏层维度
OUTPUT_DIM = 1  # 输出维度
LEARNING_RATE = 0.001  # 学习率
WEIGHT_DECAY = 1e-5  # 权重衰减
NUM_EPOCHS = 150  # 训练轮数
KERNEL_SIZE = 3  # 卷积核大小
STRIDE = 1  # 卷积步长
PADDING = 1  # 卷积填充

# 定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, dataframe, max_length=MAX_LENGTH):
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
        if mutation == 'WT':
            return [len(self.le.classes_)]  # 为WT数据使用一个特殊的编码
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
train_df = pd.read_csv(r'../data/compare/flu/flu_train_compared.csv')
valid_df = pd.read_csv(r'../data/compare/flu/flu_valid_compared.csv')
test_df = pd.read_csv(r'../data/compare/flu/flu_test_compared.csv')

# 创建数据集和数据加载器
train_dataset = ProteinDataset(train_df)
test_dataset = ProteinDataset(test_df)
valid_dataset = ProteinDataset(valid_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 定义Xception模块
class SeparableConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=KERNEL_SIZE, stride=STRIDE, padding=PADDING):
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
        self.block4 = XceptionBlock(hidden_dim, hidden_dim)  # 新增的层
        self.fc = nn.Linear(hidden_dim * max_length, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, max_length)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # 新增的层
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# 初始化模型、损失函数和优化器
input_dim = len(train_dataset.le.classes_) + 1  # 包含填充的0
model = Xception1D(input_dim, HIDDEN_DIM, OUTPUT_DIM, MAX_LENGTH)
model.to('cuda')  # 将模型移动到 GPU

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# 训练模型
train_losses = []
valid_losses = []
spearman_scores = []
epoch_times = []

start_time = time.time()
best_spearman = -1  # 初始化最佳Spearman相关系数

for epoch in range(NUM_EPOCHS):
    epoch_start_time = time.time()
    model.train()
    total_train_loss = 0
    for inputs, targets, _ in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{NUM_EPOCHS}'):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # 验证模型
    model.eval()
    total_valid_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets, _ in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{NUM_EPOCHS}'):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_valid_loss += loss.item()
            all_outputs.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)

    # 计算 Spearman 相关系数
    spearman_corr, _ = spearmanr(all_outputs, all_targets)
    spearman_scores.append(spearman_corr)

    # 保存训练集损失小于0.5且验证集损失小于0.5的Spearman指标最高的模型权重
    if  spearman_corr > best_spearman:
        best_spearman = spearman_corr
        if avg_train_loss < 0.1 and avg_valid_loss < 0.2:
            torch.save(model.state_dict(), '../output/XceptionV4/wt_con_best_xception1dV4BS64_model.pth')
            print(f'New con best model saved with Spearman: {spearman_corr:.4f}')
        else:
            torch.save(model.state_dict(), '../output/XceptionV4/wt_best_xception1dV4BS64_model.pth')
            print(f'New best model saved with Spearman: {spearman_corr:.4f}')

    # 打印和记录损失
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_time)
    print(f'Epoch {epoch + 1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Spearman: {spearman_corr:.4f}, Epoch Time: {epoch_time:.2f}s')

total_time = time.time() - start_time
print(f'Total Training Time: {total_time:.2f}s')

# 可视化训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(range(NUM_EPOCHS), train_losses, label='Train Loss')
plt.plot(range(NUM_EPOCHS), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('WT XceptionV4BS64 Training and Validation Loss')
plt.legend()
plt.savefig('wt_XceptionV4BS64_training_validation_loss.png')
plt.show()

# 可视化Spearman相关系数
plt.figure(figsize=(10, 5))
plt.plot(range(NUM_EPOCHS), spearman_scores, label='Spearman Correlation')
plt.xlabel('Epoch')
plt.ylabel('Spearman Correlation')
plt.title('WT XceptionV4BS64 Spearman Correlation Over Epochs')
plt.legend()
plt.savefig('wt_XceptionV4BS64_spearman_correlation.png')
plt.show()

# 加载Spearman指标最高的模型权重
model.load_state_dict(torch.load('../output/XceptionV4/wt_best_xception1dV4BS64_model.pth'))
model.eval()
# 测试模型
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
results_df.to_csv('flu_XceptionV4BS64_pre.csv', index=False)
print("预测值与实际值已保存为 flu_XceptionV4BS64_pre.csv")