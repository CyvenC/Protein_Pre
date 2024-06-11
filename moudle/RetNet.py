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
train_df = pd.read_csv(r'../data/compare/flu/flu_train_compared.csv')
valid_df = pd.read_csv(r'../data/compare/flu/flu_valid_compared.csv')
test_df = pd.read_csv(r'../data/compare/flu/flu_test_compared.csv')

max_length = 237  # 固定序列长度为 237

# 创建数据集和数据加载器
train_dataset = ProteinDataset(train_df, max_length=max_length)
test_dataset = ProteinDataset(test_df, max_length=max_length)
valid_dataset = ProteinDataset(valid_df, max_length=max_length)
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# 定义残差块
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 定义神经网络模型
class ResNet1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_length):
        super(ResNet1D, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.layer1 = ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.layer2 = ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.layer3 = ResidualBlock(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(hidden_dim * max_length, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # (batch_size, hidden_dim, max_length)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 初始化模型、损失函数和优化器
input_dim = len(train_dataset.le.classes_) + 1  # 包含填充的0
hidden_dim = 256
output_dim = 1

model = ResNet1D(input_dim, hidden_dim, output_dim, max_length)
model.to('cuda')  # 将模型移动到 GPU

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 150
train_losses = []
valid_losses = []
spearman_scores = []
epoch_times = []

start_time = time.time()
best_spearman = -1  # 初始化最佳Spearman相关系数

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    total_train_loss = 0
    for inputs, targets, _ in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
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
        for inputs, targets, _ in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
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
        if avg_train_loss < 0.5 and avg_valid_loss < 0.5:
            torch.save(model.state_dict(), '../output/resnet/120_con_best_resnet1d_model.pth')
            print(f'New 120 con best model saved with Spearman: {spearman_corr:.4f}')
        else:
            torch.save(model.state_dict(), '../output/resnet/120_best_resnet1d_model.pth')
            print(f'New 120 best model saved with Spearman: {spearman_corr:.4f}')


    # 打印和记录损失
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_time)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Spearman: {spearman_corr:.4f}, Epoch Time: {epoch_time:.2f}s')

total_time = time.time() - start_time
print(f'Total Training Time: {total_time:.2f}s')

# 可视化训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ResNet Training and Validation Loss')
plt.legend()
plt.savefig('ResNet_training_validation_loss.png')
plt.show()

# 可视化Spearman相关系数
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), spearman_scores, label='Spearman Correlation')
plt.xlabel('Epoch')
plt.ylabel('Spearman Correlation')
plt.title('120 ResNet Spearman Correlation Over Epochs')
plt.legend()
plt.savefig('120_ResNet_spearman_correlation.png')
plt.show()

# 加载Spearman指标最高的模型权重
model.load_state_dict(torch.load('../output/resnet/120_con_best_resnet1d_model.pth'))
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
results_df.to_csv('120_flu_ResNet_pre.csv', index=False)
print("预测值与实际值已保存为 120_flu_ResNet_pre.csv")