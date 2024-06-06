import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import time
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import GradScaler, autocast

# 定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, dataframe):
        self.data = dataframe
        self.mutations = self.data['Mutation']
        self.log_fluorescence = self.data['Log_Fluorescence']
        self.le = LabelEncoder()
        self.le.fit([char for mutation in self.mutations for char in mutation.replace('-', '') if mutation != 'WT'])  # 忽略WT标记
        self.mutations_encoded = [self.encode_mutation(mutation) for mutation in self.mutations]

    def encode_mutation(self, mutation):
        if mutation == 'WT':
            return [0]  # WT标记为单个0向量
        return self.le.transform(list(mutation.replace('-', '')))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mutation = torch.tensor(self.mutations_encoded[idx], dtype=torch.long)
        log_fluorescence = torch.tensor(self.log_fluorescence.iloc[idx], dtype=torch.float32)
        return mutation, log_fluorescence

# 自定义collate函数
def collate_fn(batch):
    mutations, log_fluorescences = zip(*batch)
    mutations_padded = pad_sequence(mutations, batch_first=True, padding_value=0)
    log_fluorescences = torch.tensor(log_fluorescences)
    return mutations_padded, log_fluorescences

# Transformer模型定义
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, dim_feedforward):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        self.fc = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, x):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# 数据加载
train_df = pd.read_csv(r'../data/compare/flu/flu_train_compared.csv')
valid_df = pd.read_csv(r'../data/compare/flu/flu_valid_compared.csv')
test_df = pd.read_csv(r'../data/compare/flu/flu_test_compared.csv')

train_dataset = ProteinDataset(train_df)
valid_dataset = ProteinDataset(valid_df)
test_dataset = ProteinDataset(test_df)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, collate_fn=collate_fn)

# 模型、损失函数和优化器
input_dim = len(train_dataset.le.classes_) + 1
d_model = 128
nhead = 8
num_encoder_layers = 3
dim_feedforward = 512
model = TransformerModel(input_dim, d_model, nhead, num_encoder_layers, dim_feedforward)
model.to('cuda')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = GradScaler()  # 混合精度训练

# 训练和验证模型
num_epochs = 50
train_losses = []
valid_losses = []
epoch_times = []  # 记录每个epoch的时间

start_time = time.time()  # 记录总训练时间

for epoch in range(num_epochs):
    epoch_start_time = time.time()  # 记录每个epoch的开始时间
    model.train()
    total_train_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        optimizer.zero_grad()
        with autocast():  # 使用混合精度训练
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    total_valid_loss = 0
    all_outputs = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
            inputs, targets = inputs.to('cuda'), targets.to('cuda')
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            total_valid_loss += loss.item()
            all_outputs.extend(outputs.squeeze().cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    avg_valid_loss = total_valid_loss / len(valid_loader)
    valid_losses.append(avg_valid_loss)

    spearman_corr, _ = spearmanr(all_outputs, all_targets)
    epoch_end_time = time.time()  # 记录每个epoch的结束时间
    epoch_time = epoch_end_time - epoch_start_time  # 计算每个epoch的用时
    epoch_times.append(epoch_time)  # 保存每个epoch的用时
    print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Spearman: {spearman_corr:.4f}, Epoch Time: {epoch_time:.2f}s')

total_time = time.time() - start_time  # 计算总训练时间
print(f'Total Training Time: {total_time:.2f}s')

# 保存模型
torch.save(model.state_dict(), 'transformer_model.pth')
print("Model saved as transformer_model.pth")

# 可视化训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Transformers Training and Validation Loss')
plt.legend()
plt.savefig('transformers_training_validation_loss.png')
plt.show()

# 可视化每个epoch的用时
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), epoch_times, label='Epoch Time')
plt.xlabel('Epoch')
plt.ylabel('Time (s)')
plt.title('Time per Epoch')
plt.legend()
plt.savefig('epoch_times.png')
plt.show()

# 测试模型并保存结果
model.eval()
total_test_loss = 0
all_outputs = []
all_targets = []
mutations = []
with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc='Testing'):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        total_test_loss += loss.item()
        all_outputs.extend(outputs.squeeze().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        mutations.extend(inputs.cpu().numpy())
avg_test_loss = total_test_loss / len(test_loader)
spearman_corr, _ = spearmanr(all_outputs, all_targets)
print(f'Test Loss: {avg_test_loss:.4f}, Spearman: {spearman_corr:.4f}')

# 保存预测值与实际值到CSV文件
results_df = pd.DataFrame({
    'Mutation': [''.join(train_dataset.le.inverse_transform(mutation)) for mutation in mutations],
    'Actual': all_targets,
    'Predicted': all_outputs
})
results_df.to_csv('flu_pre.csv', index=False)
print("预测值与实际值已保存为 flu_pre.csv")

# 可视化Spearman相关系数
plt.figure(figsize=(10, 5))
plt.scatter(all_targets, all_outputs, alpha=0.5)
plt.xlabel('Actual Log Fluorescence')
plt.ylabel('Predicted Log Fluorescence')
plt.title(f'Transformers Spearman Correlation: {spearman_corr:.4f}')
plt.savefig('transformer_spearman_correlation.png')
plt.show()
