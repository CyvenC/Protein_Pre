import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
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
        return torch.tensor(self.mutations_padded[idx], dtype=torch.long), torch.tensor(self.log_fluorescence.iloc[idx], dtype=torch.float32)


# 读取数据
train_df = pd.read_csv(r'C:\Users\逖姬\Desktop\CS\Python\transformerDemo\data\compare\flu\flu_train_compared.csv')
test_df = pd.read_csv(r'C:\Users\逖姬\Desktop\CS\Python\transformerDemo\data\compare\flu\flu_test_compared.csv')
valid_df = pd.read_csv(r'C:\Users\逖姬\Desktop\CS\Python\transformerDemo\data\compare\flu\flu_valid_compared.csv')

max_length = 237  # 固定序列长度为 237

# 创建数据集和数据加载器
train_dataset = ProteinDataset(train_df, max_length=max_length)
test_dataset = ProteinDataset(test_df, max_length=max_length)
valid_dataset = ProteinDataset(valid_df, max_length=max_length)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)


# 定义神经网络模型
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, max_length):
        super(SimpleNN, self).__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * max_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# 初始化模型、损失函数和优化器
input_dim = len(train_dataset.le.classes_) + 1  # 包含填充的0
hidden_dim = 256
output_dim = 1

model = SimpleNN(input_dim, hidden_dim, output_dim, max_length)
model.to('cuda')  # 将模型移动到 GPU

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 初始化TensorBoard
log_dir = 'logsv3'  # 指定日志目录
writer = SummaryWriter(log_dir=log_dir)

# 训练模型
num_epochs = 100
train_losses = []
valid_losses = []
epoch_times = []

start_time = time.time()

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    model.train()
    total_train_loss = 0
    for inputs, targets in tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{num_epochs}'):
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
        for inputs, targets in tqdm(valid_loader, desc=f'Validation Epoch {epoch + 1}/{num_epochs}'):
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

    # 打印和记录损失
    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time
    epoch_times.append(epoch_time)
    print(
        f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_valid_loss:.4f}, Spearman: {spearman_corr:.4f}, Epoch Time: {epoch_time:.2f}s')
    writer.add_scalar('Loss/Train', avg_train_loss, epoch)
    writer.add_scalar('Loss/Validation', avg_valid_loss, epoch)
    writer.add_scalar('Time/Epoch', epoch_time, epoch)
    writer.add_scalar('Spearman/Validation', spearman_corr, epoch)

total_time = time.time() - start_time
print(f'Total Training Time: {total_time:.2f}s')

# 关闭TensorBoard记录器
writer.close()

# 保存模型
torch.save(model.state_dict(), 'simple_nn_model.pth')
print("Model saved as simple_nn_model.pth")

# 可视化训练和验证损失
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), valid_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation LossV3')
plt.legend()
plt.savefig('training_validation_lossV3.png')
plt.show()



# 测试模型
model.eval()
total_test_loss = 0
all_outputs = []
all_targets = []
with torch.no_grad():
    for inputs, targets in tqdm(test_loader, desc='Testing'):
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        total_test_loss += loss.item()
        all_outputs.extend(outputs.squeeze().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
avg_test_loss = total_test_loss / len(test_loader)
spearman_corr, _ = spearmanr(all_outputs, all_targets)
print(f'Test Loss: {avg_test_loss:.4f}, Spearman: {spearman_corr:.4f}')

# 保存预测值与实际值到CSV文件
results_df = pd.DataFrame({
    'Actual': all_targets,
    'Predicted': all_outputs
})
results_df.to_csv('flu_prev3.csv', index=False)
print("预测值与实际值已保存为 flu_pre.csv")
