import pandas as pd

# 读取原始的 CSV 文件
input_csv = r'C:\Users\逖姬\Desktop\CS\Python\transformerDemo\data\extract\flu\fluorescence_valid.csv'
df = pd.read_csv(input_csv)

# 检查是否存在 Num_Mutations 列
if 'Num_Mutations' not in df.columns:
    raise ValueError("The column 'Num_Mutations' does not exist in the data.")

# 查找突变数为0的行数据，即WT序列
wt_rows = df[df['Num_Mutations'] == 0]
if wt_rows.empty:
    raise ValueError("No row found with 'Num_Mutations' equal to 0.")

wt_row = wt_rows.iloc[0]
wt_sequence = wt_row['Primary']

print(f"WT Sequence: {wt_sequence}")

def generate_mutation(wt_seq, seq):
    mutations = []
    for i, (wt, mut) in enumerate(zip(wt_seq, seq)):
        if wt != mut:
            mutations.append(f"{wt}{i+1}{mut}")
    return '-'.join(mutations) if mutations else 'WT'

# 生成 Mutation 列
df['Mutation'] = df['Primary'].apply(lambda seq: generate_mutation(wt_sequence, seq))

# 重新排列列顺序，将 Mutation 列放到第一列
cols = ['Mutation'] + [col for col in df.columns if col != 'Mutation']
df = df[cols]

# 将 WT 行移动到第一行
wt_index = df[df['Mutation'] == 'WT'].index[0]
df = pd.concat([df.loc[[wt_index]], df.drop(wt_index)], ignore_index=True)

# 保存到新的 CSV 文件
output_csv = 'flu_valid_compared.csv'
df.to_csv(output_csv, index=False)

print(f"Processed data saved to {output_csv}")
