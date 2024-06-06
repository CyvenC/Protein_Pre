import pandas as pd

# 读取原始的 CSV 文件
input_csv = r'C:\Users\逖姬\Desktop\CS\Python\transformerDemo\data\extract\flu\fluorescence_valid.csv'
df = pd.read_csv(input_csv)

# 定义 WT 序列
wt_sequence = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

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

# 保存到新的 CSV 文件
output_csv = 'flu_valid_compared.csv'
df.to_csv(output_csv, index=False)

print(f"Processed data saved to {output_csv}")
