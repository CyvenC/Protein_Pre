import os
import lmdb
import pickle
import csv

# 读取 lmdb 文件
def read_lmdb(lmdb_path):
    env = lmdb.open(lmdb_path, readonly=True, lock=False)
    with env.begin() as txn:
        cursor = txn.cursor()
        for key, value in cursor:
            try:
                data = pickle.loads(value)
                yield key, data
            except Exception as e:
                print(f"Error reading key {key}: {e}")

# 将数据保存到 CSV 文件
def save_to_csv(data_generator, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Key', 'Primary', 'Protein_Length', 'Log_Fluorescence', 'Num_Mutations'])  # Example header
        for key, data in data_generator:
            if isinstance(data, dict):
                primary = data.get('primary', '')
                protein_length = data.get('protein_length', 0)
                log_fluorescence = data.get('log_fluorescence', [0.0])[0]
                num_mutations = data.get('num_mutations', 0)
                writer.writerow([key.decode('utf-8'), primary, protein_length, log_fluorescence, num_mutations])
            else:
                print(f"Skipping key {key.decode('utf-8')}: data is not a dictionary")

# 主流程
extract_path = r'../data/origin'

# 遍历解压后的目录，处理其中的 lmdb 文件
for root, dirs, files in os.walk(extract_path):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        lmdb_path = os.path.join(dir_path, 'data.mdb')
        if os.path.exists(lmdb_path):
            data_generator = read_lmdb(dir_path)
            output_csv = f'{dir_name}.csv'
            save_to_csv(data_generator, output_csv)
