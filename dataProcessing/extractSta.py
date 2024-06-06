import os
import lmdb
import pickle
import csv

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

def save_to_csv(data_generator, output_csv):
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        first_row = True
        for key, data in data_generator:
            if isinstance(data, dict):
                if first_row:
                    headers = ['Key'] + list(data.keys())
                    writer.writerow(headers)
                    first_row = False
                row = [key.decode('utf-8')] + list(data.values())
                writer.writerow(row)
            else:
                print(f"Skipping key {key.decode('utf-8')}: data is not a dictionary")

# 主流程
base_path = r'/data/origin/stability'

# 遍历解压后的目录，处理其中的 lmdb 文件
for root, dirs, files in os.walk(base_path):
    for dir_name in dirs:
        dir_path = os.path.join(root, dir_name)
        lmdb_path = os.path.join(dir_path, 'data.mdb')
        if os.path.exists(lmdb_path):
            data_generator = read_lmdb(dir_path)
            output_csv = f'{dir_name.replace(".lmdb", "")}.csv'
            save_to_csv(data_generator, output_csv)
            print(f"Data from {lmdb_path} saved to {output_csv}")
