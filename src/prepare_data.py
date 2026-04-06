import pandas as pd
import numpy as np

# Đường dẫn đến các file dữ liệu (cần điều chỉnh nếu file nằm ở vị trí khác)
DEMO_PATH = './data/DEMO25Q4.txt'
DRUG_PATH = './data/DRUG25Q4.txt'
REAC_PATH = './data/REAC25Q4.txt'

def prepare_dataset():
    print("--- Đang đọc dữ liệu (Chỉ lấy các cột cần thiết để tiết kiệm RAM) ---")
    
    # 1. Đọc file DEMO (nhân khẩu học)
    demo = pd.read_csv(DEMO_PATH, sep='$', usecols=['primaryid', 'age', 'sex', 'wt'], low_memory=False)
    
    # 2. Đọc file REAC và dán nhãn ngay lập tức
    reac = pd.read_csv(REAC_PATH, sep='$', usecols=['primaryid', 'pt'], low_memory=False)
    addiction_keywords = ['drug dependence', 'withdrawal', 'drug abuse', 'substance use disorder', 'addiction']
    # Tạo nhãn: 1 nếu tìm thấy từ khóa gây nghiện trong cột 'pt'
    reac['label'] = reac['pt'].str.contains('|'.join(addiction_keywords), case=False, na=False).astype(int)
    # Gom nhóm theo primaryid: nếu 1 ca có nhiều phản ứng, chỉ cần 1 cái gây nghiện là tính cả ca đó là 1
    labels = reac.groupby('primaryid')['label'].max().reset_index()

    # 3. Đọc file DRUG
    drug = pd.read_csv(DRUG_PATH, sep='$', usecols=['primaryid', 'drugname', 'role_cod'], low_memory=False)
    # Chỉ lấy những thuốc là "Nghi ngờ chính" (Primary Suspect - PS)
    drug = drug[drug['role_cod'] == 'PS']

    print("--- Đang gộp dữ liệu (Merging) ---")
    # Gộp DEMO với Labels
    df = pd.merge(demo, labels, on='primaryid', how='inner')
    # Gộp tiếp với DRUG
    final_df = pd.merge(df, drug[['primaryid', 'drugname']], on='primaryid', how='inner')

    # Xử lý dữ liệu thiếu sơ bộ
    final_df['age'] = pd.to_numeric(final_df['age'], errors='coerce').fillna(final_df['age'].median(numeric_only=True))
    final_df['sex'] = final_df['sex'].fillna('Unknown')

    print(f"Hoàn thành! Kích thước tập dữ liệu: {final_df.shape}")
    print(f"Số ca gây nghiện tìm thấy: {final_df['label'].sum()}")
    
    # Lưu file đã làm sạch để chuẩn bị training
    final_df.to_csv('cleaned_data.csv', index=False)
    print("Đã lưu file: cleaned_data.csv")

if __name__ == "__main__":
    prepare_dataset()