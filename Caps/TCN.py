import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 파일 경로 설정
file_path_1 = "E:\\LiFePO4 Dynamic Profile Files\\LiFePO4 Dynamic Profile Files\\0도\\LiFePO4_DST_SOC_0_1.xlsx"
file_path_2 = "E:\\LiFePO4 Dynamic Profile Files\\LiFePO4 Dynamic Profile Files\\0도\\LiFePO4_DST_SOC_0_2.xlsx"

# 데이터 불러오기
data_1 = pd.read_excel(file_path_1)
data_2 = pd.read_excel(file_path_2)

# 데이터 확인
print(data_1.head())
print(data_2.head())

# 결측치 확인
print(data_1.isnull().sum())
print(data_2.isnull().sum())

# 결측치 처리
data_1 = data_1.fillna(data_1.mean())
data_2 = data_2.fillna(data_2.mean())

# Min-Max 정규화
scaler = MinMaxScaler()
normalized_data_1 = pd.DataFrame(scaler.fit_transform(data_1), columns=data_1.columns)
normalized_data_2 = pd.DataFrame(scaler.fit_transform(data_2), columns=data_2.columns)

# 정규화된 데이터 확인
print(normalized_data_1.head())
print(normalized_data_2.head())

