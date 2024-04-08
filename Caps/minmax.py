import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
data = pd.read_excel("E:/LiFePO4 Dynamic Profile Files/LiFePO4 Dynamic Profile Files/10도/LiFePO4_DST_SOC_10_1.xlsx")

# 정규화할 열 선택
columns_to_normalize = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'SOC(t)']

# Min-Max 스케일러 생성
scaler = MinMaxScaler()

# 선택한 열에 대해 Min-Max 스케일링 적용
data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

# 정규화된 데이터 출력
print(data.head())
