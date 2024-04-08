import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 데이터 불러오기
data = pd.read_excel("E:/LiFePO4 Dynamic Profile Files/LiFePO4 Dynamic Profile Files/10도/LiFePO4_DST_SOC_10_1.xlsx")

# 비수치 열 제외
numerical_columns = ['Test_Time(s)', 'Step_Time(s)', 'Step_Index', 'Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'SOC(t)']
data_numeric = data[numerical_columns]

# 결측치 처리
data_numeric = data_numeric.dropna()

# 정규화할 열 선택
columns_to_normalize = ['Current(A)', 'Voltage(V)', 'Temperature (C)_1', 'SOC(t)']

# Min-Max 스케일러 생성
scaler = MinMaxScaler()

# 선택한 열에 대해 Min-Max 스케일링 적용
data_normalized = pd.DataFrame(scaler.fit_transform(data_numeric[columns_to_normalize]), columns=columns_to_normalize)

# 정규화된 데이터 출력
print(data_normalized.head())

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# TCN 모델 정의
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size - 1) // 2),
            nn.ReLU(),
            nn.Conv1d(num_channels, output_size, kernel_size, padding=(kernel_size - 1) // 2)
        )

    def forward(self, x):
        return self.tcn(x)



# 데이터셋 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx])

# 모델 학습을 위한 데이터 로드
# 예시 데이터 (data_normalized를 사용)
normalized_data = data_normalized.values

# 데이터셋 및 데이터로더 생성
dataset = CustomDataset(normalized_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 모델 인스턴스화 및 손실 함수 및 옵티마이저 정의
model = TCN(input_size=1, output_size=1, num_channels=64, kernel_size=3, dropout=0.2)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 10
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        inputs = batch[:, :-1]  # 입력 데이터
        targets = batch[:, -1]   # 타겟 데이터
        outputs = model(inputs.float().unsqueeze(1))

        loss = criterion(outputs.squeeze(), targets)  # 손실 계산
        loss.backward()
        optimizer.step()


# 학습된 모델을 사용하여 추론 등을 수행할 수 있습니다.
