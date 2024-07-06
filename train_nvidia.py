import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import gradio as gr

# Đọc dữ liệu từ tệp CSV
data_path = 'HistoricalData_1720150942432.csv'
df = pd.read_csv(data_path)

df.rename(columns={'Close/Last': 'Close'}, inplace=True)

print(df.columns)
# Loại bỏ ký hiệu đô la và chuyển đổi về dạng số thực
df['Close'] = df['Close'].replace({'\$': ''}, regex=True).astype(float)

# Tiền xử lý dữ liệu
scaler = MinMaxScaler(feature_range=(-1, 1))
df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_size = int(len(df) * 0.8)
train_data = df['Close'][:train_size]
test_data = df['Close'][train_size:]


# Tạo hàm để tạo các tập dữ liệu theo từng chuỗi thời gian
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L - tw):
        train_seq = input_data[i:i + tw]
        train_label = input_data[i + tw:i + tw + 1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_window = 12
train_inout_seq = create_inout_sequences(train_data, train_window)


# Xây dựng mô hình LSTM
class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super(LSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        input_seq = torch.tensor(input_seq, dtype=torch.float32)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Huấn luyện mô hình
epochs = 10

for i in range(epochs):
    mean_loss = []
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                             torch.zeros(1, 1, model.hidden_layer_size))

        seq = torch.FloatTensor(seq.values)
        labels = torch.FloatTensor(labels.values)
        y_pred = model(seq)
        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

        mean_loss.append(single_loss.detach().numpy())

    print(f'epoch: {i:3} loss: {np.array(mean_loss).mean():10.10f}')

# Save the pretrained model
torch.save(model.state_dict(), 'lstm_model.pth')
