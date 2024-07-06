import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
import gradio as gr
# from pypfopt.expected_returns import mean_historical_return
# from pypfopt.risk_models import sample_cov

# Đọc dữ liệu từ tệp CSV
data_path = 'HistoricalData_1720150942432.csv'
df = pd.read_csv(data_path)

df.rename(columns={'Close/Last': 'Close'}, inplace=True)

# Loại bỏ ký hiệu đô la và chuyển đổi về dạng số thực
df['Close'] = df['Close'].replace({'\$': ''}, regex=True).astype(float)

# Chuyển đổi cột 'Date' về định dạng datetime và sắp xếp theo thứ tự thời gian
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

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
        input_seq = torch.tensor(input_seq.values, dtype=torch.float32)
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
# loss_function = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.load_state_dict(torch.load('lstm_model.pth'))


# Hàm để dự đoán giá cổ phiếu
def predict_stock_prices(data, model, train_window):
    model.eval()
    predictions = []
    for i in range(len(data) - train_window):
        seq = data[i:i + train_window]
        with torch.no_grad():
            model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                                 torch.zeros(1, 1, model.hidden_layer_size))
            predictions.append(model(seq).item())
    return predictions


# Các hàm phụ trợ để vẽ biểu đồ
def plot_stock_prices(df):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Close Price'))
    fig.update_layout(title='Stock Prices Over Time', xaxis_title='Date', yaxis_title='Price', template='plotly_white')
    return fig


# def plot_cumulative_returns(df):
#     df['Cumulative Return'] = (1 + df['Close']).cumprod()
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=df['Date'], y=df['Cumulative Return'], mode='lines', name='Cumulative Return'))
#     fig.update_layout(title='Cumulative Returns Over Time', xaxis_title='Date', yaxis_title='Cumulative Return',
#                       template='plotly_white')
#     return fig


def plot_predicted_stock_prices(df, model, train_window):
    predictions = predict_stock_prices(df['Close'], model, train_window)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'][train_window:], y=predictions, mode='lines', name='Predicted Prices'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual Prices'))
    fig.update_layout(title='Predicted vs Actual Stock Prices', xaxis_title='Date', yaxis_title='Price',
                      template='plotly_white')
    return fig


def calculate_expected_returns_and_covariance(df):
    df = df.dropna().reset_index(drop=True)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna().reset_index(drop=True)
    returns = df['Close'].pct_change().dropna()
    returns_df = returns.to_frame(name='Returns')
    mu = mean_historical_return(returns_df)
    S = sample_cov(returns_df)
    return mu, S


# Thiết lập giao diện Gradio
def gradio_interface():
    stock_fig = plot_stock_prices(df)
    # cum_return_fig = plot_cumulative_returns(df)
    predicted_stock_fig = plot_predicted_stock_prices(df, model, train_window)
    # mu, S = calculate_expected_returns_and_covariance(df)
    return stock_fig, predicted_stock_fig


# Khởi chạy Gradio
interface = gr.Interface(
    fn=gradio_interface,
    inputs=[],
    outputs=["plot", "plot"],
    live=True,
    title="NVIDIA Stock Price Prediction and Analysis",
    description="This interface provides stock price predictions and various analytical plots based on historical data."
)

interface.launch()
