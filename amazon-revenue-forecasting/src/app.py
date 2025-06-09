import streamlit as st
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SalesLSTM

# Cấu hình thư mục
DATA_DIR = r"/workspaces/ducnandbdt/amazon-revenue-forecasting/data/sales_history"
MODEL_DIR = r"/workspaces/ducnandbdt/amazon-revenue-forecasting/best_models"

# Load mô hình từ file
@st.cache_resource
def load_model(asin_id):
    model = SalesLSTM()
    model_path = os.path.join(MODEL_DIR, f"lstm_model_{asin_id}.pt")
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# Hàm dự báo doanh thu ngày tiếp theo
def predict_next_day(model, recent_sales):
    input_seq = torch.tensor(recent_sales[-30:], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(input_seq).squeeze().item()
    return output

# Lấy danh sách file lịch sử
def get_available_products():
    files = os.listdir(DATA_DIR)
    return [f.replace("_sales_history.csv", "") for f in files if f.endswith("_sales_history.csv")]

# Giao diện Streamlit
st.set_page_config(page_title="Amazon Sales Forecast", layout="wide")

st.title("Amazon Sales Forecasting")
st.markdown("Chọn một mã sản phẩm để xem lịch sử doanh thu và dự báo cho ngày tiếp theo.")

available_asins = get_available_products()
asin_id = st.selectbox("🔍Chọn mã sản phẩm (ASIN):", available_asins)

# Đọc dữ liệu lịch sử
csv_path = os.path.join(DATA_DIR, f"{asin_id}_sales_history.csv")
df = pd.read_csv(csv_path, parse_dates=["date"])

# Hiển thị thông tin sản phẩm
with st.expander("Thông tin sản phẩm"):
    st.write(f"**Tên sản phẩm:** {df['title'].iloc[0]}")
    st.write(f"**Giá:** {df['price'].iloc[0]}")
    st.write(f"**Xếp hạng trung bình:** {df['average_rating'].iloc[0]}")
    st.write(f"**Tổng số lượt đánh giá:** {df['num_reviews'].iloc[0]}")

# Biểu đồ lịch sử doanh thu
st.subheader(" Doanh thu theo ngày")
st.line_chart(df.set_index("date")["daily_revenue"])

# Dự báo
if len(df) >= 30:
    recent_sales = df["daily_revenue"].values.reshape(-1, 1).astype(np.float32)
    recent_sales = (recent_sales - recent_sales.min()) / (recent_sales.max() - recent_sales.min())  # scale
    recent_sales = [[x[0]] for x in recent_sales]

    try:
        model = load_model(asin_id)
        prediction = predict_next_day(model, recent_sales)
        prediction_scaled = prediction * (df["daily_revenue"].max() - df["daily_revenue"].min()) + df["daily_revenue"].min()
        prediction_scaled = abs(prediction_scaled)
        st.success(f"Dự báo doanh thu ngày tiếp theo: **${prediction_scaled:.2f}**")
    except Exception as e:
        st.error(f"Không thể dự báo: {str(e)}")
else:
    st.warning("Cần ít nhất 30 ngày doanh thu để dự báo.")

