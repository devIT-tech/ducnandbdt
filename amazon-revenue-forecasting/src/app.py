# app.py

import streamlit as st
import pandas as pd
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from model import SalesLSTM
from flask import Flask, request, jsonify
import threading
import time
import requests
import json

# =========================
# 🔸 CẤU HÌNH THƯ MỤC
# =========================
DATA_DIR = "data/sales_history"
MODEL_DIR = "best_models"

# =========================
# 🔸 FLASK API PHỤ TRỢ (chạy nền)
# =========================

flask_app = Flask(__name__)

def get_sales_history(asin_id):
    file_path = os.path.join(DATA_DIR, f"{asin_id}_sales_history.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sales history for ASIN {asin_id} not found")
    df = pd.read_csv(file_path)
    return df

@flask_app.route("/")
def home():
    return "Welcome to Amazon Sales Forecasting API"

@flask_app.route("/history", methods=["GET"])
def history():
    try:
        asin_id = request.args.get("asin_id")
        if not asin_id:
            return jsonify({"error": "asin_id is required"}), 400

        df = get_sales_history(asin_id)

        if df.empty:
            return jsonify({"error": f"No sales data found for ASIN {asin_id}"}), 404

        history_data = df[["date", "daily_revenue"]].to_dict(orient="records")
        return jsonify({"asin_id": asin_id, "history": history_data})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@flask_app.route("/predict", methods=["POST"])
def forecast():
    try:
        data = request.get_json()
        asin_id = data.get("asin_id")
        recent_sales = data.get("recent_sales")

        if not asin_id or not recent_sales:
            return jsonify({"error": "asin_id and recent_sales are required"}), 400
        if len(recent_sales) < 30:
            return jsonify({"error": "Input sequence must have at least 30 data points"}), 400

        model = SalesLSTM()
        model_path = os.path.join(MODEL_DIR, f"lstm_model_{asin_id}.pt")
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        model.eval()

        input_seq = torch.tensor(recent_sales[-30:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            output = model(input_seq).squeeze().item()

        return jsonify({
            "asin_id": asin_id,
            "next_day_prediction": round(output, 2)
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def run_flask():
    flask_app.run(host="0.0.0.0", port=8000)

# Chạy Flask thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
time.sleep(1)

# =========================
# 🔸 STREAMLIT GIAO DIỆN
# =========================

st.set_page_config(page_title="Amazon Sales Forecast", layout="wide")
st.title("📦 Amazon Sales Forecasting")
st.markdown("Chọn một mã sản phẩm để xem lịch sử doanh thu và dự báo cho ngày tiếp theo.")

def get_available_products():
    files = os.listdir(DATA_DIR)
    return [f.replace("_sales_history.csv", "") for f in files if f.endswith("_sales_history.csv")]

available_asins = get_available_products()
asin_id = st.selectbox("🔍Chọn mã sản phẩm (ASIN):", available_asins)

csv_path = os.path.join(DATA_DIR, f"{asin_id}_sales_history.csv")
df = pd.read_csv(csv_path, parse_dates=["date"])

with st.expander("Thông tin sản phẩm"):
    st.write(f"**Tên sản phẩm:** {df['title'].iloc[0]}")
    st.write(f"**Giá:** {df['price'].iloc[0]}")
    st.write(f"**Xếp hạng trung bình:** {df['average_rating'].iloc[0]}")
    st.write(f"**Tổng số lượt đánh giá:** {df['num_reviews'].iloc[0]}")

st.subheader("📈 Doanh thu theo ngày")
st.line_chart(df.set_index("date")["daily_revenue"])

if len(df) >= 30:
    recent_sales = df["daily_revenue"].values.reshape(-1, 1).astype(np.float32)
    recent_sales_norm = (recent_sales - recent_sales.min()) / (recent_sales.max() - recent_sales.min())
    recent_sales_input = [[x[0]] for x in recent_sales_norm]

    try:
        response = requests.post(
            "http://localhost:8000/predict",
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "asin_id": asin_id,
                "recent_sales": recent_sales_input
            })
        )
        result = response.json()
        prediction = result["next_day_prediction"]
        prediction_scaled = prediction * (df["daily_revenue"].max() - df["daily_revenue"].min()) + df["daily_revenue"].min()
        prediction_scaled = abs(prediction_scaled)
        st.success(f"Dự báo doanh thu ngày tiếp theo: **${prediction_scaled:.2f}**")
    except Exception as e:
        st.error(f"Không thể dự báo: {str(e)}")
else:
    st.warning("Cần ít nhất 30 ngày doanh thu để dự báo.")
