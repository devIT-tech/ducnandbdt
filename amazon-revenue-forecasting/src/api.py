from flask import Flask, request, jsonify
from inference import predict_sales
import pandas as pd
import os

app = Flask(__name__)

def get_sales_history(asin_id, data_dir=r"/workspaces/ducnandbdt/amazon-revenue-forecasting/data/sales_history"):
    file_path = os.path.join(data_dir, f"{asin_id}_sales_history.csv")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sales history for ASIN {asin_id} not found")
    df = pd.read_csv(file_path)
    return df


@app.route("/")
def home():
    return "Welcome to Amazon Sales Forecasting API"

# Endpoint trả về lịch sử doanh thu
@app.route("/history", methods=["GET"])
def history():
    try:
        # Lấy mã sản phẩm từ query parameter
        asin_id = request.args.get("asin_id")
        if not asin_id:
            return jsonify({"error": "asin_id is required"}), 400
        
        # Lọc lịch sử doanh thu cho sản phẩm
        product_history = df[df["asin_id"] == asin_id]
        
        if product_history.empty:
            return jsonify({"error": f"No sales data found for ASIN {asin_id}"}), 404
        
        # Chuyển đổi lịch sử doanh thu thành danh sách để trả về
        history_data = product_history[["date", "revenue"]].to_dict(orient="records")
        
        return jsonify({
            "asin_id": asin_id,
            "history": history_data
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def forecast():
    try:
        data = request.get_json()

        # Kiểm tra định dạng đầu vào
        if "asin_id" not in data or "recent_sales" not in data:
            return jsonify({"error": "asin_id and recent_sales are required"}), 400

        asin_id = data["asin_id"]
        recent_sales = data["recent_sales"]

        if len(recent_sales) < 30:
            return jsonify({"error": "Input sequence must have at least 30 data points"}), 400

        # Dự báo doanh thu cho ngày tiếp theo
        input_seq = [[x] for x in recent_sales[-30:]]  # Chỉ lấy 30 ngày cuối cùng để làm đầu vào

        prediction = predict_sales(asin_id, input_seq)

        return jsonify({
            "asin_id": asin_id,
            "next_day_prediction": round(prediction, 2)
        })

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
