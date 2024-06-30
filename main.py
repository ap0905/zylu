from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
from surprise import Dataset, Reader, SVD
import pandas as pd

# Load scikit-learn models
model = joblib.load('/Users/alfinpatel/Downloads/intern-task/models/customer_return_model.pkl')
model_repeat = joblib.load('/Users/alfinpatel/Downloads/intern-task/models/repeat_purchase_model.pkl')

# Load Surprise SVD model
df = pd.read_excel('/Users/alfinpatel/Downloads/intern-task/online_retail_II.xlsx')
df['TotalPrice'] = df['Quantity'] * df['Price']
df['TotalPrice'] = (df['TotalPrice'] - df['TotalPrice'].mean()) / df['TotalPrice'].std()
df_filtered = df[['Customer ID', 'StockCode', 'TotalPrice']]
reader = Reader(rating_scale=(df_filtered['TotalPrice'].min(), df_filtered['TotalPrice'].max()))
data = Dataset.load_from_df(df_filtered, reader)
algo = SVD()
trainset = data.build_full_trainset()
algo.fit(trainset)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict_return', methods=['POST'])
def predict_return():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        recency = data.get('recency')
        frequency = data.get('frequency')
        monetary = data.get('monetary')

        # Perform prediction based on input data
        input_data = np.array([[recency, frequency, monetary]])
        prediction = model.predict(input_data)[0]

        return jsonify({'returning_customer': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/predict_repeat', methods=['POST'])
def predict_repeat():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        recency = data.get('recency')
        frequency = data.get('frequency')
        monetary = data.get('monetary')

        input_data = np.array([[recency, frequency, monetary]])
        prediction = model_repeat.predict(input_data)[0]

        return jsonify({'repeat_purchase': int(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/recommend_products', methods=['POST'])
def recommend_products():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        recommendations = get_recommendations(customer_id)

        return jsonify({'recommendations': recommendations})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_recommendations(customer_id, top_n=10):
    try:
        items = df['StockCode'].unique()
        est_ratings = []
        for item in items:
            est_rating = algo.predict(customer_id, item).est
            est_ratings.append((item, est_rating))
        est_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recommendations = est_ratings[:top_n]

        recommendations = []
        for stock_code, rating in top_recommendations:
            product_name = df[df['StockCode'] == stock_code]['Description'].values[0]
            recommendations.append({'product_name': product_name, 'rating': rating})

        return recommendations
    except Exception as e:
        return {'error': str(e)}  # Handle any exceptions during recommendation generation

@app.route('/test', methods=['GET'])
def test():
    return jsonify({'message': 'API is working'})

if __name__ == '__main__':
    app.run(debug=True)
