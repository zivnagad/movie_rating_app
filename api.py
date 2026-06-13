from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

from assets_data_prep import prepare_data

app = Flask(__name__)

# Load model once on startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # --- Validate required fields ---
        required_fields = ['startYear', 'runtimeMinutes', 'genres', 'Language', 'Country']
        for field in required_fields:
            if field not in data or str(data[field]).strip() == '':
                return jsonify({'error': f'שדה חסר: {field}'}), 400

        # --- Validate numeric fields ---
        try:
            start_year = float(data['startYear'])
            runtime = float(data['runtimeMinutes'])
        except (ValueError, TypeError):
            return jsonify({'error': 'שנת יציאה ואורך הסרט חייבים להיות מספרים'}), 400

        if not (1900 <= start_year <= 2025):
            return jsonify({'error': 'שנת יציאה חייבת להיות בין 1900 ל-2025'}), 400

        if not (1 <= runtime <= 500):
            return jsonify({'error': 'אורך הסרט חייב להיות בין 1 ל-500 דקות'}), 400
        
        # --- Build single-row DataFrame ---
        row = {
            'startYear':      start_year,
            'runtimeMinutes': runtime,
            'genres':         str(data['genres']),
            'Language':       str(data['Language']),
            'Country':        str(data['Country']),
        }
        df_input = pd.DataFrame([row])

        # --- Prepare features ---
        df_prepared = prepare_data(df_input)

        # --- Predict ---
        prediction = model.predict(df_prepared)[0]
        predicted_rating = round(float(prediction), 1)

        return jsonify({'predicted_rating': predicted_rating})

    except KeyError as e:
        return jsonify({'error': f'שדה חסר: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'שגיאה פנימית: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
