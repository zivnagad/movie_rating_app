# Movie Rating Prediction App

A Flask web application that predicts IMDB movie ratings using a trained ElasticNet machine learning model.

## Team
- Ziv Nagad — 322558271
- Tiferet Baluka — 325204113

## Installation

1. Create a virtual environment:
python -m venv venv

2. Activate the virtual environment:
- Windows: venv\Scripts\activate
- Mac/Linux: source venv/bin/activate

3. Install dependencies:
pip install -r requirements.txt

## Running the App

python api.py

Then open your browser at:
http://localhost:5000

## Input Fields

| Field | Description | Expected Range |
|-------|-------------|----------------|
| startYear | Movie release year | 1900 – 2025 |
| runtimeMinutes | Movie duration in minutes | 1 – 500 |
| Language | Primary language | e.g. English, French |
| Country | Country of production | e.g. United States, Japan |
| genres | One or more genres | Drama, Comedy, Documentary, Horror, Action, Romance, Thriller, Crime |
