import os
import pickle
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# --- Load Model & Vectorizer ---
MODEL_PATH = 'model.pkl'
VECTORIZER_PATH = 'vectorizer.pkl'

model = None
vectorizer = None

def load_assets():
    global model, vectorizer
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        with open(MODEL_PATH, 'rb') as f_model:
            model = pickle.load(f_model)
        with open(VECTORIZER_PATH, 'rb') as f_vec:
            vectorizer = pickle.load(f_vec)
    else:
        print("Warning: 'model.pkl' or 'vectorizer.pkl' not found. Ensure both files exist in the directory.")

load_assets()

# --- HTML / CSS Layout ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sentiment Analysis Engine</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --accent-blue: #38bdf8;
            --accent-purple: #818cf8;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --pos-color: #4ade80;
            --neg-color: #f87171;
            --border-color: #334155;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: var(--bg-color);
            color: var(--text-main);
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.5), 0 8px 10px -6px rgba(0, 0, 0, 0.5);
            border: 1px solid var(--border-color);
        }

        .header {
            text-align: center;
            margin-bottom: 30px;
        }

        .header h1 {
            font-size: 2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 8px;
        }

        .header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 12px;
            margin-bottom: 24px;
        }

        textarea {
            width: 100%;
            height: 140px;
            background: #0f172a;
            border: 1px solid var(--border-color);
            border-radius: 10px;
            padding: 14px;
            color: var(--text-main);
            font-size: 1rem;
            resize: vertical;
            transition: border-color 0.2s, box-shadow 0.2s;
        }

        textarea:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.25);
        }

        button {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #0f172a;
            font-weight: 600;
            font-size: 1rem;
            padding: 14px;
            border: none;
            border-radius: 10px;
            cursor: pointer;
            transition: opacity 0.2s, transform 0.1s;
        }

        button:hover {
            opacity: 0.9;
        }

        button:active {
            transform: scale(0.99);
        }

        .result-box {
            display: none;
            margin-top: 20px;
            padding: 20px;
            border-radius: 10px;
            background: #0f172a;
            border: 1px solid var(--border-color);
            text-align: center;
        }

        .result-title {
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: var(--text-muted);
            margin-bottom: 8px;
        }

        .badge {
            display: inline-block;
            font-size: 1.5rem;
            font-weight: 700;
            padding: 6px 18px;
            border-radius: 20px;
            text-transform: capitalize;
        }

        .positive {
            color: var(--pos-color);
            background: rgba(74, 222, 128, 0.1);
            border: 1px solid var(--pos-color);
        }

        .negative {
            color: var(--neg-color);
            background: rgba(248, 113, 113, 0.1);
            border: 1px solid var(--neg-color);
        }

        .error {
            color: var(--neg-color);
            font-size: 0.9rem;
            margin-top: 10px;
            display: none;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="header">
        <h1>Sentiment Analyzer</h1>
        <p>Type standard text below to run inference using your trained Naive Bayes classifier.</p>
    </div>

    <div class="input-group">
        <textarea id="textInput" placeholder="Enter text here..."></textarea>
        <button onclick="analyzeSentiment()">Analyze Sentiment</button>
    </div>

    <div id="errorBox" class="error"></div>

    <div id="resultBox" class="result-box">
        <div class="result-title">Predicted Sentiment</div>
        <div id="badge" class="badge"></div>
    </div>
</div>

<script>
    async function analyzeSentiment() {
        const text = document.getElementById('textInput').value.trim();
        const errorBox = document.getElementById('errorBox');
        const resultBox = document.getElementById('resultBox');
        const badge = document.getElementById('badge');

        errorBox.style.display = 'none';
        resultBox.style.display = 'none';

        if (!text) {
            errorBox.textContent = 'Please provide text before analyzing.';
            errorBox.style.display = 'block';
            return;
        }

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: text })
            });

            const data = await response.json();

            if (!response.ok) {
                errorBox.textContent = data.error || 'An error occurred on prediction.';
                errorBox.style.display = 'block';
                return;
            }

            badge.textContent = data.sentiment;
            badge.className = 'badge ' + data.sentiment.toLowerCase();
            resultBox.style.display = 'block';

        } catch (err) {
            errorBox.textContent = 'Failed to connect to backend server.';
            errorBox.style.display = 'block';
        }
    }
</script>

</body>
</html>
"""

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or Vectorizer pickle files not loaded correctly on server.'}), 500

    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    transformed_text = vectorizer.transform([text])
    prediction = model.predict(transformed_text)[0]

    return jsonify({'sentiment': str(prediction)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
