from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_distances

# --------------------------------------------------
# CONFIG (IDENTICAL TO NOTEBOOK)
# --------------------------------------------------
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_THRESHOLD = 0.70

# --------------------------------------------------
# LAZY LOAD MODEL (ONLY CHANGE)
# --------------------------------------------------
model = None

def get_model():
    global model
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME)
    return model

# --------------------------------------------------
# LOAD ARTIFACTS (UNCHANGED)
# --------------------------------------------------
cluster_centers_matrix = np.load("cluster_centers_matrix.npy")
cluster_ids_sorted = np.load("cluster_ids_sorted.npy")
cluster_labels = joblib.load("cluster_labels.pkl")

# --------------------------------------------------
# FLASK APP
# --------------------------------------------------
app = Flask(__name__, template_folder="templates")

# --------------------------------------------------
# HOME → FRONTEND
# --------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# --------------------------------------------------
# PREDICTION LOGIC (IDENTICAL)
# --------------------------------------------------
def predict_job_role_with_confidence(description, threshold=DEFAULT_THRESHOLD):
    emb = get_model().encode([description], convert_to_numpy=True)

    distances = cosine_distances(emb, cluster_centers_matrix)[0]
    best_index = np.argmin(distances)
    best_distance = distances[best_index]

    best_cluster_id = cluster_ids_sorted[best_index]

    confidence = 1 / (1 + best_distance)
    confidence = round(float(confidence), 4)

    if confidence < threshold:
        return "Unknown Role", confidence

    label = cluster_labels.get(best_cluster_id, "Unknown Role")
    return label, confidence

# --------------------------------------------------
# API — SINGLE PREDICTION
# --------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Text cannot be empty"}), 400

    label, confidence = predict_job_role_with_confidence(text)

    return jsonify({
        "predicted_role": label,
        "confidence": confidence
    })

# --------------------------------------------------
# API — BULK PREDICTION
# --------------------------------------------------
@app.route("/predict_bulk", methods=["POST"])
def predict_bulk():
    data = request.get_json()
    texts = data.get("texts", [])

    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({"error": "Texts must be a non-empty list"}), 400

    results = []
    for text in texts:
        label, confidence = predict_job_role_with_confidence(text)
        results.append({
            "input_text": text,
            "predicted_role": label,
            "confidence": confidence
        })

    return jsonify({"results": results})

# --------------------------------------------------
# RUN SERVER
# --------------------------------------------------
if __name__ == "__main__":
    print("🚀 Server running at http://127.0.0.1:5000/")
    app.run(debug=True)
