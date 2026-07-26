from flask import Blueprint, jsonify, request

from model import FEATURE_FIELDS, load_model, predict


api = Blueprint("api", __name__, url_prefix="/api")


@api.get("/health")
def health():
    return jsonify({"status": "ok", "model_exists": load_model_exists()})


@api.post("/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    missing_fields = [field for field in FEATURE_FIELDS if field not in payload]
    if missing_fields:
        return jsonify({"error": f"Missing fields: {', '.join(missing_fields)}"}), 400

    try:
        values = {field: float(payload[field]) for field in FEATURE_FIELDS}
        model = load_model()
        prediction_value, probability = predict(model, values)
    except (ValueError, TypeError, FileNotFoundError) as error:
        status_code = 503 if isinstance(error, FileNotFoundError) else 400
        return jsonify({"error": str(error)}), status_code

    return jsonify({"prediction": prediction_value, "probability": probability})


def load_model_exists():
    try:
        load_model()
    except FileNotFoundError:
        return False
    return True