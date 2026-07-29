from flask import Blueprint, jsonify, request

from model import load_model, model_exists, predict, validate_feature_values


api = Blueprint("api", __name__, url_prefix="/api")


@api.get("/health")
def health():
    return jsonify({"status": "ok", "model_exists": model_exists()})


@api.post("/predict")
def api_predict():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    try:
        values = validate_feature_values(payload)
        model = load_model()
        prediction_value, probability = predict(model, values)
    except (ValueError, TypeError) as error:
        return jsonify({"error": str(error)}), 400
    except (FileNotFoundError, OSError) as error:
        return jsonify({"error": str(error)}), 503

    return jsonify({"prediction": prediction_value, "probability": probability})
