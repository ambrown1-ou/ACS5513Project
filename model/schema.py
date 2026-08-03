from copy import deepcopy


SCHEMA_ID = "cleveland_v1"
SCHEMA_VERSION = "1"
SCHEMA_LABEL = "Cleveland heart disease schema"

FIELD_DEFINITIONS = {
    "age": {
        "label": "Age",
        "alias_label": "Age",
        "role": "feature",
        "units": "years",
        "aliases": ["age", "age_years", "patient_age"],
        "description": "Age of the patient.",
        "minimum": 0,
        "maximum": 120,
    },
    "sex": {
        "label": "Sex",
        "alias_label": "Sex",
        "role": "feature",
        "units": "binary",
        "aliases": ["sex", "gender", "biological_sex"],
        "description": "Biological sex of the patient.",
        "allowed": [0, 1],
    },
    "cp": {
        "label": "Chest Pain",
        "alias_label": "Chest Pain",
        "role": "feature",
        "units": "coded",
        "aliases": ["cp", "chest_pain", "chest_pain_type", "chestpain"],
        "description": "Type of chest pain experienced.",
        "allowed": [1, 2, 3, 4],
    },
    "trestbps": {
        "label": "Blood Pressure",
        "alias_label": "Blood Pressure",
        "role": "feature",
        "units": "mmHg",
        "aliases": [
            "trestbps",
            "resting_bp",
            "resting_blood_pressure",
            "systolic_bp",
            "blood_pressure",
        ],
        "description": "Resting blood pressure on admission.",
        "minimum": 0,
        "maximum": 300,
        "unit_options": ["mmHg", "kPa"],
        "conversions": {
            "kPa": {"id": "kPa_to_mmHg", "factor": 7.50061683, "offset": 0},
        },
    },
    "chol": {
        "label": "Cholesterol",
        "alias_label": "Cholesterol",
        "role": "feature",
        "units": "mg/dL",
        "aliases": ["chol", "cholesterol", "serum_cholesterol", "serum_cholestrol"],
        "description": "Serum cholesterol measurement.",
        "minimum": 0,
        "maximum": 1000,
        "unit_options": ["mg/dL", "mmol/L"],
        "conversions": {
            "mmol/L": {"id": "mmol_L_to_mg_dL", "factor": 38.66976, "offset": 0},
        },
    },
    "fbs": {
        "label": "Fasting Blood Sugar",
        "alias_label": "Fasting Blood Sugar",
        "role": "feature",
        "units": "binary",
        "aliases": ["fbs", "fasting_blood_sugar", "fasting_blood_glucose"],
        "description": "Whether fasting blood sugar is greater than 120 mg/dL.",
        "allowed": [0, 1],
    },
    "restecg": {
        "label": "Resting ECG",
        "alias_label": "Resting ECG",
        "role": "feature",
        "units": "coded",
        "aliases": ["restecg", "resting_ecg", "resting_electrocardiogram", "ecg"],
        "description": "Resting electrocardiographic result.",
        "allowed": [0, 1, 2],
    },
    "thalach": {
        "label": "Maximum Heart Rate",
        "alias_label": "Maximum Heart Rate",
        "role": "feature",
        "units": "bpm",
        "aliases": ["thalach", "max_heart_rate", "maximum_heart_rate", "max_hr", "heart_rate"],
        "description": "Maximum heart rate achieved.",
        "minimum": 0,
        "maximum": 250,
    },
    "exang": {
        "label": "Exercise Angina",
        "alias_label": "Exercise Angina",
        "role": "feature",
        "units": "binary",
        "aliases": ["exang", "exercise_angina", "exercise_induced_angina"],
        "description": "Exercise-induced angina.",
        "allowed": [0, 1],
    },
    "oldpeak": {
        "label": "ST Depression",
        "alias_label": "ST Depression",
        "role": "feature",
        "units": "mm",
        "aliases": ["oldpeak", "st_depression", "st_depression_mm", "exercise_st_depression"],
        "description": "ST depression induced by exercise relative to rest.",
        "minimum": 0,
        "maximum": 10,
        "unit_options": ["mm", "cm"],
        "conversions": {
            "cm": {"id": "cm_to_mm", "factor": 10, "offset": 0},
        },
    },
    "slope": {
        "label": "ST Slope",
        "alias_label": "ST Slope",
        "role": "feature",
        "units": "coded",
        "aliases": ["slope", "st_slope", "peak_exercise_slope"],
        "description": "The slope of the peak exercise ST segment.",
        "allowed": [1, 2, 3],
    },
    "ca": {
        "label": "Major Vessels",
        "alias_label": "Major Vessels",
        "role": "feature",
        "units": "count",
        "aliases": ["ca", "vessels", "colored_vessels", "major_vessels"],
        "description": "Number of major vessels colored by fluoroscopy.",
        "allowed": [0, 1, 2, 3],
    },
    "thal": {
        "label": "Thalassemia",
        "alias_label": "Thalassemia",
        "role": "feature",
        "units": "coded",
        "aliases": ["thal", "thalassemia"],
        "description": "A coded thalassemia status.",
        "allowed": [3, 6, 7],
    },
    "target": {
        "label": "Diagnosis",
        "alias_label": "Diagnosis",
        "role": "classifier",
        "units": "binary",
        "aliases": [
            "target",
            "heart_disease_binary",
            "heart_disease",
            "diagnosis",
            "outcome",
            "label",
            "num",
        ],
        "description": "Binary heart disease diagnosis status.",
        "allowed": [0, 1],
        "source_aliases": {"num": {"normalization": "num_positive"}},
    },
}


def _field_payload(name, definition):
    payload = {
        "field": name,
        "label": definition["label"],
        "alias_label": definition.get("alias_label", definition["label"]),
        "role": definition["role"],
        "units": definition["units"],
        "aliases": list(definition.get("aliases", [])),
        "description": definition["description"],
        "unit_options": list(definition.get("unit_options", [definition["units"]])),
    }
    if "allowed" in definition:
        payload["allowed"] = list(definition["allowed"])
    if "minimum" in definition:
        payload["minimum"] = definition["minimum"]
    if "maximum" in definition:
        payload["maximum"] = definition["maximum"]
    payload["conversions"] = {
        unit: {"id": conversion["id"]}
        for unit, conversion in definition.get("conversions", {}).items()
    }
    return payload


def get_schema(schema_id=SCHEMA_ID):
    if schema_id != SCHEMA_ID:
        raise ValueError(f"Unknown schema: {schema_id}")
    return {
        "schema_id": SCHEMA_ID,
        "version": SCHEMA_VERSION,
        "label": SCHEMA_LABEL,
        "fields": [_field_payload(name, definition) for name, definition in FIELD_DEFINITIONS.items()],
    }


def schema_catalog():
    return [get_schema()]


def field_definition(field_name):
    try:
        return FIELD_DEFINITIONS[field_name]
    except KeyError as error:
        raise ValueError(f"Unknown schema field: {field_name}") from error


def feature_fields():
    return [name for name, definition in FIELD_DEFINITIONS.items() if definition["role"] == "feature"]


def target_field():
    return next(name for name, definition in FIELD_DEFINITIONS.items() if definition["role"] == "classifier")


def feature_rules():
    return {
        name: {
            key: set(value) if key == "allowed" else value
            for key, value in definition.items()
            if key in {"allowed", "minimum", "maximum"}
        }
        for name, definition in FIELD_DEFINITIONS.items()
        if definition["role"] == "feature"
    }


def data_dictionary():
    return [
        {
            "field": name,
            "label": definition["label"],
            "units": definition["units"],
            "domain": (
                ", ".join(str(value) for value in definition["allowed"])
                if "allowed" in definition
                else f"Numeric ({definition.get('minimum', '')}-{definition.get('maximum', '')})"
            ),
            "description": definition["description"],
            "role": definition["role"],
        }
        for name, definition in FIELD_DEFINITIONS.items()
    ]


def schema_field_names(schema_id=SCHEMA_ID):
    get_schema(schema_id)
    return list(FIELD_DEFINITIONS)


def copy_field_definitions():
    return deepcopy(FIELD_DEFINITIONS)
