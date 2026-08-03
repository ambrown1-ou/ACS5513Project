from difflib import SequenceMatcher
from pathlib import Path
import math
import re

import pandas as pd

from .schema import (
    FIELD_DEFINITIONS,
    SCHEMA_ID,
    field_definition,
    get_schema,
    schema_field_names,
)


MAX_SAMPLE_ROWS = 5
PROVENANCE_COLUMNS = {"source", "source_row"}


def normalize_name(value):
    value = str(value or "").strip().lower()
    return re.sub(r"[^a-z0-9]+", "_", value).strip("_")


def _read_csv(data_path):
    try:
        return pd.read_csv(Path(data_path), keep_default_na=True, na_values=["?"])
    except Exception as error:
        raise ValueError("The dataset could not be read as CSV.") from error


def _unit_lookup(field_name):
    definition = field_definition(field_name)
    units = list(definition.get("unit_options", [definition["units"]]))
    return {normalize_name(unit): unit for unit in units}


def normalize_unit(field_name, unit):
    lookup = _unit_lookup(field_name)
    requested = unit or field_definition(field_name)["units"]
    try:
        return lookup[normalize_name(requested)]
    except KeyError as error:
        supported = ", ".join(lookup.values())
        raise ValueError(f"Unsupported unit '{requested}' for {field_name}. Use: {supported}.") from error


def _candidate_score(source_name, alias):
    source_normalized = normalize_name(source_name)
    alias_normalized = normalize_name(alias)
    if not source_normalized or not alias_normalized:
        return 0.0, "none"
    if source_normalized == alias_normalized:
        return 1.0, "exact"
    if source_normalized.replace("_", "") == alias_normalized.replace("_", ""):
        return 0.98, "normalized"
    return SequenceMatcher(None, source_normalized, alias_normalized).ratio(), "similar"


def suggest_field_mappings(source_name, schema_id=SCHEMA_ID, limit=3):
    get_schema(schema_id)
    candidates = []
    for field_name, definition in FIELD_DEFINITIONS.items():
        best_score = 0.0
        best_reason = "similar"
        best_alias = field_name
        for alias in definition.get("aliases", [field_name]):
            score, reason = _candidate_score(source_name, alias)
            if score > best_score:
                best_score = score
                best_reason = reason
                best_alias = alias
        if best_score >= 0.45:
            candidates.append({
                "schema_field": field_name,
                "label": definition["label"],
                "role": definition["role"],
                "score": round(best_score, 4),
                "reason": best_reason,
                "matched_alias": best_alias,
                "units": definition["units"],
            })
    candidates.sort(key=lambda item: (-item["score"], item["schema_field"]))
    return candidates[:limit]


def analyze_dataset(data_path, schema_id=SCHEMA_ID):
    get_schema(schema_id)
    data = _read_csv(data_path)
    source_columns = []
    for column in data.columns:
        if normalize_name(column) in PROVENANCE_COLUMNS:
            continue
        values = data[column]
        source_columns.append({
            "name": str(column),
            "dtype": str(values.dtype),
            "rows": int(len(values)),
            "missing_count": int(values.isna().sum()),
            "unique_count": int(values.nunique(dropna=True)),
            "candidates": suggest_field_mappings(str(column), schema_id=schema_id),
        })
    return {
        "schema": get_schema(schema_id),
        "total_rows": int(len(data)),
        "source_columns": source_columns,
        "classifier_candidates": [
            column["name"]
            for column in source_columns
            if any(candidate["role"] == "classifier" for candidate in column["candidates"][:1])
        ],
    }


def _mapping_entries(mapping):
    if isinstance(mapping, dict):
        mapping = mapping.get("entries", mapping.get("mapping", []))
    if not isinstance(mapping, list):
        raise ValueError("Mapping entries must be an array.")
    return mapping


def validate_mapping(mapping, source_columns, schema_id=SCHEMA_ID):
    get_schema(schema_id)
    source_names = {str(column) for column in source_columns}
    entries = _mapping_entries(mapping)
    if not entries:
        raise ValueError("Map at least one source column before applying the mapping.")

    normalized = []
    seen_sources = set()
    seen_fields = set()
    classifier_count = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each mapping entry must be an object.")
        source_column = entry.get("source_column") or entry.get("source")
        schema_field = entry.get("schema_field") or entry.get("field")
        if not isinstance(source_column, str) or not source_column:
            raise ValueError("Each mapping entry needs a source_column.")
        if source_column not in source_names:
            raise ValueError(f"Unknown source column: {source_column}")
        if source_column in seen_sources:
            raise ValueError(f"Source column is mapped more than once: {source_column}")
        if not isinstance(schema_field, str) or schema_field not in FIELD_DEFINITIONS:
            raise ValueError(f"Unknown schema field: {schema_field}")
        if schema_field in seen_fields:
            raise ValueError(f"Schema field is mapped more than once: {schema_field}")

        definition = FIELD_DEFINITIONS[schema_field]
        source_unit = normalize_unit(schema_field, entry.get("source_unit") or definition["units"])
        seen_sources.add(source_column)
        seen_fields.add(schema_field)
        if definition["role"] == "classifier":
            classifier_count += 1
        normalized.append({
            "source_column": source_column,
            "schema_field": schema_field,
            "source_unit": source_unit,
            "canonical_unit": definition["units"],
            "role": definition["role"],
        })

    schema_names = schema_field_names(schema_id)
    selected_columns = [field_name for field_name in schema_names if field_name in seen_fields]
    feature_fields = [
        field_name
        for field_name in selected_columns
        if FIELD_DEFINITIONS[field_name]["role"] == "feature"
    ]
    target_fields = [
        field_name
        for field_name in selected_columns
        if FIELD_DEFINITIONS[field_name]["role"] == "classifier"
    ]
    return {
        "entries": normalized,
        "mapped_source_columns": sorted(seen_sources),
        "unmapped_source_columns": sorted(source_names - seen_sources),
        "mapped_schema_fields": selected_columns,
        "unmapped_schema_fields": [field_name for field_name in schema_names if field_name not in seen_fields],
        "selected_columns": selected_columns,
        "feature_fields": feature_fields,
        "target_field": target_fields[0] if len(target_fields) == 1 else None,
        "classifier_mapped": classifier_count == 1,
    }


def _missing_mask(values):
    return values.isna() | values.astype("string").str.strip().eq("")


def _sample_rows(mask, values, issue, limit=MAX_SAMPLE_ROWS):
    rows = []
    for index in values.index[mask][:limit]:
        value = values.loc[index]
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, float) and math.isnan(value):
            value = None
        rows.append({"row": int(index) + 1, "issue": issue, "value": value})
    return rows


def _field_issue_masks(field_name, source_column, values, converted_values):
    definition = FIELD_DEFINITIONS[field_name]
    missing = _missing_mask(values)
    source_numeric = pd.to_numeric(values, errors="coerce")
    conversion_errors = ~missing & source_numeric.isna()
    numeric = pd.to_numeric(converted_values, errors="coerce")
    valid_numeric = numeric.mask(missing | conversion_errors)

    if field_name == "target" and normalize_name(source_column) == "num":
        out_of_range = valid_numeric.notna() & ~valid_numeric.isin({0, 1, 2, 3, 4})
    else:
        out_of_range = pd.Series(False, index=values.index)
        allowed = definition.get("allowed")
        if allowed is not None:
            out_of_range |= valid_numeric.notna() & ~valid_numeric.isin(allowed)
        minimum = definition.get("minimum")
        maximum = definition.get("maximum")
        if minimum is not None:
            out_of_range |= valid_numeric.notna() & (valid_numeric < minimum)
        if maximum is not None:
            out_of_range |= valid_numeric.notna() & (valid_numeric > maximum)

    return {
        "missing": missing,
        "conversion_error": conversion_errors,
        "out_of_range": out_of_range,
    }


def _field_report(field_name, source_column, source_unit, values, converted_values):
    definition = FIELD_DEFINITIONS[field_name]
    masks = _field_issue_masks(field_name, source_column, values, converted_values)
    missing = masks["missing"]
    conversion_errors = masks["conversion_error"]
    out_of_range = masks["out_of_range"]

    samples = (
        _sample_rows(missing, values, "missing")
        + _sample_rows(conversion_errors, values, "conversion_error")
        + _sample_rows(out_of_range, values, "out_of_range")
    )[:MAX_SAMPLE_ROWS]
    valid_values = pd.to_numeric(converted_values, errors="coerce")
    valid_values = valid_values.mask(missing | conversion_errors | out_of_range).dropna()
    return {
        "source_column": source_column,
        "alias": definition.get("alias_label", definition["label"]),
        "aliases": list(definition.get("aliases", [])),
        "schema_field": field_name,
        "role": definition["role"],
        "source_unit": source_unit,
        "canonical_unit": definition["units"],
        "missing_count": int(missing.sum()),
        "conversion_error_count": int(conversion_errors.sum()),
        "out_of_range_count": int(out_of_range.sum()),
        "invalid_count": int((conversion_errors | out_of_range).sum()),
        "impute_possible": definition["role"] != "classifier" and not valid_values.empty,
        "sample_rows": samples,
    }


def _convert_values(field_name, source_column, source_unit, values):
    definition = FIELD_DEFINITIONS[field_name]
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    missing = _missing_mask(values)
    conversion_errors = ~missing & numeric.isna()
    converted = numeric.copy()
    conversion = definition.get("conversions", {}).get(source_unit)
    if conversion:
        valid = ~missing & ~conversion_errors
        converted.loc[valid] = numeric.loc[valid] * conversion["factor"] + conversion.get("offset", 0)

    if field_name == "target" and normalize_name(source_column) == "num":
        valid = ~missing & ~conversion_errors
        converted.loc[valid] = (numeric.loc[valid] > 0).astype(int)
    return converted, conversion["id"] if conversion else None


def apply_mapping(data_path, output_path, mapping, schema_id=SCHEMA_ID):
    data = _read_csv(data_path)
    validation = validate_mapping(mapping, list(data.columns), schema_id=schema_id)
    entries = validation["entries"]
    entry_by_source = {entry["source_column"]: entry for entry in entries}
    output = pd.DataFrame(index=data.index)
    if "source_row" not in data.columns:
        output["source_row"] = range(1, len(data) + 1)
    field_reports = []
    issue_masks = []
    conversion_ids = {}

    for source_column in data.columns:
        entry = entry_by_source.get(str(source_column))
        if entry:
            values, conversion_id = _convert_values(
                entry["schema_field"],
                entry["source_column"],
                entry["source_unit"],
                data[source_column],
            )
            output[entry["schema_field"]] = values
            if conversion_id:
                conversion_ids[entry["schema_field"]] = conversion_id
            field_reports.append(
                _field_report(
                    entry["schema_field"],
                    entry["source_column"],
                    entry["source_unit"],
                    data[source_column],
                    values,
                )
            )
            issue_masks.append(
                _field_issue_masks(
                    entry["schema_field"],
                    entry["source_column"],
                    data[source_column],
                    values,
                )
            )
        elif str(source_column) not in FIELD_DEFINITIONS:
            output[str(source_column)] = data[source_column]

    report = {
        "total_rows": int(len(data)),
        "total_rows_before_review": int(len(data)),
        "rows_with_missing_data": 0,
        "rows_with_out_of_range_values": 0,
        "rows_with_conversion_errors": 0,
        "missing_schema_fields": validation["unmapped_schema_fields"],
        "unmapped_source_columns": validation["unmapped_source_columns"],
        "fields": field_reports,
        "classifier": next((item for item in field_reports if item["role"] == "classifier"), None),
        "sample_rows": [],
    }
    missing_rows = set()
    range_rows = set()
    conversion_rows = set()
    for field_report in field_reports:
        for sample in field_report["sample_rows"]:
            if sample["issue"] == "missing":
                missing_rows.add(sample["row"])
            elif sample["issue"] == "out_of_range":
                range_rows.add(sample["row"])
            elif sample["issue"] == "conversion_error":
                conversion_rows.add(sample["row"])
        report["sample_rows"].extend(
            {**sample, "schema_field": field_report["schema_field"]}
            for sample in field_report["sample_rows"]
        )
    for masks in issue_masks:
        missing_rows.update(int(index) + 1 for index in data.index[masks["missing"]])
        range_rows.update(int(index) + 1 for index in data.index[masks["out_of_range"]])
        conversion_rows.update(int(index) + 1 for index in data.index[masks["conversion_error"]])
    report["rows_with_missing_data"] = len(missing_rows)
    report["rows_with_out_of_range_values"] = len(range_rows)
    report["rows_with_conversion_errors"] = len(conversion_rows)
    report["sample_rows"] = report["sample_rows"][:MAX_SAMPLE_ROWS]
    source_row_values = _source_row_values(data)
    report["source_row_ids"] = [int(value) for value in source_row_values.tolist()]
    report["selected_columns"] = list(validation["selected_columns"])
    report["feature_fields"] = list(validation["feature_fields"])
    report["target_field"] = validation["target_field"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)
    return {
        "path": output_path,
        "schema_id": schema_id,
        "schema_version": get_schema(schema_id)["version"],
        "source_columns": [str(column) for column in data.columns],
        "source_row_ids": [int(value) for value in source_row_values.tolist()],
        "mapping": validation,
        "report": report,
        "conversion_ids": conversion_ids,
    }


def _review_masks(data):
    masks = {
        "missing": pd.Series(False, index=data.index),
        "out_of_range": pd.Series(False, index=data.index),
        "conversion_error": pd.Series(False, index=data.index),
    }
    for field_name, definition in FIELD_DEFINITIONS.items():
        if field_name not in data.columns:
            continue
        values = data[field_name]
        missing = _missing_mask(values)
        numeric = pd.to_numeric(values, errors="coerce")
        conversion_error = ~missing & numeric.isna()
        out_of_range = pd.Series(False, index=data.index)
        valid_numeric = numeric.mask(missing | conversion_error)
        if field_name == "target":
            allowed = {0, 1}
        else:
            allowed = set(definition.get("allowed", []))
        if allowed:
            out_of_range |= valid_numeric.notna() & ~valid_numeric.isin(allowed)
        minimum = definition.get("minimum")
        maximum = definition.get("maximum")
        if minimum is not None:
            out_of_range |= valid_numeric.notna() & (valid_numeric < minimum)
        if maximum is not None:
            out_of_range |= valid_numeric.notna() & (valid_numeric > maximum)
        masks["missing"] |= missing
        masks["out_of_range"] |= out_of_range
        masks["conversion_error"] |= conversion_error
    return masks


FIELD_REVIEW_ACTIONS = {"replace_null", "impute", "drop_rows", "drop_column"}


def _validate_review_selection(review_metadata):
    if not review_metadata.get("target_field"):
        raise ValueError("Review decisions must leave exactly one target field.")
    if not review_metadata.get("feature_fields"):
        raise ValueError("Review decisions must leave at least one feature field.")


def _field_issue_masks_for_review(data, field_name):
    if field_name not in data.columns:
        return {
            "missing": pd.Series(True, index=data.index),
            "invalid": pd.Series(False, index=data.index),
        }
    values = data[field_name]
    masks = _field_issue_masks(field_name, field_name, values, values)
    return {
        "missing": masks["missing"],
        "invalid": masks["conversion_error"] | masks["out_of_range"],
    }


def _field_imputation_value(field_name, values, issue_mask):
    definition = FIELD_DEFINITIONS[field_name]
    if definition["role"] == "classifier":
        return None
    numeric = pd.to_numeric(values, errors="coerce")
    valid_values = numeric[~issue_mask].dropna()
    allowed = definition.get("allowed")
    if allowed:
        valid_values = valid_values[valid_values.isin(allowed)]
        if valid_values.empty:
            return None
        return valid_values.mode().sort_values().iloc[0]
    if valid_values.empty:
        return None
    return valid_values.median()


def _apply_field_decisions(data, report, field_decisions):
    if not isinstance(field_decisions, dict):
        raise ValueError("Field decisions must be an object.")

    issue_fields = {
        field["schema_field"]
        for field in report.get("fields", [])
        if field.get("missing_count", 0) or field.get("invalid_count", 0)
    }
    unknown_fields = set(field_decisions) - issue_fields
    if unknown_fields:
        raise ValueError("Decisions were provided for fields without reported issues: " + ", ".join(sorted(unknown_fields)))
    missing_decisions = issue_fields - set(field_decisions)
    if missing_decisions:
        raise ValueError("Choose a handling action for each reported field: " + ", ".join(sorted(missing_decisions)))

    drop_mask = pd.Series(False, index=data.index)
    resolved = {}
    for field_name in sorted(issue_fields):
        action = field_decisions[field_name]
        if action not in FIELD_REVIEW_ACTIONS:
            choices = ", ".join(sorted(FIELD_REVIEW_ACTIONS))
            raise ValueError(f"Invalid action '{action}' for {field_name}. Use: {choices}.")
        masks = _field_issue_masks_for_review(data, field_name)
        issue_mask = masks["missing"] | masks["invalid"]
        if action == "drop_rows":
            drop_mask |= issue_mask
        elif action == "drop_column":
            data = data.drop(columns=[field_name])
        elif action == "replace_null":
            if field_name not in data.columns:
                data[field_name] = float("nan")
            else:
                data.loc[issue_mask, field_name] = float("nan")
        else:
            if field_name not in data.columns:
                raise ValueError(f"Imputation is not possible for missing schema field {field_name}.")
            fill_value = _field_imputation_value(field_name, data[field_name], issue_mask)
            if fill_value is None:
                raise ValueError(f"Imputation is not possible for {field_name}.")
            data.loc[issue_mask, field_name] = fill_value
        resolved[field_name] = action
    return data, drop_mask, resolved


def _source_row_values(data):
    if "source_row" not in data.columns:
        return pd.Series(range(1, len(data) + 1), index=data.index, dtype="int64")

    values = pd.to_numeric(data["source_row"], errors="coerce")
    if values.isna().any() or (values % 1 != 0).any() or (values < 1).any() or values.duplicated().any():
        raise ValueError("The source_row column must contain unique positive integer row IDs.")
    return values.astype("int64")


def _review_metadata(mapping_result, source_row_values, reviewed, drop_mask):
    mapping = mapping_result.get("mapping", {})
    selected_columns = [
        field_name
        for field_name in list(mapping.get("selected_columns") or mapping.get("mapped_schema_fields", []))
        if field_name in reviewed.columns
    ]
    feature_fields = [
        field_name
        for field_name in list(mapping.get("feature_fields", []))
        if field_name in reviewed.columns
    ]
    target_field = mapping.get("target_field")
    if target_field not in reviewed.columns:
        target_field = None
    source_row_ids = [int(value) for value in source_row_values.tolist()]
    dropped_row_ids = [
        int(value)
        for value, dropped in zip(source_row_values.tolist(), drop_mask.tolist())
        if dropped
    ]
    selected_row_ids = [
        int(value)
        for value, dropped in zip(source_row_values.tolist(), drop_mask.tolist())
        if not dropped
    ]
    return {
        "source_row_ids": source_row_ids,
        "selected_row_ids": selected_row_ids,
        "dropped_row_ids": dropped_row_ids,
        "selected_columns": selected_columns,
        "feature_fields": feature_fields,
        "target_field": target_field,
        "source_columns": list(mapping_result.get("source_columns", [])),
        "rows_before": int(len(source_row_values)),
        "total_rows_before_review": int(len(source_row_values)),
        "rows_after": int(len(reviewed)),
        "dropped_rows": int(drop_mask.sum()),
    }


def review_mapping(data_path, output_path, mapping_result, decisions=None, field_decisions=None):
    """Apply explicit review decisions to a mapped canonical CSV."""
    data = _read_csv(data_path)
    if not isinstance(mapping_result, dict) or not isinstance(mapping_result.get("report"), dict):
        raise ValueError("A mapping report is required before review.")
    report = mapping_result["report"]
    source_row_values = _source_row_values(data)
    if "source_row" not in data.columns:
        data.insert(0, "source_row", source_row_values)
    else:
        data["source_row"] = source_row_values
    if field_decisions is not None:
        data, drop_mask, resolved = _apply_field_decisions(
            data,
            mapping_result["report"],
            field_decisions,
        )
        reviewed = data.loc[~drop_mask].copy()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        reviewed.to_csv(output_path, index=False)
        review_metadata = _review_metadata(
            mapping_result,
            source_row_values,
            reviewed,
            drop_mask,
        )
        _validate_review_selection(review_metadata)
        return {
            "path": output_path,
            "decisions": resolved,
            "report": report,
            **review_metadata,
        }

    if not isinstance(decisions, dict):
        raise ValueError("Review decisions must be an object.")

    resolved = {
        "missing_rows": decisions.get("missing_rows", "keep"),
        "missing_columns": decisions.get("missing_columns", "allow"),
        "out_of_range": decisions.get("out_of_range", "keep"),
        "conversion_errors": decisions.get("conversion_errors", "reject"),
        "unmapped_columns": decisions.get("unmapped_columns", "keep"),
    }
    allowed_decisions = {
        "missing_rows": {"drop", "keep"},
        "missing_columns": {"allow", "reject"},
        "out_of_range": {"drop", "keep", "reject"},
        "conversion_errors": {"drop", "reject"},
        "unmapped_columns": {"keep", "drop"},
    }
    for decision_name, value in resolved.items():
        if value not in allowed_decisions[decision_name]:
            choices = ", ".join(sorted(allowed_decisions[decision_name]))
            raise ValueError(f"Invalid {decision_name} decision '{value}'. Use: {choices}.")

    missing_schema_fields = list(report.get("missing_schema_fields", []))
    if missing_schema_fields and resolved["missing_columns"] == "reject":
        raise ValueError(
            "Review rejected because schema fields are missing: "
            + ", ".join(missing_schema_fields)
        )

    masks = _review_masks(data)
    if masks["out_of_range"].any() and resolved["out_of_range"] == "reject":
        raise ValueError("Review rejected because values are outside the schema domain or range.")
    if masks["conversion_error"].any() and resolved["conversion_errors"] == "reject":
        raise ValueError("Review rejected because values could not be converted to numbers.")

    drop_mask = pd.Series(False, index=data.index)
    if resolved["missing_rows"] == "drop":
        drop_mask |= masks["missing"]
    if resolved["out_of_range"] == "drop":
        drop_mask |= masks["out_of_range"]
    if resolved["conversion_errors"] == "drop":
        drop_mask |= masks["conversion_error"]
    if resolved["unmapped_columns"] == "drop":
        mapped_fields = set(mapping_result.get("mapping", {}).get("mapped_schema_fields", []))
        keep_columns = [
            column for column in data.columns
            if column in mapped_fields or column in {"source", "source_row"}
        ]
        data = data[keep_columns]

    reviewed = data.loc[~drop_mask].copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reviewed.to_csv(output_path, index=False)
    review_metadata = _review_metadata(
        mapping_result,
        source_row_values,
        reviewed,
        drop_mask,
    )
    _validate_review_selection(review_metadata)
    return {
        "path": output_path,
        "decisions": resolved,
        "report": report,
        **review_metadata,
    }


def identity_mapping(source_columns, schema_id=SCHEMA_ID):
    get_schema(schema_id)
    normalized_columns = {normalize_name(column): str(column) for column in source_columns}
    entries = []
    for field_name in schema_field_names(schema_id):
        definition = FIELD_DEFINITIONS[field_name]
        aliases = definition.get("aliases", [field_name])
        selected = next((normalized_columns[normalize_name(alias)] for alias in aliases if normalize_name(alias) in normalized_columns), None)
        if selected is not None and not any(entry["schema_field"] == field_name for entry in entries):
            entries.append({
                "source_column": selected,
                "schema_field": field_name,
                "source_unit": definition["units"],
            })
    return entries
