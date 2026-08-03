(function () {
  const app = document.querySelector('[data-intake-app]');
  if (!app) return;

  const apiRequest = async (path, options) => {
    const response = await fetch(path, options);
    let payload = {};
    try {
      payload = await response.json();
    } catch (_) {
      payload = {};
    }
    if (!response.ok) {
      const error = new Error(payload.error || response.statusText || 'API request failed.');
      error.payload = payload;
      error.status = response.status;
      throw error;
    }
    return payload;
  };

  const state = {
    datasets: [],
    intake: null,
    schema: null,
    currentDataset: null,
    analysis: null,
  };

  const uploadForm = app.querySelector('[data-intake-upload]');
  const uploadSchema = app.querySelector('[data-upload-schema]');
  const uploadModes = app.querySelector('[data-upload-modes]');
  const validationModeWarning = app.querySelector('[data-validation-mode-warning]');
  const uploadStatus = app.querySelector('[data-upload-status]');
  const intakeStatus = app.querySelector('[data-intake-status]');
  const mappingStage = app.querySelector('#field-mapping-stage');
  const mappingRows = app.querySelector('[data-mapping-rows]');
  const mappingButton = app.querySelector('[data-apply-mapping]');
  const mappingMessage = app.querySelector('[data-mapping-status-message]');
  const reviewStage = app.querySelector('#review-stage');
  const reviewForm = app.querySelector('[data-review-form]');
  const reviewReport = app.querySelector('[data-validation-report]');
  const reviewMessage = app.querySelector('[data-review-status-message]');
  const readyStage = app.querySelector('#ready-stage');
  const readySummary = app.querySelector('[data-ready-summary]');
  const readyStatus = app.querySelector('[data-ready-status]');
  const datasetList = app.querySelector('[data-dataset-list]');

  const setStatus = (element, message, isError) => {
    if (!element) return;
    element.hidden = !message;
    element.textContent = message || '';
    element.dataset.status = isError ? 'error' : 'info';
  };

  const createOption = (value, label, disabled) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    option.disabled = Boolean(disabled);
    return option;
  };

  const createElement = (tagName, className, text) => {
    const element = document.createElement(tagName);
    if (className) element.className = className;
    if (text !== undefined) element.textContent = text;
    return element;
  };

  const schemaFields = () => state.schema && Array.isArray(state.schema.fields) ? state.schema.fields : [];

  const schemaField = (fieldName) => schemaFields().find((field) => field.field === fieldName);

  const datasetLabel = (dataset) => dataset && dataset.label ? dataset.label : dataset && dataset.dataset_key;

  const selectedColumnsLabel = (dataset) => {
    const columns = dataset && Array.isArray(dataset.selected_columns) ? dataset.selected_columns : [];
    return columns.length ? `Selected columns: ${columns.join(', ')}` : 'Columns not selected';
  };

  const statusLabel = (dataset) => {
    if (!dataset) return 'Unavailable';
    const labels = {
      mapping: 'Mapping required',
      review: 'Review required',
      ready: 'Approved',
      trusted: 'Trusted input',
      legacy: 'Existing dataset',
    };
    return labels[dataset.intake_status] || dataset.intake_status || 'Unknown';
  };

  const trainingLabel = (dataset) => {
    if (!dataset) return 'Unavailable';
    if (dataset.training_available) return 'Available for training';
    return 'Complete intake first';
  };

  const setStepState = (status) => {
    const activeStep = status === 'mapping' ? 'mapping' : status === 'review' ? 'review' : 'ready';
    const completed = status === 'review'
      ? ['mapping']
      : ['mapping', 'review'].includes(status)
        ? ['mapping', 'review']
        : status === 'ready' || status === 'trusted' || status === 'legacy'
          ? ['mapping', 'review']
          : [];

    app.querySelectorAll('[data-intake-step]').forEach((step) => {
      const stepName = step.dataset.intakeStep;
      step.classList.toggle('is-active', stepName === activeStep);
      step.classList.toggle('is-complete', completed.includes(stepName));
    });
  };

  const publishCatalog = () => {
    const detail = {
      datasets: state.datasets,
    };
    window.__intakeCatalog = detail;
    window.dispatchEvent(new CustomEvent('intake:catalog-ready', { detail }));
  };

  const updateDataset = (dataset) => {
    if (!dataset || !dataset.dataset_key) return;
    const existingIndex = state.datasets.findIndex((item) => item.dataset_key === dataset.dataset_key);
    if (existingIndex === -1) state.datasets.push(dataset);
    else state.datasets[existingIndex] = dataset;
    state.currentDataset = dataset;
    renderDatasetList();
    publishCatalog();
  };

  const updateValidationModeWarning = (modeValue) => {
    if (!validationModeWarning) return;
    const messages = {
      NO_TEST: 'Warning: No Test skips mapping and validation checks. If the data is not already properly prepared for the selected schema, training may not be possible.',
    };
    validationModeWarning.textContent = messages[modeValue] || '';
    validationModeWarning.hidden = !messages[modeValue];
  };

  const renderUploadModes = () => {
    if (!uploadModes) return;
    const modes = state.intake && state.intake.validation_modes ? state.intake.validation_modes : [];
    const selectedValue = uploadModes.value || 'NORMAL';
    if (modes.length) {
      uploadModes.replaceChildren();
      modes.forEach((mode, index) => {
        uploadModes.appendChild(createOption(mode.value, mode.label, false));
      });
      uploadModes.value = modes.some((mode) => mode.value === selectedValue)
        ? selectedValue
        : modes[0].value;
    }
    updateValidationModeWarning(uploadModes.value);
  };

  const renderSchemaOptions = () => {
    if (!uploadSchema) return;
    uploadSchema.replaceChildren();
    state.schema = state.schema || (state.intake && state.intake.schemas ? state.intake.schemas[0] : null);
    const schemas = state.schemas || (state.schema ? [state.schema] : []);
    schemas.forEach((schema) => {
      const label = schema.schema_id === 'cleveland_v1'
        ? 'ClevelandV1'
        : `${schema.label} v${schema.version}`;
      uploadSchema.appendChild(createOption(schema.schema_id, label));
    });
    if (state.schema) uploadSchema.value = state.schema.schema_id;
  };

  const renderDatasetList = () => {
    if (!datasetList) return;
    datasetList.replaceChildren();
    if (!state.datasets.length) {
      const row = createElement('tr');
      const cell = createElement('td', 'field-help', 'No datasets available.');
      cell.colSpan = 4;
      row.appendChild(cell);
      datasetList.appendChild(row);
      return;
    }

    state.datasets.forEach((dataset) => {
      const row = document.createElement('tr');
      const datasetCell = createElement('td');
      datasetCell.append(
        createElement('strong', '', datasetLabel(dataset)),
        createElement('small', 'dataset-source', dataset.source || 'Unknown source'),
        createElement('small', 'dataset-selected-columns', selectedColumnsLabel(dataset)),
      );
      const statusCell = createElement('td', '', statusLabel(dataset));
      const trainingCell = createElement('td', '', trainingLabel(dataset));
      const actionCell = createElement('td', 'dataset-actions');
      const selectButton = createElement('button', 'dataset-select-button', 'Select');
      selectButton.type = 'button';
      selectButton.dataset.selectDataset = dataset.dataset_key;
      actionCell.appendChild(selectButton);
      if (dataset.deletable) {
        const deleteButton = createElement('button', 'btn-danger', 'Delete');
        deleteButton.type = 'button';
        deleteButton.dataset.deleteDataset = dataset.dataset_key;
        actionCell.appendChild(deleteButton);
      } else {
        actionCell.appendChild(createElement('span', 'dataset-readonly', 'Bundled'));
      }
      row.append(datasetCell, statusCell, trainingCell, actionCell);
      datasetList.appendChild(row);
    });
  };

  const updateMappingSourceOptions = () => {
    if (!mappingRows) return;
    const selects = Array.from(mappingRows.querySelectorAll('[data-mapping-source]'));
    const selectedValues = selects.map((select) => select.value).filter(Boolean);
    selects.forEach((select) => {
      const currentValue = select.value;
      Array.from(select.options).forEach((option) => {
        option.disabled = Boolean(option.value && option.value !== currentValue && selectedValues.includes(option.value));
      });
    });
  };

  const renderUnitOptions = (row, fieldName, selectedUnit) => {
    const unitSelect = row.querySelector('[data-mapping-unit]');
    if (!unitSelect) return;
    unitSelect.replaceChildren();
    const definition = schemaField(fieldName);
    const units = definition && definition.unit_options ? definition.unit_options : [];
    units.forEach((unit, index) => {
      const option = createOption(unit, unit);
      if (unit === selectedUnit || (!selectedUnit && index === 0)) option.selected = true;
      unitSelect.appendChild(option);
    });
    const sourceSelect = row.querySelector('[data-mapping-source]');
    unitSelect.disabled = !sourceSelect || !sourceSelect.value || units.length <= 1;
  };

  const renderMappingOperation = (row, fieldName, sourceColumn, sourceUnit) => {
    const operation = row.querySelector('[data-mapping-operation]');
    if (!operation) return;
    const status = operation.querySelector('[data-mapping-operation-status]');
    if (!status) return;
    if (!sourceColumn) {
      status.textContent = 'No operation';
      operation.dataset.operation = 'none';
      return;
    }
    const definition = schemaField(fieldName);
    if (!definition) {
      status.textContent = 'Select a schema field';
      operation.dataset.operation = 'none';
      return;
    }
    const conversion = definition.conversions && definition.conversions[sourceUnit];
    if (conversion) {
      status.textContent = `Convert ${sourceUnit} to ${definition.units}`;
      operation.dataset.operation = 'convert';
      return;
    }
    status.textContent = `Use ${sourceUnit || definition.units}`;
    operation.dataset.operation = 'none';
  };

  const renderMappingRows = (analysis) => {
    if (!mappingRows) return;
    mappingRows.replaceChildren();
    const sourceColumns = analysis.source_columns || [];
    const fields = schemaFields();
    if (!fields.length) {
      mappingRows.appendChild(createElement('p', 'field-help', 'No schema fields are available.'));
      if (mappingButton) mappingButton.disabled = true;
      return;
    }
    const header = createElement('div', 'mapping-header');
    ['Schema field', 'Source column', 'Operation'].forEach((label) => {
      header.appendChild(createElement('span', '', label));
    });
    mappingRows.appendChild(header);
    const usedSources = new Set();
    fields.forEach((field) => {
      const row = createElement('div', 'mapping-row');
      row.dataset.schemaField = field.field;
      const schemaCell = createElement('div', 'mapping-schema-field');
      schemaCell.append(
        createElement('strong', '', field.label),
        createElement('small', '', `${field.field} - ${field.role}`),
      );

      const sourceLabel = createElement('label');
      sourceLabel.appendChild(createElement('span', 'mapping-label', 'Source column'));
      const sourceSelect = document.createElement('select');
      sourceSelect.dataset.mappingSource = '';
      sourceSelect.appendChild(createOption('', 'Leave unmapped'));
      const directColumn = sourceColumns.find((column) => {
        if (usedSources.has(column.name)) return false;
        return (column.candidates || []).some((candidate) => (
          candidate.schema_field === field.field
          && ['exact', 'normalized'].includes(candidate.reason)
        ));
      });
      sourceColumns.forEach((column) => {
        const option = createOption(column.name, column.name);
        if (directColumn && column.name === directColumn.name) option.selected = true;
        sourceSelect.appendChild(option);
      });
      if (directColumn) usedSources.add(directColumn.name);
      sourceLabel.appendChild(sourceSelect);

      const operation = createElement('div', 'mapping-operation');
      operation.dataset.mappingOperation = '';
      const unitLabel = createElement('label');
      unitLabel.appendChild(createElement('span', 'mapping-label', 'Source unit'));
      const unitSelect = document.createElement('select');
      unitSelect.dataset.mappingUnit = '';
      unitLabel.appendChild(unitSelect);
      operation.append(
        unitLabel,
        createElement('span', 'mapping-operation-status', 'No operation'),
      );
      operation.querySelector('.mapping-operation-status').dataset.mappingOperationStatus = '';
      row.append(schemaCell, sourceLabel, operation);
      mappingRows.appendChild(row);
      renderUnitOptions(row, field.field, '');
      renderMappingOperation(row, field.field, sourceSelect.value, unitSelect.value);

      sourceSelect.addEventListener('change', () => {
        renderUnitOptions(row, field.field, '');
        renderMappingOperation(row, field.field, sourceSelect.value, unitSelect.value);
        updateMappingSourceOptions();
      });
      unitSelect.addEventListener('change', () => {
        renderMappingOperation(row, field.field, sourceSelect.value, unitSelect.value);
      });
    });
    updateMappingSourceOptions();
    if (mappingButton) mappingButton.disabled = !sourceColumns.length;
  };

  const collectMapping = () => Array.from(mappingRows.querySelectorAll('.mapping-row')).map((row) => {
    const sourceSelect = row.querySelector('[data-mapping-source]');
    const unitSelect = row.querySelector('[data-mapping-unit]');
    return {
      source_column: sourceSelect ? sourceSelect.value : '',
      schema_field: row.dataset.schemaField,
      source_unit: unitSelect ? unitSelect.value : '',
    };
  }).filter((entry) => entry.source_column);

  const renderValidationReport = (report) => {
    if (!reviewReport) return;
    reviewReport.replaceChildren();
    const fields = (report.fields || [])
      .filter((field) => field.missing_count > 0 || field.invalid_count > 0)
      .map((field) => ({ ...field, missing_display: String(field.missing_count || 0) }));
    const totalRowsBeforeReview = Number(report.total_rows_before_review ?? report.total_rows);
    if (Number.isFinite(totalRowsBeforeReview)) {
      reviewReport.appendChild(createElement('p', 'field-help', `Total rows before review: ${totalRowsBeforeReview}`));
    }
    if (!fields.length) {
      const message = report.missing_schema_fields && report.missing_schema_fields.length
        ? `Unmapped fields remain excluded: ${report.missing_schema_fields.join(', ')}.`
        : 'No missing or invalid field values were found.';
      reviewReport.appendChild(createElement('p', 'field-help', message));
      return;
    }

    const table = document.createElement('table');
    table.className = 'validation-field-table';
    const head = document.createElement('thead');
    const headRow = document.createElement('tr');
    ['Schema field', 'Field alias', 'Missing', 'Invalid', 'Action'].forEach((label) => headRow.appendChild(createElement('th', '', label)));
    head.appendChild(headRow);
    const body = document.createElement('tbody');
    const actions = state.intake && state.intake.field_issue_actions || [
      { value: 'replace_null', label: 'Replace with NULL (NaN)' },
      { value: 'impute', label: 'Impute' },
      { value: 'drop_rows', label: 'Drop affected rows' },
      { value: 'drop_column', label: 'Drop column' },
    ];
    fields.forEach((field) => {
      const row = document.createElement('tr');
      const imputePossible = field.impute_possible !== undefined
        ? field.impute_possible
        : field.role !== 'classifier' && !field.missing_schema_field;
      const actionSelect = document.createElement('select');
      actionSelect.required = true;
      actionSelect.dataset.fieldDecision = field.schema_field;
      actions.forEach((action) => {
        if (action.value === 'impute' && !imputePossible) return;
        const option = createOption(action.value, action.label);
        option.selected = action.value === 'replace_null';
        actionSelect.appendChild(option);
      });
      row.append(
        createElement('td', '', field.schema_field),
        createElement('td', '', field.alias || field.source_column || '-'),
        createElement('td', '', field.missing_display || String(field.missing_count || 0)),
        createElement('td', '', String(field.invalid_count || 0)),
        createElement('td', 'validation-action-cell'),
      );
      row.lastElementChild.appendChild(actionSelect);
      body.appendChild(row);
    });
    table.append(head, body);
    reviewReport.appendChild(table);
  };

  const showReadyStage = (dataset) => {
    const status = dataset && dataset.intake_status || 'ready';
    setStepState(status);
    if (mappingStage) mappingStage.hidden = true;
    if (reviewStage) reviewStage.hidden = true;
    if (readyStage) readyStage.hidden = false;
    if (readyStatus) readyStatus.textContent = statusLabel(dataset);
    if (readySummary) {
      const finalRowCount = Number.isFinite(Number(dataset && dataset.final_row_count))
        ? ` Final row count: ${Number(dataset.final_row_count)}.`
        : '';
      const selectedColumns = selectedColumnsLabel(dataset);
      const droppedRows = Array.isArray(dataset && dataset.dropped_row_ids) && dataset.dropped_row_ids.length
        ? ` Dropped source rows: ${dataset.dropped_row_ids.join(', ')}.`
        : '';
      const message = status === 'ready'
        ? `${datasetLabel(dataset)} is approved. ${selectedColumns}.${finalRowCount}${droppedRows} It is now available to select for training.`
        : `${datasetLabel(dataset)} is treated as ${statusLabel(dataset).toLowerCase()}. ${selectedColumns} It can proceed to training.${finalRowCount}${droppedRows}`;
      readySummary.textContent = message;
    }
  };

  const showReviewStage = (dataset, report) => {
    setStepState('review');
    if (mappingStage) mappingStage.hidden = true;
    if (reviewStage) reviewStage.hidden = false;
    if (readyStage) readyStage.hidden = true;
    if (reviewReport) renderValidationReport(report || {});
    const status = app.querySelector('[data-review-status]');
    if (status) status.textContent = 'Decision required';
  };

  const loadAnalysis = async (dataset) => {
    if (!dataset) return;
    setStepState('mapping');
    if (mappingStage) mappingStage.hidden = false;
    if (reviewStage) reviewStage.hidden = true;
    if (readyStage) readyStage.hidden = true;
    const status = app.querySelector('[data-mapping-status]');
    if (status) status.textContent = 'Loading source columns';
    if (mappingRows) mappingRows.replaceChildren(createElement('p', 'field-help', 'Analyzing source columns...'));
    try {
      const analysis = await apiRequest(`/api/datasets/${encodeURIComponent(dataset.dataset_key)}/field-analysis`);
      if (!state.currentDataset || state.currentDataset.dataset_key !== dataset.dataset_key) return;
      state.analysis = analysis;
      state.schema = analysis.schema || state.schema;
      renderMappingRows(analysis);
      if (status) status.textContent = 'Map schema fields';
    } catch (error) {
      if (mappingRows) mappingRows.replaceChildren(createElement('p', 'field-help', error.message));
      if (mappingButton) mappingButton.disabled = true;
      if (status) status.textContent = 'Analysis failed';
    }
  };

  const activateDataset = async (datasetKey) => {
    const dataset = state.datasets.find((item) => item.dataset_key === datasetKey);
    state.currentDataset = dataset || null;
    state.analysis = null;
    if (!dataset) {
      if (mappingStage) mappingStage.hidden = true;
      if (reviewStage) reviewStage.hidden = true;
      if (readyStage) readyStage.hidden = true;
      setStepState('mapping');
      return;
    }
    const report = dataset.intake && dataset.intake.validation_report;
    if (dataset.intake_status === 'mapping') await loadAnalysis(dataset);
    else if (dataset.intake_status === 'review') showReviewStage(dataset, report);
    else showReadyStage(dataset);
  };

  const refreshDatasets = async (preferredKey) => {
    state.datasets = await apiRequest('/api/datasets');
    renderDatasetList();
    publishCatalog();
    const selectedKey = preferredKey || state.currentDataset && state.currentDataset.dataset_key;
    if (selectedKey && state.datasets.some((dataset) => dataset.dataset_key === selectedKey)) {
      await activateDataset(selectedKey);
    } else if (state.datasets.length) {
      await activateDataset(state.datasets[0].dataset_key);
    } else {
      await activateDataset('');
    }
  };

  const submitMapping = async () => {
    if (!state.currentDataset || !mappingRows) return;
    const entries = collectMapping();
    if (!entries.length) {
      setStatus(mappingMessage, 'Map at least one source column.', true);
      return;
    }
    mappingButton.disabled = true;
    setStatus(mappingMessage, 'Applying field mapping...', false);
    try {
      const result = await apiRequest(`/api/datasets/${encodeURIComponent(state.currentDataset.dataset_key)}/field-mapping`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ schema_id: state.schema.schema_id, mapping: entries }),
      });
      updateDataset(result.dataset);
      state.currentDataset = result.dataset;
      showReviewStage(result.dataset, result.report);
      setStatus(mappingMessage, '', false);
    } catch (error) {
      setStatus(mappingMessage, error.message, true);
      mappingButton.disabled = false;
    }
  };

  const submitReview = async (event) => {
    event.preventDefault();
    if (!state.currentDataset || !reviewForm || !reviewReport) return;
    if (!reviewForm.reportValidity()) return;
    const fieldDecisions = {};
    reviewReport.querySelectorAll('[data-field-decision]').forEach((select) => {
      fieldDecisions[select.dataset.fieldDecision] = select.value;
    });
    const button = reviewForm.querySelector('[data-submit-review]');
    if (button) button.disabled = true;
    setStatus(reviewMessage, 'Applying review decisions...', false);
    try {
      const result = await apiRequest(`/api/datasets/${encodeURIComponent(state.currentDataset.dataset_key)}/review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ field_decisions: fieldDecisions }),
      });
      updateDataset(result.dataset);
      state.currentDataset = result.dataset;
      showReadyStage(result.dataset);
      setStatus(reviewMessage, '', false);
    } catch (error) {
      setStatus(reviewMessage, error.message, true);
      if (button) button.disabled = false;
    }
  };

  const submitUpload = async (event) => {
    event.preventDefault();
    if (!uploadForm.reportValidity()) return;
    const submitButton = uploadForm.querySelector('button[type="submit"]');
    if (submitButton) submitButton.disabled = true;
    setStatus(uploadStatus, 'Uploading dataset...', false);
    try {
      const result = await apiRequest('/api/datasets/upload', {
        method: 'POST',
        body: new FormData(uploadForm),
      });
      if (result.dataset) updateDataset(result.dataset);
      uploadForm.reset();
      if (uploadSchema && state.schema) uploadSchema.value = state.schema.schema_id;
      setStatus(uploadStatus, result.message || 'Dataset uploaded.', false);
      setStatus(intakeStatus, '', false);
      await refreshDatasets(result.dataset_key);
    } catch (error) {
      setStatus(uploadStatus, error.message, true);
    } finally {
      if (submitButton) submitButton.disabled = false;
    }
  };

  const selectDatasetFromTable = (datasetKey) => {
    activateDataset(datasetKey);
    const visibleStage = [mappingStage, reviewStage, readyStage].find((stage) => stage && !stage.hidden);
    if (visibleStage) visibleStage.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const deleteDataset = async (datasetKey) => {
    const dataset = state.datasets.find((item) => item.dataset_key === datasetKey);
    if (!dataset || !window.confirm(`Delete ${datasetLabel(dataset)}?`)) return;
    try {
      await apiRequest(`/api/datasets/${encodeURIComponent(datasetKey)}`, { method: 'DELETE' });
      await refreshDatasets();
      setStatus(intakeStatus, 'Dataset deleted.', false);
    } catch (error) {
      setStatus(intakeStatus, error.message, true);
    }
  };

  const initialize = async () => {
    try {
      const [intake, schemas] = await Promise.all([
        apiRequest('/api/intake'),
        apiRequest('/api/schemas'),
      ]);
      state.intake = intake;
      state.schemas = Array.isArray(schemas) ? schemas : [];
      state.schema = state.schemas[0] || null;
      renderUploadModes();
      renderSchemaOptions();
    } catch (error) {
      setStatus(intakeStatus, `Intake configuration could not be loaded: ${error.message}`, true);
      return;
    }

    try {
      const datasets = await apiRequest('/api/datasets');
      state.datasets = Array.isArray(datasets) ? datasets : [];
      renderDatasetList();
      publishCatalog();
      const initialKey = new URLSearchParams(window.location.search).get('dataset');
      await activateDataset(initialKey && datasets.some((dataset) => dataset.dataset_key === initialKey)
        ? initialKey
        : datasets[0] && datasets[0].dataset_key);
    } catch (error) {
      setStatus(intakeStatus, `Dataset list could not be loaded: ${error.message}`, true);
    }
  };

  uploadForm.addEventListener('submit', submitUpload);
  if (uploadModes) {
    uploadModes.addEventListener('change', () => updateValidationModeWarning(uploadModes.value));
    updateValidationModeWarning(uploadModes.value);
  }
  if (mappingButton) mappingButton.addEventListener('click', submitMapping);
  if (reviewForm) reviewForm.addEventListener('submit', submitReview);
  datasetList.addEventListener('click', (event) => {
    const selectButton = event.target.closest('[data-select-dataset]');
    if (selectButton) selectDatasetFromTable(selectButton.dataset.selectDataset);
    const deleteButton = event.target.closest('[data-delete-dataset]');
    if (deleteButton) deleteDataset(deleteButton.dataset.deleteDataset);
  });
  initialize();
})();
