(function () {
  const apiRequest = async (path, options) => {
    const response = await fetch(path, options);
    let payload = {};
    try {
      payload = await response.json();
    } catch (_) {
      payload = {};
    }
    if (!response.ok) {
      throw new Error(payload.error || response.statusText || 'API request failed.');
    }
    return payload;
  };

  const setStatus = (element, message, isError) => {
    if (!element) return;
    element.hidden = !message;
    element.textContent = message || '';
    element.dataset.status = isError ? 'error' : 'info';
  };

  const addOption = (select, value, label, disabled) => {
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    option.disabled = Boolean(disabled);
    option.selected = Boolean(disabled);
    select.appendChild(option);
    return option;
  };

  const readJsonElement = (element) => {
    if (!element) return {};
    try {
      return JSON.parse(element.textContent || '{}');
    } catch (_) {
      return {};
    }
  };

  const buildParameterControl = (parameter, values) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'parameter-field';

    const label = document.createElement('label');
    const controlId = `training-parameter-${parameter.name}`;
    label.htmlFor = controlId;
    label.textContent = parameter.label || parameter.name;

    let control;
    if (parameter.type === 'choice') {
      control = document.createElement('select');
      addOption(control, '', 'Choose a value', true);
      (parameter.choices || []).forEach((choice) => {
        addOption(control, String(choice), String(choice));
      });
    } else {
      control = document.createElement('input');
      control.type = parameter.type === 'float' ? 'number' : parameter.type === 'int' ? 'number' : 'text';
      if (parameter.minimum !== undefined && parameter.minimum !== null) control.min = parameter.minimum;
      if (parameter.maximum !== undefined && parameter.maximum !== null) control.max = parameter.maximum;
      if (parameter.step !== undefined && parameter.step !== null) control.step = parameter.step;
    }

    control.id = controlId;
    control.name = parameter.name;
    control.required = false;
    const initialValue = values[parameter.name] ?? parameter.default;
    if (initialValue !== undefined && initialValue !== null) control.value = initialValue;
    label.appendChild(control);

    if (parameter.help) {
      const help = document.createElement('small');
      help.textContent = parameter.help;
      label.appendChild(help);
    }

    wrapper.appendChild(label);
    return wrapper;
  };

  const renderMethodParameters = (method, container, guide, initialValues) => {
    container.replaceChildren();
    guide.replaceChildren();
    if (!method) {
      container.hidden = true;
      return;
    }

    const parameters = method.params || [];
    parameters.forEach((parameter) => {
      container.appendChild(buildParameterControl(parameter, initialValues));
    });
    container.hidden = parameters.length === 0;

    const card = document.createElement('article');
    card.className = 'training-guide-card method-guide-card';
    const heading = document.createElement('h3');
    heading.textContent = method.label;
    card.appendChild(heading);

    const summary = document.createElement('p');
    summary.className = 'method-summary';
    summary.textContent = parameters.length
      ? `Parameters: ${parameters.map((parameter) => parameter.label || parameter.name).join(', ')}.`
      : 'Parameters: None.';
    card.appendChild(summary);

    if (parameters.length) {
      const title = document.createElement('h4');
      title.textContent = 'Method parameters';
      card.appendChild(title);
      const list = document.createElement('ul');
      list.className = 'training-guide-list';
      parameters.forEach((parameter) => {
        const item = document.createElement('li');
        const strong = document.createElement('strong');
        strong.textContent = `${parameter.label || parameter.name} (${parameter.name}):`;
        item.appendChild(strong);
        if (parameter.help) item.append(` ${parameter.help}`);
        if (parameter.type) item.append(` Type: ${parameter.type};`);
        if (parameter.default !== undefined && parameter.default !== null) item.append(` Default: ${parameter.default};`);
        if (parameter.minimum !== undefined && parameter.minimum !== null) item.append(` Min: ${parameter.minimum};`);
        if (parameter.maximum !== undefined && parameter.maximum !== null) item.append(` Max: ${parameter.maximum};`);
        list.appendChild(item);
      });
      card.appendChild(list);
    }
    guide.appendChild(card);
  };

  const initializeTrainingForm = async () => {
    const form = document.querySelector('[data-api-form="training"]');
    if (!form) return;

    const methodSelect = form.querySelector('[data-training-method]');
    const datasetSelect = form.querySelector('[data-training-dataset]');
    const parameterContainer = form.querySelector('[data-training-parameters]');
    const guide = document.querySelector('[data-training-guide]');
    const status = form.querySelector('[data-api-status]');
    const state = readJsonElement(document.getElementById('training-form-state'));

    try {
      const [datasets, methods] = await Promise.all([
        apiRequest('/api/datasets'),
        apiRequest('/api/metadata/methods'),
      ]);

      datasetSelect.replaceChildren();
      addOption(datasetSelect, '', 'Choose a dataset...', true);
      datasets.filter((dataset) => dataset.training_available).forEach((dataset) => {
        addOption(datasetSelect, dataset.dataset_key, dataset.label);
      });
      if (state.dataset && datasets.some((dataset) => dataset.dataset_key === state.dataset && dataset.training_available)) {
        datasetSelect.value = state.dataset;
      }

      methodSelect.replaceChildren();
      methods.forEach((method) => addOption(methodSelect, method.value, method.label));
      const selectedMethod = methods.some((method) => method.value === state.method)
        ? state.method
        : methods[0] ? methods[0].value : '';
      methodSelect.value = selectedMethod;

      const renderSelectedMethod = () => {
        const method = methods.find((candidate) => candidate.value === methodSelect.value);
        renderMethodParameters(method, parameterContainer, guide, state);
      };
      methodSelect.addEventListener('change', renderSelectedMethod);
      renderSelectedMethod();

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        setStatus(status, 'Training model...', false);
        const payload = Object.fromEntries(new FormData(form).entries());
        delete payload.form_id;
        try {
          const result = await apiRequest('/api/models/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
          });
          setStatus(status, `Model ${result.model_id || ''} trained successfully.`, false);
          window.location.assign('/results');
        } catch (error) {
          setStatus(status, error.message, true);
        }
      });
    } catch (error) {
      setStatus(status, `Form configuration could not be loaded: ${error.message}`, true);
    }
  };

  const PREDICTION_GROUPS = [
    {
      label: 'Patient profile',
      description: 'Baseline demographics and presenting symptom.',
      fields: ['age', 'sex', 'cp'],
    },
    {
      label: 'Resting measures',
      description: 'Measurements recorded before exercise testing.',
      fields: ['trestbps', 'chol', 'fbs', 'restecg'],
    },
    {
      label: 'Exercise and vessel signals',
      description: 'Response to exertion and coded cardiac findings.',
      fields: ['thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'],
    },
  ];

  const createPredictionField = (field, definition) => {
    const wrapper = document.createElement('div');
    wrapper.className = 'prediction-field';
    wrapper.dataset.field = field;

    const label = document.createElement('label');
    label.className = 'prediction-field-label';
    const controlId = `prediction-field-${field}`;
    label.htmlFor = controlId;

    const labelText = document.createElement('span');
    labelText.className = 'prediction-field-name';
    labelText.textContent = definition.label || field;
    const fieldCode = document.createElement('span');
    fieldCode.className = 'prediction-field-code';
    fieldCode.textContent = field;
    fieldCode.title = `Schema field: ${field}`;
    fieldCode.setAttribute('aria-label', `Short name: ${field}`);
    labelText.appendChild(fieldCode);
    label.appendChild(labelText);

    if (definition.units) {
      const units = document.createElement('span');
      units.className = 'prediction-field-units';
      units.textContent = definition.units;
      label.appendChild(units);
    }

    let control;
    if (definition.type === 'select' || definition.options) {
      control = document.createElement('select');
      control.dataset.domainSelect = 'true';
      addOption(control, '', 'Choose a value', true);
      (definition.options || []).forEach((option) => {
        addOption(control, String(option.value), option.label);
      });
    } else {
      control = document.createElement('input');
      control.type = definition.type || 'number';
      if (definition.minimum !== undefined) control.min = definition.minimum;
      if (definition.maximum !== undefined) control.max = definition.maximum;
      if (definition.step !== undefined) control.step = definition.step;
      control.inputMode = definition.step && String(definition.step).includes('.') ? 'decimal' : 'numeric';
    }

    control.id = controlId;
    control.name = field;
    control.required = definition.required !== false;
    control.autocomplete = 'off';
    label.appendChild(control);
    wrapper.appendChild(label);

    const hint = document.createElement('span');
    hint.className = 'hint prediction-field-hint';
    hint.textContent = definition.description || '';
    wrapper.appendChild(hint);
    return wrapper;
  };

  const createPredictionGroup = (group, fields, fieldDefinitions) => {
    const fieldset = document.createElement('fieldset');
    fieldset.className = 'prediction-group';

    const legend = document.createElement('legend');
    legend.textContent = group.label;
    fieldset.appendChild(legend);

    const description = document.createElement('p');
    description.className = 'prediction-group-description';
    description.textContent = group.description;
    fieldset.appendChild(description);

    const grid = document.createElement('div');
    grid.className = 'prediction-field-grid';
    fields.forEach((field) => {
      if (fieldDefinitions[field]) grid.appendChild(createPredictionField(field, fieldDefinitions[field]));
    });
    fieldset.appendChild(grid);
    return fieldset;
  };

  const renderPredictionResult = (container, result) => {
    container.replaceChildren();
    container.className = `api-result ${Number(result.prediction) === 1 ? 'is-warning' : 'is-success'}`;
    const title = document.createElement('strong');
    title.className = 'api-result-title';
    title.textContent = Number(result.prediction) === 1 ? 'Heart disease indication' : 'No heart disease indication';
    const probability = document.createElement('span');
    probability.className = 'api-result-probability';
    probability.textContent = `${(Number(result.probability) * 100).toFixed(1)}% model probability`;
    container.append(title, probability);
    container.hidden = false;
  };

  const initializePredictionForm = async () => {
    const form = document.querySelector('[data-api-form="prediction"]');
    if (!form) return;

    const modelSelect = form.querySelector('[data-prediction-model]');
    const fieldsContainer = form.querySelector('[data-prediction-fields]');
    const status = form.querySelector('[data-api-status]');
    const resultContainer = form.querySelector('[data-prediction-result]');
    const fieldCount = form.querySelector('[data-prediction-field-count]');
    let models = [];
    let fieldDefinitions = {};
    const predictionValues = new Map();

    try {
      [models, fieldDefinitions] = await Promise.all([
        apiRequest('/api/models'),
        apiRequest('/api/metadata/field-definitions'),
      ]);

      modelSelect.replaceChildren();
      addOption(modelSelect, '', models.length ? 'Select a model to deploy' : 'No trained models available', true);
      models.forEach((model) => {
        addOption(modelSelect, model.model_id, model.display_name || model.model_id);
      });
      modelSelect.disabled = models.length === 0;
      form.querySelector('button[type="submit"]').disabled = models.length === 0;

      const rememberPredictionValues = () => {
        fieldsContainer.querySelectorAll('[name]').forEach((control) => {
          predictionValues.set(control.name, control.value);
        });
      };

      const restorePredictionValues = () => {
        fieldsContainer.querySelectorAll('[name]').forEach((control) => {
          if (predictionValues.has(control.name)) {
            control.value = predictionValues.get(control.name);
          }
        });
      };

      const renderSelectedModelFields = () => {
        rememberPredictionValues();
        fieldsContainer.replaceChildren();
        const selectedModel = models.find((model) => model.model_id === modelSelect.value);
        const fields = selectedModel && Array.isArray(selectedModel.feature_fields) && selectedModel.feature_fields.length
          ? selectedModel.feature_fields
          : Object.keys(fieldDefinitions);
        const renderedFields = new Set();
        PREDICTION_GROUPS.forEach((group) => {
          const groupFields = group.fields.filter((field) => fields.includes(field));
          if (!groupFields.length) return;
          fieldsContainer.appendChild(createPredictionGroup(group, groupFields, fieldDefinitions));
          groupFields.forEach((field) => renderedFields.add(field));
        });
        fields.filter((field) => !renderedFields.has(field) && fieldDefinitions[field]).forEach((field) => {
          fieldsContainer.appendChild(createPredictionGroup(
            { label: 'Additional inputs', description: 'Fields supplied by the selected model.' },
            [field],
            fieldDefinitions,
          ));
        });
        restorePredictionValues();
        if (fieldCount) fieldCount.textContent = `${renderedFields.size} schema fields`;
      };

      modelSelect.addEventListener('change', renderSelectedModelFields);
      renderSelectedModelFields();

      form.addEventListener('submit', async (event) => {
        event.preventDefault();
        setStatus(status, 'Running prediction...', false);
        if (resultContainer) resultContainer.hidden = true;
        try {
          const result = await apiRequest('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(Object.fromEntries(new FormData(form).entries())),
          });
          setStatus(status, '', false);
          if (resultContainer) renderPredictionResult(resultContainer, result);
        } catch (error) {
          setStatus(status, error.message, true);
        }
      });
    } catch (error) {
      setStatus(status, `Form configuration could not be loaded: ${error.message}`, true);
    }
  };

  initializeTrainingForm();
  initializePredictionForm();
})();
