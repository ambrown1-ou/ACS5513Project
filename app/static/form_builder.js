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

  const createPredictionField = (field, definition) => {
    const label = document.createElement('label');
    label.textContent = definition.label || field;

    let control;
    if (definition.type === 'select' || definition.options) {
      control = document.createElement('select');
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
    }

    control.name = field;
    control.required = definition.required !== false;
    label.appendChild(control);

    const hint = document.createElement('span');
    hint.className = 'hint';
    hint.textContent = `${definition.description || ''}${definition.domain ? ` Domain: ${definition.domain}` : ''}`;
    label.appendChild(hint);
    return label;
  };

  const renderPredictionResult = (container, result) => {
    container.replaceChildren();
    const title = document.createElement('strong');
    title.textContent = `Prediction: ${result.prediction}`;
    const probability = document.createElement('span');
    probability.textContent = ` Probability: ${result.probability}`;
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
    let models = [];
    let fieldDefinitions = {};

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

      const renderSelectedModelFields = () => {
        fieldsContainer.replaceChildren();
        const selectedModel = models.find((model) => model.model_id === modelSelect.value);
        const fields = selectedModel && Array.isArray(selectedModel.feature_fields) && selectedModel.feature_fields.length
          ? selectedModel.feature_fields
          : Object.keys(fieldDefinitions);
        fields.forEach((field) => {
          if (fieldDefinitions[field]) fieldsContainer.appendChild(createPredictionField(field, fieldDefinitions[field]));
        });
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
