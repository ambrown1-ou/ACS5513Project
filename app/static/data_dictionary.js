(function () {
  const dictionary = document.querySelector('[data-schema-dictionary]');
  if (!dictionary) return;

  const rows = dictionary.querySelector('[data-dictionary-rows]');
  const status = dictionary.querySelector('[data-dictionary-status]');
  const schemaLabel = dictionary.querySelector('[data-dictionary-schema]');

  const createCell = (tagName, text) => {
    const cell = document.createElement(tagName);
    cell.textContent = text;
    return cell;
  };

  const formatDomain = (field) => {
    if (Array.isArray(field.allowed)) return `Allowed: ${field.allowed.join(', ')}`;
    if (field.minimum !== undefined || field.maximum !== undefined) {
      return `Numeric (${field.minimum ?? ''}-${field.maximum ?? ''})`;
    }
    return 'Not constrained';
  };

  const renderSchema = (schema) => {
    const fields = Array.isArray(schema.fields) ? schema.fields : [];
    if (schemaLabel) schemaLabel.textContent = `${schema.label} v${schema.version}`;
    if (!rows) return;
    rows.replaceChildren();
    fields.forEach((field) => {
      const row = document.createElement('tr');
      const fieldCell = createCell('td', field.field || '');
      const fieldCode = document.createElement('code');
      fieldCode.textContent = field.field || '';
      fieldCell.replaceChildren(fieldCode);
      row.append(
        fieldCell,
        createCell('td', field.role === 'classifier' ? 'Target' : 'Feature'),
        createCell('td', formatDomain(field)),
        createCell('td', field.units || ''),
        createCell('td', field.description || ''),
      );
      rows.appendChild(row);
    });
    if (status) status.hidden = true;
  };

  fetch('/api/schemas')
    .then((response) => {
      if (!response.ok) throw new Error('Schema metadata could not be loaded.');
      return response.json();
    })
    .then((schemas) => {
      const schema = Array.isArray(schemas) ? schemas[0] : null;
      if (!schema) throw new Error('No schema is available.');
      renderSchema(schema);
    })
    .catch((error) => {
      if (status) {
        status.hidden = false;
        status.textContent = error.message;
      }
      if (schemaLabel) schemaLabel.textContent = 'Unavailable';
    });
}());