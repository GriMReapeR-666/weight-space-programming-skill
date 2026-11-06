import jsonschema

def validate_schema(instance, schema):
    """Validate *instance* against a JSON *schema* using ``jsonschema``.

    Raises ``jsonschema.exceptions.ValidationError`` if the data does not
    conform. Returns ``True`` on success.
    """
    jsonschema.validate(instance=instance, schema=schema)
    return True
