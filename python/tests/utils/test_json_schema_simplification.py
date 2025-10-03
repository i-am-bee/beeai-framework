import pytest

from beeai_framework.utils.schema import simplify_schema


@pytest.mark.unit
def test_drop_not_empty_schema() -> None:
    schema = {"anyOf": [{"not": {}}, {"type": "string"}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": ["string"]}


@pytest.mark.unit
def test_flatten_nested_anyof() -> None:
    schema = {"anyOf": [{"anyOf": [{"type": "string"}]}, {"type": "null"}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": ["null", "string"]}


@pytest.mark.unit
def test_flatten_nested_oneof() -> None:
    schema = {"oneOf": [{"oneOf": [{"type": "string"}, {"type": "number"}]}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": ["number", "string"]}


@pytest.mark.unit
def test_deduplicate_types_in_anyof() -> None:
    schema = {"anyOf": [{"type": "string"}, {"type": "string"}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": ["string"]}


@pytest.mark.unit
def test_collapse_type_only_anyof() -> None:
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": ["null", "string"]}


@pytest.mark.unit
def test_dont_collapse_mixed_constraints() -> None:
    schema = {"anyOf": [{"type": "string"}, {"minimum": 5}]}
    simplified = simplify_schema(schema)
    assert simplified == {"anyOf": [{"type": "string"}, {"type": "number", "minimum": 5}]}


@pytest.mark.unit
def test_string_constraint_without_type() -> None:
    schema = {"maxLength": 10}
    simplified = simplify_schema(schema)
    assert simplified == {"type": "string", "maxLength": 10}


@pytest.mark.unit
def test_numeric_constraint_without_type() -> None:
    schema = {"minimum": 5}
    simplified = simplify_schema(schema)
    assert simplified == {"type": "number", "minimum": 5}


@pytest.mark.unit
def test_object_properties_adds_type() -> None:
    schema = {"properties": {"x": {"type": "string"}}}
    simplified = simplify_schema(schema)
    assert simplified == {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    }


@pytest.mark.unit
def test_items_adds_type_array() -> None:
    schema = {"items": {"type": "string"}}
    simplified = simplify_schema(schema)
    assert simplified == {"type": "array", "items": {"type": "string"}}


@pytest.mark.unit
def test_merge_allof_simple() -> None:
    schema = {"allOf": [{"type": "string"}, {"maxLength": 5}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": "string", "maxLength": 5}


@pytest.mark.unit
def test_allof_conflict_not_merged() -> None:
    schema = {"allOf": [{"type": "string"}, {"type": "number"}]}
    simplified = simplify_schema(schema)
    assert simplified == {"allOf": [{"type": "string"}, {"type": "number"}]}


@pytest.mark.unit
def test_nested_properties_simplification() -> None:
    schema = {
        "type": "object",
        "properties": {
            "namespace": {
                "anyOf": [
                    {"anyOf": [{"not": {}}, {"type": "string"}]},
                    {"type": "null"},
                ]
            }
        },
    }
    simplified = simplify_schema(schema)
    assert simplified == {
        "type": "object",
        "properties": {"namespace": {"type": ["null", "string"]}},
    }


@pytest.mark.unit
def test_empty_anyof_removed_to_empty_schema() -> None:
    schema = {"anyOf": [{"not": {}}]}
    simplified = simplify_schema(schema)
    # Nothing valid remains, schema should accept anything (empty schema)
    assert simplified == {}


@pytest.mark.unit
def test_oneof_single_schema_unwraps() -> None:
    schema = {"oneOf": [{"type": "string"}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": "string"}


@pytest.mark.unit
def test_anyof_single_schema_unwraps() -> None:
    schema = {"anyOf": [{"type": "string"}]}
    simplified = simplify_schema(schema)
    assert simplified == {"type": "string"}
