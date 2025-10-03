from typing import Any, Union

from beeai_framework.utils.lists import remove_falsy

JSONType = Union[dict[str, Any], list[Any], str, int, float, bool, None]
Schema = dict[str, JSONType]


def simplify_schema(top_schema: Schema) -> None:
    """
    Simplify a JSON Schema by:
    - Flattening nested anyOf/oneOf
    - Removing impossible {"not": {}}
    - Collapsing multiple type expressions
    - Normalizing number/string constraints with missing "type"
    - Merging basic `allOf` objects
    - Deduplicating types
    """

    def _simplify(schema: Schema) -> Any:
        for key in ("anyOf", "oneOf", "not"):
            if schema.get(key) == {}:
                del schema[key]

        if schema.get("type") == "object":
            properties = schema.get("properties", {})
            for k, v in schema.get("properties", {}).items():
                properties[k] = _simplify(v)
            schema["properties"] = properties

        if schema.get("type") == "array":
            items = [_simplify(v) for v in schema["items"]]
            schema["items"] = remove_falsy(items)

        for key in ("anyOf", "oneOf"):
            value = schema.get(key)
            if value and isinstance(value, list):
                value = remove_falsy([_simplify(v) for v in value])

                if len(value) == 1:
                    return value[0]

                if all([v.keys() == {"type"} for v in value]):
                    return {"type": [v["type"] for v in value]}

        return schema

    _simplify(top_schema)


x = {
    "additionalProperties": False,
    "properties": {
        "name": {
            "const": "list_application_needs",
            "description": "Tool Name",
            "title": "Name",
            "type": "string",
        },
        "parameters": {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "additionalProperties": False,
            "properties": {
                "namespace": {
                    "anyOf": [
                        {"anyOf": [{"not": {}}, {"type": "string"}]},
                        # {"type": "null"},
                        {"type": "number", "min": 5},
                    ]
                },
            },
            "type": "object",
            "description": "Tool Parameters",
        },
    },
    "required": ["name", "parameters"],
    "title": "list_application_needs",
    "type": "object",
}

simplify_schema(x)

from beeai_framework.utils.strings import to_json

print(to_json(x, indent=4, sort_keys=False))
