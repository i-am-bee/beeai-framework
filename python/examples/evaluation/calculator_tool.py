# Copyright 2025 © BeeAI a Series of LF Projects, LLC
# SPDX-License-Identifier: Apache-2.0

import ast
import operator
from typing import Self

from pydantic import BaseModel, Field

from beeai_framework.context import RunContext
from beeai_framework.emitter import Emitter
from beeai_framework.tools import StringToolOutput, Tool, ToolRunOptions
from beeai_framework.tools.errors import ToolInputValidationError

# Only these node/operator types are ever evaluated — no names, calls, attribute access,
# comprehensions, etc. This intentionally cannot execute arbitrary code.
_ALLOWED_BINOPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_node(node: ast.expr) -> float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, int | float) and not isinstance(node.value, bool):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BINOPS[type(node.op)](left, right)  # type: ignore[operator]

    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARYOPS:
        return _ALLOWED_UNARYOPS[type(node.op)](_eval_node(node.operand))  # type: ignore[operator]

    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def evaluate_arithmetic(expression: str) -> float:
    """Safely evaluate a basic arithmetic expression (+ - * / // % **, parentheses, unary +/-).

    Unlike `eval()`, this only ever walks a whitelisted subset of the AST — numeric
    literals and arithmetic operators — so it cannot execute arbitrary code, names,
    attribute access, or function calls."""
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression: {expression!r}") from e
    return _eval_node(tree.body)


class CalculatorToolInput(BaseModel):
    expression: str = Field(
        description="A basic arithmetic expression using only numbers, + - * / // % **, "
        "parentheses, and unary +/- (e.g. '(3 + 4) * 2'). No variables or function calls."
    )


class CalculatorTool(Tool[CalculatorToolInput]):
    """A minimal, dependency-free arithmetic tool — no external code interpreter required.

    Evaluates numeric expressions through a restricted AST walk (see evaluate_arithmetic)
    and cannot execute arbitrary code."""

    name = "Calculator"
    description = (
        "Evaluate a basic arithmetic expression (+ - * / // % ** and parentheses) and "
        "return the numeric result. Use for calculations only — no code execution."
    )
    input_schema = CalculatorToolInput

    async def clone(self) -> Self:
        tool = self.__class__()
        tool.name = self.name
        tool.description = self.description
        tool.input_schema = self.input_schema
        tool.middlewares.extend(self.middlewares)
        tool._cache = await self.cache.clone()
        return tool

    def _create_emitter(self) -> Emitter:
        return Emitter.root().child(
            namespace=["tool", "evaluation", "calculator"],
            creator=self,
        )

    # pyrefly: ignore [bad-param-name-override]
    async def _run(
        self, tool_input: CalculatorToolInput, options: ToolRunOptions | None, context: RunContext
    ) -> StringToolOutput:
        try:
            result = evaluate_arithmetic(tool_input.expression)
        except ValueError as e:
            raise ToolInputValidationError(str(e)) from e
        return StringToolOutput(str(result))
