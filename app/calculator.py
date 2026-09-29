import ast
import json
import math
import operator
import re
import sys

from simpleeval import InvalidExpression, simple_eval


# ---------- Configuration ----------

MAX_EXPRESSION_LENGTH = 200

OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


# ---------- Calculator tool ----------

def calculator(expression: str) -> dict:
    """Evaluate basic arithmetic and return a result or clear error."""
    if not isinstance(expression, str):
        return {"ok": False, "error": "Expression must be a string."}

    expression = expression.strip()

    if not expression:
        return {"ok": False, "error": "Expression cannot be empty."}

    if len(expression) > MAX_EXPRESSION_LENGTH:
        return {"ok": False, "error": "Expression exceeds 200 characters."}

    # Allow numbers, whitespace, parentheses and arithmetic symbols only.
    if not re.fullmatch(r"[0-9\s.+*/%()\-]+", expression):
        return {
            "ok": False,
            "error": "Use numbers and arithmetic symbols only.",
        }

    # Exponentiation is outside this basic calculator's scope.
    if "**" in expression:
        return {"ok": False, "error": "Exponentiation is not supported."}

    try:
        result = simple_eval(
            expression,
            operators=OPERATORS,
            names={},
            functions={},
        )

        if type(result) not in (int, float):
            return {"ok": False, "error": "The result must be a number."}

        if isinstance(result, float) and not math.isfinite(result):
            return {"ok": False, "error": "The result is not finite."}

        return {
            "ok": True,
            "expression": expression,
            "result": result,
        }

    except ZeroDivisionError:
        return {"ok": False, "error": "Cannot divide by zero."}

    except (InvalidExpression, SyntaxError, TypeError, ValueError):
        return {"ok": False, "error": "Invalid arithmetic expression."}

    except (OverflowError, RecursionError):
        return {"ok": False, "error": "Expression is too large or complex."}


# ---------- Run directly from the terminal ----------

def main() -> None:
    """Accept a quoted expression from the command line."""
    expression = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "2 + 3"
    print(json.dumps(calculator(expression), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()