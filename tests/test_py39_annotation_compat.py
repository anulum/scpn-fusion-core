from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "scpn_fusion"


def _has_future_annotations(tree: ast.Module) -> bool:
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            if any(alias.name == "annotations" for alias in node.names):
                return True
    return False


def _annotation_nodes(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.returns is not None:
                yield node.returns
            for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
                if arg.annotation is not None:
                    yield arg.annotation
            if node.args.vararg and node.args.vararg.annotation is not None:
                yield node.args.vararg.annotation
            if node.args.kwarg and node.args.kwarg.annotation is not None:
                yield node.args.kwarg.annotation
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _contains_pep604_union(expr: ast.AST) -> bool:
    for subnode in ast.walk(expr):
        if isinstance(subnode, ast.BinOp) and isinstance(subnode.op, ast.BitOr):
            return True
    return False


def test_pep604_annotations_require_future_import_for_py39_compat() -> None:
    offenders: list[str] = []

    for py_file in SRC_ROOT.rglob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        if _has_future_annotations(tree):
            continue

        for annotation in _annotation_nodes(tree):
            if _contains_pep604_union(annotation):
                rel = py_file.relative_to(ROOT).as_posix()
                lineno = getattr(annotation, "lineno", 0)
                offenders.append(f"{rel}:{lineno}")
                break

    assert offenders == [], (
        "PEP 604 '|' type annotations found without "
        "`from __future__ import annotations` (breaks Python 3.9 runtime import):\n"
        + "\n".join(offenders)
    )
