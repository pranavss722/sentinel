"""Tests for locustfile.py load test definition."""
import ast
import os

import pytest


LOCUSTFILE = "locustfile.py"


def _parse_locustfile() -> ast.Module:
    with open(LOCUSTFILE) as f:
        return ast.parse(f.read(), filename=LOCUSTFILE)


def _find_http_user_subclass(tree: ast.Module) -> ast.ClassDef | None:
    """Find a class that inherits from HttpUser via AST inspection."""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "HttpUser":
                    return node
                if isinstance(base, ast.Attribute) and base.attr == "HttpUser":
                    return node
    return None


def _get_task_methods(cls_node: ast.ClassDef) -> list[ast.FunctionDef]:
    """Return methods decorated with @task."""
    tasks = []
    for item in cls_node.body:
        if isinstance(item, ast.FunctionDef):
            for dec in item.decorator_list:
                dec_name = None
                if isinstance(dec, ast.Name):
                    dec_name = dec.id
                elif isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name):
                    dec_name = dec.func.id
                if dec_name == "task":
                    tasks.append(item)
    return tasks


def test_locustfile_exists():
    assert os.path.isfile(LOCUSTFILE), "locustfile.py not found in project root"


def test_locustfile_is_valid_python():
    _parse_locustfile()  # raises SyntaxError if invalid


def test_locust_user_class_exists():
    tree = _parse_locustfile()
    cls = _find_http_user_subclass(tree)
    assert cls is not None, "No HttpUser subclass found in locustfile"


def test_locust_tasks_defined():
    tree = _parse_locustfile()
    cls = _find_http_user_subclass(tree)
    assert cls is not None
    tasks = _get_task_methods(cls)
    assert len(tasks) >= 2, f"Expected at least 2 @task methods, found {len(tasks)}"


def test_locust_wait_time_defined():
    tree = _parse_locustfile()
    cls = _find_http_user_subclass(tree)
    assert cls is not None
    has_wait_time = any(
        isinstance(item, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "wait_time"
            for t in item.targets
        )
        for item in cls.body
    )
    assert has_wait_time, "wait_time not defined on user class"


def test_locust_host_defined():
    tree = _parse_locustfile()
    cls = _find_http_user_subclass(tree)
    assert cls is not None
    for item in cls.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name) and target.id == "host":
                    assert isinstance(item.value, ast.Constant)
                    assert item.value.value == "http://localhost:8000", (
                        f"host is {item.value.value!r}"
                    )
                    return
    pytest.fail("host not defined on user class")
