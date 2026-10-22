"""Guard the install path a developer or contributor actually follows.

`pip install -r requirements.txt` then `python -m setu run` is the basic installation contract of
this repository: no API keys, no service accounts, one command. It only works if
every module the code imports is declared. That was broken once - pandas was used
throughout and listed nowhere - so it is now checked rather than trusted.
"""

import argparse
import ast
import re
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # requires-python allows 3.10; tomllib is 3.11+
    tomllib = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
STDLIB = set(sys.stdlib_module_names)

# PyPI distribution name -> the name it is imported under.
ALIAS = {"scikit-learn": "sklearn", "PyYAML": "yaml"}


def _name(spec: str) -> str:
    return re.split(r"[<>=!;\[]", spec)[0].strip()


def imported_modules() -> set[str]:
    found = set()
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def declared_requirements() -> set[str]:
    names = set()
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        if line:
            names.add(_name(line))
    return names


def pyproject_dependencies() -> tuple[set[str], set[str]]:
    py = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    core = {_name(d) for d in py["project"]["dependencies"]}
    extras = {_name(d)
              for group in py["project"]["optional-dependencies"].values()
              for d in group}
    return core, extras


@pytest.mark.skipif(tomllib is None,
                    reason="tomllib arrived in 3.11; requires-python allows 3.10")
def test_requirements_and_pyproject_declare_the_same_core():
    core, _ = pyproject_dependencies()
    declared = declared_requirements() | set(ALIAS)
    missing = sorted(core - declared)
    assert not missing, f"in pyproject but not requirements.txt: {missing}"


@pytest.mark.skipif(tomllib is None,
                    reason="tomllib arrived in 3.11; requires-python allows 3.10")
def test_the_api_stays_an_optional_extra():
    """`setu serve` needs fastapi; the pipeline must not, so that a plain
    `pip install -r requirements.txt` on a developer machine is a small install."""
    core, extras = pyproject_dependencies()
    assert not core & {"fastapi", "uvicorn"}
    assert {"fastapi", "uvicorn"} <= extras


def test_no_undeclared_third_party_import():
    declared = declared_requirements()
    reverse = {v: k for k, v in ALIAS.items()}
    missing = []
    for module in sorted(imported_modules()):
        if module in STDLIB or module == "setu":
            continue
        canonical = reverse.get(module, module)
        if canonical not in declared:
            missing.append(f"{module} ({canonical})")
    assert not missing, f"imported by src/ but absent from requirements.txt: {missing}"


def test_cli_entrypoint_resolves_to_a_real_callable():
    py_text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    target = re.search(r'setu\s*=\s*"([\w.]+):(\w+)"', py_text)
    assert target, "no setu console script declared"
    module = __import__(target.group(1), fromlist=[target.group(2)])
    assert callable(getattr(module, target.group(2)))


def test_run_and_check_are_real_subcommands():
    from setu.cli import build_parser

    parser = build_parser()
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            assert {"run", "check", "serve", "fetch", "bom"} <= set(action.choices)
            return
    raise AssertionError("cli.py exposes no subcommands")
