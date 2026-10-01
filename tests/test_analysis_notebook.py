"""Structure, prose consistency, saved outputs, and execution of analysis.ipynb.

The notebook is presented without re-running it, so these checks read the
committed file: every number it quotes must match ``analysis.findings``, and
every chart must already be rendered in the saved outputs.
"""

import ast
import re
from pathlib import Path

import nbformat
import pytest
from nbclient import NotebookClient

from analysis import findings

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "analysis.ipynb"


@pytest.fixture(scope="module")
def nb():
    return nbformat.read(NOTEBOOK, as_version=4)


@pytest.fixture(scope="module")
def code_cells(nb):
    return [cell for cell in nb.cells if cell.cell_type == "code"]


@pytest.fixture(scope="module")
def prose(nb):
    return "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")


def test_code_cells_are_short_calls_without_definitions(code_cells):
    for cell in code_cells:
        tree = ast.parse(cell.source)
        definitions = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        ]
        assert not definitions, cell.source
        large = [node for node in ast.walk(tree) if isinstance(node, ast.Dict) and len(node.keys) >= 8]
        assert not large, cell.source
        assert len([line for line in cell.source.splitlines() if line.strip()]) <= 3, cell.source


def test_code_cells_are_collapsed_in_jupyterlab(code_cells):
    for cell in code_cells:
        assert cell.metadata.get("jupyter", {}).get("source_hidden") is True, cell.source


def test_prose_quotes_every_finding(d, prose):
    missing = {key: text for key, text in findings.compute(d).items() if text not in prose}
    assert not missing, missing


# Numbers with units that are facts about the setup, not measurements.
CONSTANTS = {"10×", "2×", "2 GB", "32 MiB", "4 KiB"}
QUANTITY = re.compile(
    r"±?\d[\d,]*(?:\.\d+)?(?:×| ns| µs| ms| GB/s|%| MiB| KiB| GB| B\b| ULP| billion elements per second)"
)


def test_every_quoted_quantity_is_a_current_finding(d, prose):
    # A number edited in one place survives the check above if it is quoted
    # elsewhere too; this catches it, and any number typed in by hand.
    known = set(findings.compute(d).values()) | CONSTANTS
    unknown = sorted(set(QUANTITY.findall(prose)) - known)
    assert not unknown, unknown


def test_no_changelog_language(prose):
    assert not re.search(r"earlier (version|reading)|supersede|previous version|no longer claim", prose, re.I)


def test_every_figure_has_a_saved_rendered_chart(code_cells):
    for cell in code_cells:
        if "figures." not in cell.source:
            continue
        html = [output["data"].get("text/html", "") for output in cell.outputs if "data" in output]
        assert any("vegaEmbed" in h for h in html), cell.source


def test_saved_outputs_have_no_errors_or_stderr(code_cells):
    for cell in code_cells:
        for output in cell.outputs:
            assert output.output_type != "error", cell.source
            assert not (output.output_type == "stream" and output.get("name") == "stderr"), cell.source


def test_every_lesson_has_the_same_shape(prose):
    for lesson in range(1, 9):
        assert re.search(rf"^## {lesson} · ", prose, re.M), lesson
    assert prose.count("**What this means in practice**") == 8
    assert prose.count("<summary>Details</summary>") == 8


@pytest.mark.slow
def test_notebook_executes_end_to_end(nb):
    NotebookClient(
        nbformat.from_dict(nb),
        timeout=900,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
