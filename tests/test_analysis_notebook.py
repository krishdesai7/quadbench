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


# A number in running text: not part of a word (fp32, A100, L2) and not a
# fragment of a longer number (the 400 in 400,000).
NUMBER = re.compile(r"(?<![\w.,])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\w|[.,]\d)")

# Numbers in the prose that describe the setup rather than a measurement:
# sizes, counts of things, thresholds, and examples. Anything else must come
# from analysis.findings.
SETUP_NUMBERS = {
    # lesson and list numbers, and small counts in the run table
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "13", "16", "18",
    # problem sizes named in the text
    "1,000", "1000", "256,000", "100", "250",
    # facts about the platform: CPython 3.14, float object sizes, 128-bit quad, 64-bit
    "3.14", "24", "40", "128", "64",
}


def _unknown_numbers(prose: str, texts: dict[str, str]) -> list[str]:
    """Numbers in the prose that are neither a finding nor a setup constant."""
    known = {number for text in texts.values() for number in NUMBER.findall(text)} | SETUP_NUMBERS
    return sorted(set(NUMBER.findall(prose)) - known)


def _missing_findings(prose: str, texts: dict[str, str]) -> dict[str, str]:
    """Findings whose formatted text never appears as a whole number in the prose."""
    return {
        key: text
        for key, text in texts.items()
        if not re.search(rf"(?<![\w.,]){re.escape(text)}(?!\w|[.,]\d)", prose)
    }


def test_prose_quotes_every_finding(d, prose):
    missing = _missing_findings(prose, findings.compute(d))
    assert not missing, missing


def test_every_number_in_the_prose_is_a_finding_or_setup_constant(d, prose):
    unknown = _unknown_numbers(prose, findings.compute(d))
    assert not unknown, unknown


def test_no_changelog_language(prose):
    assert not re.search(r"earlier (version|reading)|supersede|previous version|no longer claim", prose, re.IGNORECASE)


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
        assert re.search(rf"^## {lesson} · ", prose, re.MULTILINE), lesson
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


@pytest.mark.parametrize(
    ("before", "after"),
    [
        ("Below about **{crossover} elements**", "Below about **26,000 elements**"),
        ("{jit_muladd_min}–", "1.97–"),
        ("{plateau_cells} cheap", "11 cheap"),  # swapping in another finding (12) is undetectable
        ("(at {f16_vs_f32_min_elems} elements)", "(at 4,000 elements)"),
    ],
)
def test_number_guard_catches_a_single_edited_copy(d, prose, before, after):
    texts = findings.compute(d)
    original, edited = before.format_map(texts), after.format_map(texts)
    assert original in prose, original
    mutated = prose.replace(original, edited, 1)
    assert _unknown_numbers(mutated, texts) or _missing_findings(mutated, texts), after
