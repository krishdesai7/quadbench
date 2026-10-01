import pytest

from analysis import data


@pytest.fixture(scope="session")
def d() -> data.AnalysisData:
    """Every benchmark run, loaded once for the whole test session."""
    return data.load_analysis()
