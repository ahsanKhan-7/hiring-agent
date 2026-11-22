import sys
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hiring_agent.graph import build_graph  # noqa: E402

JD = "We need Python, pandas, scikit-learn, sql and machine learning skills, plus git."

GOOD = {"name": "good", "text": "Ann\nExperience\nJan 2024 - Dec 2024 Data Scientist\n"
        "Python pandas scikit-learn sql machine learning git"}
STUFFER = {"name": "stuffer", "text": "Bob\nSkills\npython pandas scikit-learn sql machine learning git\n"
           "Experience\n2024 - 2025 Barista\nServed coffee"}
OFF = {"name": "off", "text": "Cy\nExperience\n2024 - 2025 Chef\nKitchen management"}
CLEAN_SET = [GOOD, {"name": "ok", "text": "Di\nExperience\nJan 2024 - Dec 2024 Analyst\nPython sql pandas reporting"}, OFF]


@pytest.fixture()
def graph():
    return build_graph(checkpointer=MemorySaver())
