import pytest
from conftest import CLEAN_SET, GOOD, JD, OFF, STUFFER
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import MemorySaver

from hiring_agent.api import create_app
from hiring_agent.graph import build_graph


@pytest.fixture()
def client():
    with TestClient(create_app(graph=build_graph(checkpointer=MemorySaver()), api_key="")) as c:
        yield c


def body(resumes, top_k=2):
    return {"jd_text": JD, "resumes": resumes, "top_k": top_k}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_clean_screening_completes_immediately(client):
    r = client.post("/screenings", json=body(CLEAN_SET))
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "completed" and data["shortlist"] == ["good", "ok"]
    assert data["ranking"][0]["rank"] == 1 and "contributions" in data["ranking"][0]
    assert client.get(f"/screenings/{data['id']}").json()["status"] == "completed"


def test_flagged_screening_waits_then_resumes(client):
    created = client.post("/screenings", json=body([GOOD, STUFFER, OFF])).json()
    assert created["status"] == "awaiting_review"
    assert created["pending_review"]["flags"][0]["kind"] == "keyword_stuffing"
    sid = created["id"]
    assert client.get(f"/screenings/{sid}").json()["status"] == "awaiting_review"

    done = client.post(f"/screenings/{sid}/review", json={"approve": True, "exclude": ["stuffer"]}).json()
    assert done["status"] == "completed" and done["shortlist"] == ["good", "off"]


def test_review_rejection(client):
    sid = client.post("/screenings", json=body([GOOD, STUFFER, OFF])).json()["id"]
    done = client.post(f"/screenings/{sid}/review", json={"approve": False}).json()
    assert done["status"] == "rejected" and done["shortlist"] == []


def test_review_is_only_allowed_while_waiting(client):
    sid = client.post("/screenings", json=body(CLEAN_SET)).json()["id"]
    assert client.post(f"/screenings/{sid}/review", json={"approve": True}).status_code == 409
    waiting = client.post("/screenings", json=body([GOOD, STUFFER, OFF])).json()["id"]
    client.post(f"/screenings/{waiting}/review", json={"approve": True})
    assert client.post(f"/screenings/{waiting}/review", json={"approve": True}).status_code == 409


@pytest.mark.parametrize("sid", ["0" * 32, "not-a-valid-id", "../etc/passwd"])
def test_unknown_or_malformed_ids_are_404(client, sid):
    assert client.get(f"/screenings/{sid}").status_code == 404
    assert client.post(f"/screenings/{sid}/review", json={"approve": True}).status_code == 404


@pytest.mark.parametrize("payload", [
    {"jd_text": "too short", "resumes": [GOOD]},
    {"jd_text": JD, "resumes": []},
    {"jd_text": JD, "resumes": [{"name": "", "text": "x"}]},
    {"jd_text": JD, "resumes": [{"name": "a", "text": "x" * 20_001}]},
    {"jd_text": JD, "resumes": [GOOD], "top_k": 0},
    {"jd_text": JD, "resumes": [{"name": str(i), "text": "x"} for i in range(51)]},
])
def test_input_validation(client, payload):
    assert client.post("/screenings", json=payload).status_code == 422


def test_api_key_is_enforced_when_configured():
    app = create_app(graph=build_graph(checkpointer=MemorySaver()), api_key="s3cret")
    with TestClient(app) as c:
        assert c.get("/health").status_code == 200  # health stays open
        assert c.post("/screenings", json=body(CLEAN_SET)).status_code == 401
        assert c.post("/screenings", json=body(CLEAN_SET), headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/screenings", json=body(CLEAN_SET), headers={"X-API-Key": "s3cret"}).status_code == 200


def test_persistent_checkpoints_survive_an_app_restart(tmp_path, monkeypatch):
    """A paused screening is resumable after the process restarts (real SQLite checkpointer)."""
    monkeypatch.setenv("CHECKPOINT_DB", str(tmp_path / "cp.db"))
    monkeypatch.delenv("API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MCP_SERVER_URL", raising=False)
    with TestClient(create_app()) as first:
        sid = first.post("/screenings", json=body([GOOD, STUFFER, OFF])).json()["id"]
    with TestClient(create_app()) as second:  # new app, new graph, same database file
        assert second.get(f"/screenings/{sid}").json()["status"] == "awaiting_review"
        done = second.post(f"/screenings/{sid}/review", json={"approve": True}).json()
        assert done["status"] == "completed"
