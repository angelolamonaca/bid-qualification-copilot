"""The web layer: the demo page, the real-identity endpoint and client-chosen bid ids."""

from fastapi.testclient import TestClient

from app.main import create_app
from app.store import MemoryStore
from tests.fakes import FakeLLM


def client() -> TestClient:
    return TestClient(create_app(llm=FakeLLM(), store=MemoryStore()))


def test_the_demo_page_is_served():
    with client() as c:
        page = c.get("/")
        assert page.status_code == 200
        assert "Bid Qualification Copilot" in page.text


def test_whoami_shows_the_real_signed_in_user_only_when_azure_provides_it():
    with client() as c:
        assert c.get("/api/whoami").json()["signed_in_as"] is None  # locally: no sign-in
        header = {"X-MS-CLIENT-PRINCIPAL-NAME": "reviewer@contoso.example"}
        assert c.get("/api/whoami", headers=header).json()["signed_in_as"] == \
            "reviewer@contoso.example"


def test_client_chosen_bid_ids_are_validated_and_unique():
    with client() as c:
        bad = c.post("/api/bids", json={"persona_id": "mia", "run_id": "../../etc"})
        assert bad.status_code == 422
        ok = c.post("/api/bids", json={"persona_id": "mia", "run_id": "0123456789ab"})
        assert ok.status_code == 200 and ok.json()["run_id"] == "0123456789ab"
        again = c.post("/api/bids", json={"persona_id": "mia", "run_id": "0123456789ab"})
        assert again.status_code == 409
