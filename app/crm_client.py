"""The copilot's CRM client: bounded retries, made safe by an idempotency key.

Retrying a write is only safe if the server can recognise the retry. Every attempt for one
registration sends the SAME Idempotency-Key, so a retry after a lost response gets back the
record that was already created instead of creating a duplicate.
"""

import time

import httpx


class CrmError(RuntimeError):
    """The CRM rejected the registration (4xx), or was still failing after the last attempt."""


def _detail(response: httpx.Response) -> str:
    """FastAPI-style services explain errors in {"detail": ...}; otherwise use the raw body."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    return str(body.get("detail", body)) if isinstance(body, dict) else str(body)


class CrmClient:
    def __init__(
        self,
        base_url: str,
        http: httpx.Client | None = None,
        max_attempts: int = 3,
        backoff_seconds: float = 0.2,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._url = base_url.rstrip("/") + "/opportunities"
        # Tests inject FastAPI's TestClient here. The timeout: never hang on a stuck CRM.
        self._http = http or httpx.Client(timeout=5.0)
        self._max_attempts = max_attempts
        self._backoff_seconds = backoff_seconds

    def register(
        self, payload: dict, idempotency_key: str, fault: str | None = None
    ) -> tuple[dict, int]:
        """Create the CRM record for one approved bid. Returns (record, attempts used)."""
        problem = ""
        for attempt in range(1, self._max_attempts + 1):  # range(1, 4) = 1, 2, 3
            headers = {"Idempotency-Key": idempotency_key}  # the same key on every attempt
            if fault and attempt == 1:
                headers["X-Mock-Fault"] = fault  # demo only: make the first attempt fail
            try:
                response = self._http.post(self._url, json=payload, headers=headers)
            except httpx.TransportError as exc:  # includes httpx.TimeoutException
                # Unknown outcome: the CRM may have saved the record. The key makes a retry safe.
                problem = f"{type(exc).__name__}: {exc}"
            else:  # Python note: a try's `else` runs only if the `try` block raised no error
                if response.is_success:
                    return response.json(), attempt
                if response.status_code < 500:
                    # 4xx: the request itself is wrong, so sending it again cannot help.
                    raise CrmError(
                        f"CRM rejected the registration ({response.status_code}): "
                        f"{_detail(response)}"
                    )
                problem = f"HTTP {response.status_code}: {_detail(response)}"
            if attempt < self._max_attempts:
                # Exponential backoff (0.2 s, 0.4 s, ...) gives a struggling CRM time to recover.
                time.sleep(self._backoff_seconds * 2 ** (attempt - 1))  # `**` = to the power of
        raise CrmError(f"CRM still failing after {self._max_attempts} attempts ({problem})")
