import json
import sys
import time
import urllib.request

BASE = "http://localhost:8000"


def wait_for_health(retries=15, delay=2):
    for _ in range(retries):
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                if json.load(r) == {"status": "ok"}:
                    return True
        except Exception:
            pass
        time.sleep(delay)
    return False


def main():
    if not wait_for_health():
        sys.exit("SMOKE TEST FAILED: /health did not respond")

    req = urllib.request.Request(
        f"{BASE}/predict",
        data=json.dumps({"values": [5.1, 3.5, 1.4, 0.2]}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        result = json.load(r)
    if result != {"prediction": 0}:
        sys.exit(f"SMOKE TEST FAILED: unexpected prediction {result}")

    with urllib.request.urlopen(f"{BASE}/model-info", timeout=5) as r:
        info = json.load(r)
    print(f"SMOKE TEST PASSED: model v{info.get('version')} accuracy={info.get('accuracy')}")


if __name__ == "__main__":
    main()