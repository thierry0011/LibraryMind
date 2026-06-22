"""
LibraryMind Smoke Test Script
Runs through every test scenario from the Part 8 requirements.
Usage:  python scripts/smoke_test.py [BASE_URL]
Default BASE_URL: http://localhost:8000
"""

import sys
import json
import time
import httpx

BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:8000"
client = httpx.Client(base_url=BASE_URL, timeout=30.0)

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

results = []


def check(label: str, passed: bool, detail: str = ""):
    symbol = PASS if passed else FAIL
    msg = f"{symbol} {label}"
    if detail:
        msg += f" — {detail}"
    print(msg)
    results.append((label, passed))


def post(path, body):
    r = client.post(path, json=body)
    return r


def get(path):
    r = client.get(path)
    return r


print(f"\n{INFO} LibraryMind Smoke Test  →  {BASE_URL}\n")

# ---------------------------------------------------------------------------
# 1. Health check
# ---------------------------------------------------------------------------
print("── Health ──────────────────────────────────────────────────")
r = get("/health/")
check("GET /health/ returns 200", r.status_code == 200)
data = r.json()
check("Health response has 'status'", "status" in data)
check("Health status is 'ok'", data.get("status") == "ok")

# ---------------------------------------------------------------------------
# 2. Search: "desert planet adventure"
# ---------------------------------------------------------------------------
print("\n── Search /search/books ────────────────────────────────────")
r = post("/search/books", {"query": "desert planet adventure", "top_k": 5})
check("POST /search/books returns 200", r.status_code == 200)
data = r.json()
check("Response has 'results' list", isinstance(data.get("results"), list))
check("At least one result returned", len(data.get("results", [])) > 0)
if data.get("results"):
    first = data["results"][0]
    check("Result has title, author, similarity", all(k in first for k in ("title", "author", "similarity")))
    genres = [b.get("genre", "") for b in data["results"]]
    has_scifi = any("fiction" in (g or "").lower() or "sci" in (g or "").lower() for g in genres)
    check("Sci-Fi books appear for desert planet query", has_scifi, f"genres returned: {genres}")

# ---------------------------------------------------------------------------
# 3. Ask: off-topic → polite refusal
# ---------------------------------------------------------------------------
print("\n── RAG Q&A /search/ask ─────────────────────────────────────")
r = post("/search/ask", {"question": "What is the meaning of life?"})
check("POST /search/ask returns 200", r.status_code == 200)
data = r.json()
check("Response has 'answer' key", "answer" in data)
check("Response has 'sources' key", "sources" in data)
check("Response has 'cached' flag", "cached" in data)
answer_lower = data.get("answer", "").lower()
check(
    "Off-topic question returns polite refusal (no hallucination)",
    any(w in answer_lower for w in ("not found", "catalogue", "catalog", "couldn't find", "cannot find", "don't have")),
    f"answer: {data.get('answer', '')[:100]}"
)

# Ask a relevant question
r = post("/search/ask", {"question": "Recommend a classic romance novel"})
check("POST /search/ask 200 for valid question", r.status_code == 200)
data = r.json()
check("Grounded answer has sources", len(data.get("sources", [])) > 0)

# Cache hit test — same question twice
r1_time = time.time()
post("/search/ask", {"question": "What science fiction books do you have about space?"})
r1_elapsed = time.time() - r1_time

r2_time = time.time()
r2 = post("/search/ask", {"question": "What science fiction books do you have about space?"})
r2_elapsed = time.time() - r2_time

check("Second identical ask returns cached=True", r2.json().get("cached") is True)
check("Second call is faster than first (cache hit)", r2_elapsed < r1_elapsed, f"{r1_elapsed:.2f}s → {r2_elapsed:.2f}s")

# ---------------------------------------------------------------------------
# 4. Chat — multi-turn memory
# ---------------------------------------------------------------------------
print("\n── Chat /chat/ ─────────────────────────────────────────────")
conv_id = "smoke-test-conv-001"

r = post("/chat/", {"conversation_id": conv_id, "message": "Recommend a thriller novel"})
check("POST /chat/ turn 1 returns 200", r.status_code == 200)
data = r.json()
check("Chat reply is non-empty", bool(data.get("reply", "").strip()))
check("Chat response has conversation_id", data.get("conversation_id") == conv_id)

r = post("/chat/", {"conversation_id": conv_id, "message": "Tell me more about that one"})
check("POST /chat/ turn 2 returns 200", r.status_code == 200)
data = r.json()
check("Follow-up reply is non-empty (memory works)", bool(data.get("reply", "").strip()))

r = get(f"/chat/{conv_id}/history")
check("GET /chat/{id}/history returns 200", r.status_code == 200)
history = r.json().get("history", [])
check("History has 4 messages after 2 turns", len(history) == 4, f"got {len(history)}")

r2 = post("/chat/", {"conversation_id": "smoke-test-conv-002", "message": "Hello"})
check("Different conversation_id starts fresh", r2.status_code == 200)
r_hist2 = get("/chat/smoke-test-conv-002/history")
check("Separate conversation has independent history", len(r_hist2.json().get("history", [])) == 2)

r_del = client.delete(f"/chat/{conv_id}")
check("DELETE /chat/{id} returns 204", r_del.status_code == 204)
r_cleared = get(f"/chat/{conv_id}/history")
check("History empty after delete", len(r_cleared.json().get("history", [])) == 0)

# ---------------------------------------------------------------------------
# 5. Classify
# ---------------------------------------------------------------------------
print("\n── Classification /classify/ticket ─────────────────────────")
r = post("/classify/ticket", {"ticket": "My library card isn't working at the self-checkout and I'm very frustrated!"})
check("POST /classify/ticket returns 200", r.status_code == 200)
data = r.json()
check("All five fields present", all(k in data for k in ("category", "priority", "sentiment", "department", "summary")))
check("category=technical", data.get("category") == "technical", f"got {data.get('category')}")
check("priority=high or urgent", data.get("priority") in ("high", "urgent"), f"got {data.get('priority')}")
check("sentiment=negative", data.get("sentiment") == "negative", f"got {data.get('sentiment')}")

r = post("/classify/ticket", {"ticket": "I love the new reading room, thank you so much!"})
check("Positive ticket returns sentiment=positive", r.json().get("sentiment") == "positive")

# ---------------------------------------------------------------------------
# 6. Summarise
# ---------------------------------------------------------------------------
print("\n── Summarisation /summarise/reviews ────────────────────────")
reviews = [
    "This book completely blew me away. The world-building is extraordinary and the characters feel real.",
    "I found the pacing a bit slow in the middle, but the ending was satisfying. Beautiful writing overall.",
    "Couldn't put it down! Some dialogue felt forced but the plot twists kept me hooked.",
    "Mixed feelings — great concept but the execution could have been tighter.",
    "A masterpiece of the genre. Minor quibbles with one subplot but otherwise flawless.",
]
r = post("/summarise/reviews", {"reviews": reviews})
check("POST /summarise/reviews returns 200", r.status_code == 200)
data = r.json()
check("All six fields present", all(k in data for k in ("overall_sentiment", "average_rating", "key_themes", "praise", "criticism", "recommendation")))
check("average_rating is between 1 and 5", 1.0 <= data.get("average_rating", 0) <= 5.0, f"got {data.get('average_rating')}")
check("key_themes is non-empty list", isinstance(data.get("key_themes"), list) and len(data["key_themes"]) > 0)
check("praise is non-empty list", isinstance(data.get("praise"), list) and len(data["praise"]) > 0)
check("criticism is non-empty list", isinstance(data.get("criticism"), list) and len(data["criticism"]) > 0)

# ---------------------------------------------------------------------------
# 7. Books ingest
# ---------------------------------------------------------------------------
print("\n── Knowledge Base /books/ ──────────────────────────────────")
r = post("/books/", {
    "title": "Smoke Test Book",
    "author": "Test Author",
    "year": "2024",
    "genre": "Test",
    "description": "A test book created by the smoke test script to verify the ingest endpoint works correctly.",
})
check("POST /books/ returns 201", r.status_code == 201, f"got {r.status_code}")
check("Response has 'id'", "id" in r.json())

# ---------------------------------------------------------------------------
# 8. Input validation (422 on bad input)
# ---------------------------------------------------------------------------
print("\n── Validation errors ───────────────────────────────────────")
r = post("/search/books", {"query": ""})
check("Empty query returns 422", r.status_code == 422)
r = post("/classify/ticket", {"ticket": ""})
check("Empty ticket returns 422", r.status_code == 422)
r = post("/summarise/reviews", {"reviews": []})
check("Empty reviews list returns 422", r.status_code == 422)
r = post("/chat/", {"conversation_id": "", "message": "hi"})
check("Empty conversation_id returns 422", r.status_code == 422)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n── Results ─────────────────────────────────────────────────")
passed = sum(1 for _, ok in results if ok)
total = len(results)
print(f"\n{passed}/{total} checks passed.\n")
if passed < total:
    print("Failed checks:")
    for label, ok in results:
        if not ok:
            print(f"  {FAIL} {label}")
    sys.exit(1)
else:
    print(f"{PASS} All smoke tests passed!")
