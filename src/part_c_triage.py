from pathlib import Path
import json
import random
import re
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mock_llm

RAW = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

tickets = pd.read_csv(RAW / "tickets.csv", dtype=str, keep_default_na=False)
segments_path = OUT / "customer_segments.csv"
segments = pd.read_csv(segments_path)
segment_lookup = segments.set_index("customer_id")["segment_name"].to_dict()
premium_lookup = segments.set_index("customer_id")["is_premium"].astype(bool).to_dict()

ALLOWED = {"billing", "delivery", "technical", "fraud"}
BASE_DELAY = 0.05
MAX_RETRIES = 3
TOKEN_BUDGET = 6000

ZERO_SYSTEM = (
    "CLASSIFY each support ticket into exactly one allowed category: "
    "billing, delivery, technical, fraud. Return JSON only with keys "
    '"category" and "reason", for example '
    '{"category":"billing","reason":"brief explanation"}.'
)
FEW_SYSTEM = ZERO_SYSTEM + """
EXAMPLES:
Ticket: I see an unauthorized payment and my account was hacked.
{"category":"fraud","reason":"unauthorized activity"}
Ticket: The rider is late and my delivery has not arrived.
{"category":"delivery","reason":"late delivery"}
Ticket: The app crashes when I try to log in.
{"category":"technical","reason":"app problem"}
Ticket: I was charged twice on my invoice.
{"category":"billing","reason":"duplicate charge"}
"""

def redact(text):
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[EMAIL]", text)
    text = re.sub(r"\b[6-9]\d{9}\b", "[PHONE]", text)
    text = re.sub(r"ORD-\d{6}", "[ORDER_ID]", text)
    return text

def count_unredacted(text):
    patterns = [
        r"[\w.+-]+@[\w-]+\.[\w.]+",
        r"\b[6-9]\d{9}\b",
        r"ORD-\d{6}",
    ]
    return sum(len(re.findall(pattern, text)) for pattern in patterns)

def make_call(messages, ticket_id, retry_log):
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = mock_llm.chat(messages)
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", 0) if usage else 0
            message = response.choices[0].message
            return message.content, int(tokens)
        except mock_llm.TransientLLMError:
            if attempt >= MAX_RETRIES:
                return None, 0
            delay = BASE_DELAY * (2 ** attempt) * random.uniform(0.9, 1.15)
            retry_log.append({
                "ticket_id": ticket_id,
                "retry_number": attempt + 1,
                "wait_seconds": round(delay, 4),
            })
            time.sleep(delay)
    return None, 0

def classify_output(content):
    if not content:
        return None, "flagged_invalid_output"

    try:
        obj = json.loads(content)
        category = obj.get("category")
        if category in ALLOWED:
            return category, "classified_json"
    except (json.JSONDecodeError, AttributeError, TypeError):
        pass

    words = re.findall(r"[a-z]+", content.lower())
    matches = [word for word in words if word in ALLOWED]
    if len(matches) == 1:
        return matches[0], "classified_keyword"
    return None, "flagged_invalid_output"

def eligible_ticket(row):
    text = row["text"]
    if not text.strip():
        return "skipped_empty"
    if mock_llm.est_tokens(text) > 400:
        return "flagged_long"
    return None

def run_classification(system_prompt, budget=None):
    mock_llm.reset()
    random.seed(42)
    rows = []
    retries = []
    token_total = 0

    for _, row in tickets.iterrows():
        ticket_id = row["ticket_id"]
        original = row["text"]
        redacted = redact(original)

        # The LLM must never receive the unredacted ticket text.
        assert count_unredacted(redacted) == 0

        status = eligible_ticket(row)
        category = ""
        tokens_used = 0
        content = None

        if status is None:
            if budget is not None and token_total >= budget:
                status = "budget_exhausted"
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": redacted},
                ]
                content, tokens_used = make_call(messages, ticket_id, retries)
                token_total += tokens_used
                category, status = classify_output(content)

        customer_id = row["customer_id"]
        rows.append({
            "ticket_id": ticket_id,
            "customer_id": customer_id,
            "status": status,
            "category": category or "",
            "tokens_used": tokens_used,
            "redacted_text": redacted,
        })

    return pd.DataFrame(rows), token_total, retries

def score_results(result, label):
    truth = tickets.set_index("ticket_id")["true_category"]
    scored = result.loc[
        result["status"].isin(["classified_json", "classified_keyword"])
        & result["category"].ne("")
    ].copy()
    scored["true_category"] = scored["ticket_id"].map(truth)
    correct = scored["category"].eq(scored["true_category"]).sum()
    print(f"\n{label}")
    print("Classified:", len(scored), "of", len(tickets))
    print("Correct:", int(correct))
    print("Accuracy among classified:", round(correct / len(scored), 4) if len(scored) else 0)
    print("Tokens:", int(result["tokens_used"].sum()))
    print("Tokens per classified ticket:",
          round(result["tokens_used"].sum() / len(scored), 2) if len(scored) else 0)
    print("Statuses:")
    print(result["status"].value_counts().to_string())
    print("Confusion table (true rows, predicted columns):")
    print(pd.crosstab(scored["true_category"], scored["category"]).to_string())

# First run: zero-shot, no token budget.
zero, zero_tokens, zero_retries = run_classification(ZERO_SYSTEM)
score_results(zero, "ZERO-SHOT RUN")
print("Retry events:", len(zero_retries))
for item in zero_retries:
    print("Retry:", item)

# Second run: few-shot, no token budget.
few, few_tokens, few_retries = run_classification(FEW_SYSTEM)
score_results(few, "FEW-SHOT RUN")
print("Retry events:", len(few_retries))

# Budget demonstration: zero-shot with a deliberately restrictive budget.
budget_result, budget_tokens, budget_retries = run_classification(
    ZERO_SYSTEM, budget=TOKEN_BUDGET
)
budget_path = OUT / "triage_budget_demo.csv"
budget_result.drop(columns=["redacted_text"]).to_csv(budget_path, index=False)
print("\nBUDGET DEMO")
print("Budget:", TOKEN_BUDGET)
print("Tokens consumed:", budget_tokens)
print("Budget-exhausted tickets:",
      int(budget_result["status"].eq("budget_exhausted").sum()))
print("Saved:", budget_path)

# Final deliverable: few-shot results, premium routing, and priority.
final = few.drop(columns=["redacted_text"]).copy()
final["segment_name"] = final["customer_id"].map(segment_lookup).fillna("")
final["priority"] = "NORMAL"
final["route"] = "support_queue"

for index, row in final.iterrows():
    status = row["status"]
    customer_id = row["customer_id"]
    category = row["category"]

    if status in {
        "skipped_empty", "flagged_long", "flagged_invalid_output",
        "budget_exhausted",
    }:
        final.at[index, "route"] = "human_review"
    elif category == "fraud":
        final.at[index, "priority"] = "HIGH"
        final.at[index, "route"] = "security_team"
    elif premium_lookup.get(customer_id, False):
        final.at[index, "priority"] = "HIGH"
        final.at[index, "route"] = "premium_desk"

final_path = OUT / "triage_results.csv"
final.to_csv(final_path, index=False)

retry_path = OUT / "triage_retry_log.jsonl"
with retry_path.open("w", encoding="utf-8") as file:
    for event in few_retries:
        file.write(json.dumps(event) + "\n")

# Check redaction behavior on every ticket.
redacted_all = tickets["text"].map(redact)
assert all(count_unredacted(text) == 0 for text in redacted_all)
assert len(final) == len(tickets)
assert len(budget_result) == len(tickets)

print("\nFINAL TRIAGE OUTPUT")
print(final["route"].value_counts().to_string())
print("Final status counts:")
print(final["status"].value_counts().to_string())
print("Saved:", final_path)
print("Retry log:", retry_path)
print("\nTRIAGE CHECKS PASSED")
