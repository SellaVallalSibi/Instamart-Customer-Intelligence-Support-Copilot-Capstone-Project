from pathlib import Path
from datetime import datetime, timezone
import json
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import mock_llm
from part_d_rag import retrieve

DB_PATH = ROOT / "data" / "instamart.db"
OUT = ROOT / "outputs"
STATE = ROOT / "state"
OUT.mkdir(exist_ok=True, parents=True)
STATE.mkdir(exist_ok=True, parents=True)

LEDGER = OUT / "refund_ledger.jsonl"
RUNS = OUT / "agent_runs.jsonl"

TOOLS_SPEC = [
    {"type": "function", "function": {
        "name": "get_order",
        "description": "Look up an order by order ID.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {
        "name": "search_policy",
        "description": "Retrieve relevant policy text.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"}}, "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "calculate_refund",
        "description": "Calculate refund based on order amount and delivery time.",
        "parameters": {"type": "object", "properties": {
            "order_amount": {"type": "number"},
            "delivery_minutes": {"type": ["number", "null"]}},
            "required": ["order_amount", "delivery_minutes"]}}},
    {"type": "function", "function": {
        "name": "issue_refund",
        "description": "Issue a permitted refund and record it.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}},
            "required": ["order_id", "refund_inr"]}}},
    {"type": "function", "function": {
        "name": "escalate_to_human",
        "description": "Send a large refund to a human lead.",
        "parameters": {"type": "object", "properties": {
            "order_id": {"type": "string"},
            "refund_inr": {"type": "number"}},
            "required": ["order_id", "refund_inr"]}}},
]

def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def get_order(order_id):
    import sqlite3
    with sqlite3.connect(DB_PATH) as con:
        row = con.execute("""
            SELECT order_id, status, order_amount, delivery_minutes
            FROM orders WHERE order_id = ?
        """, (order_id,)).fetchone()
    if row is None:
        return {"error": "order not found", "order_id": order_id}
    return {
        "order_id": row[0],
        "status": row[1],
        "order_amount": float(row[2]),
        "delivery_minutes": (
            None if row[3] is None else float(row[3])
        ),
    }

def search_policy(query):
    results = retrieve(query, k=3)
    return {"results": [
        {"source": x["source"], "score": round(x["score"], 4), "text": x["text"]}
        for x in results
    ]}

def calculate_refund(order_amount, delivery_minutes):
    amount = float(order_amount)
    eligible = delivery_minutes is not None and float(delivery_minutes) > 45
    return {
        "refund_inr": round(amount, 2) if eligible else 0.0,
        "eligible": eligible,
        "rule": "full refund when delivery exceeds 45 minutes",
    }

def ledger_action(tool_name, order_id, refund_inr):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": round(float(refund_inr), 2),
    }
    append_jsonl(LEDGER, record)
    return {"success": True, **record}

def issue_refund(order_id, refund_inr):
    return ledger_action("issue_refund", order_id, refund_inr)

def escalate_to_human(order_id, refund_inr):
    return ledger_action("escalate_to_human", order_id, refund_inr)

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}

def fallback(status, message):
    return {
        "status": status,
        "final_answer": message,
    }

def save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, status):
    checkpoint = {
        "run_id": run_id,
        "scenario_id": scenario_id,
        "messages": messages,
        "tokens_used": tokens_used,
        "steps": steps,
        "status": status,
    }
    (STATE / f"{run_id}.json").write_text(
        json.dumps(checkpoint, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def load_checkpoint(run_id):
    path = STATE / f"{run_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))

def run_agent(
    user_message,
    scenario_id,
    max_iterations=6,
    token_limit=6000,
    time_limit_seconds=10,
    crash_after_tools=None,
    run_id=None,
    resume_data=None,
):
    started = time.monotonic()
    run_id = run_id or f"{scenario_id}_{uuid.uuid4().hex[:8]}"

    if resume_data is None:
        messages = [
            {
                "role": "system",
                "content": (
                    "AGENT. Use tools to look up orders, search policy, "
                    "calculate refunds, issue eligible refunds, or escalate "
                    "refunds above INR 1000. Never invent order data."
                ),
            },
            {"role": "user", "content": user_message},
        ]
        tokens_used = 0
        steps = []
    else:
        messages = resume_data["messages"]
        tokens_used = resume_data["tokens_used"]
        steps = resume_data["steps"]

    tools_executed_this_run = 0

    while len([s for s in steps if s.get("type") == "model_call"]) < max_iterations:
        if time.monotonic() - started >= time_limit_seconds:
            result = fallback(
                "FALLBACK_TIMEOUT",
                "I’m handing this request to a human support agent because the time limit was reached.",
            )
            save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, result["status"])
            return {**result, "run_id": run_id, "steps": steps,
                    "tokens_used": tokens_used, "llm_calls": sum(
                        s.get("type") == "model_call" for s in steps)}

        response = mock_llm.chat(messages, tools=TOOLS_SPEC)
        call_tokens = int(response.usage.total_tokens)
        tokens_used += call_tokens
        steps.append({
            "type": "model_call",
            "tokens": call_tokens,
            "finish_reason": response.choices[0].finish_reason,
        })

        if tokens_used > token_limit:
            result = fallback(
                "FALLBACK_TOKEN_LIMIT",
                "I’m handing this request to a human support agent because the token limit was reached.",
            )
            save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, result["status"])
            return {**result, "run_id": run_id, "steps": steps,
                    "tokens_used": tokens_used, "llm_calls": sum(
                        s.get("type") == "model_call" for s in steps)}

        message = response.choices[0].message
        content = message.content or ""
        tool_calls = message.tool_calls or []

        if not tool_calls:
            status = content.split(":", 1)[0].strip().upper()
            result = {"status": status, "final_answer": content}
            messages.append({"role": "assistant", "content": content})
            save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, status)
            return {**result, "run_id": run_id, "steps": steps,
                    "tokens_used": tokens_used, "llm_calls": sum(
                        s.get("type") == "model_call" for s in steps)}

        assistant_calls = []
        for call in tool_calls:
            assistant_calls.append({
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                },
            })
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": assistant_calls,
        })

        for call in tool_calls:
            name = call.name
            args = call.arguments
            if isinstance(args, str):
                args = json.loads(args)

            if name not in TOOL_MAP:
                observation = {"error": f"unknown tool: {name}"}
            else:
                observation = TOOL_MAP[name](**args)

            steps.append({
                "type": "tool",
                "tool": name,
                "arguments": args,
                "observation": observation,
            })
            tools_executed_this_run += 1

            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "name": name,
                "content": json.dumps(observation, ensure_ascii=False),
            })

            # Persist immediately after every tool observation.
            save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, "RUNNING")

            if crash_after_tools is not None and tools_executed_this_run >= crash_after_tools:
                raise RuntimeError("SIMULATED_CRASH_AFTER_CHECKPOINT")

    result = fallback(
        "FALLBACK_MAX_ITERATIONS",
        "I’m handing this request to a human support agent because the step limit was reached.",
    )
    save_checkpoint(run_id, scenario_id, messages, tokens_used, steps, result["status"])
    return {**result, "run_id": run_id, "steps": steps,
            "tokens_used": tokens_used, "llm_calls": sum(
                s.get("type") == "model_call" for s in steps)}

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    saved = load_checkpoint(run_id)
    return run_agent(
        user_message="resume from saved checkpoint",
        scenario_id=saved["scenario_id"],
        max_iterations=max_iterations,
        token_limit=token_limit,
        time_limit_seconds=time_limit_seconds,
        run_id=run_id,
        resume_data=saved,
    )

def main():
    import pandas as pd

    scenarios = pd.read_csv(
        ROOT / "data" / "raw" / "agent_scenarios.csv",
        keep_default_na=False,
    )
    if LEDGER.exists():
        LEDGER.unlink()
    if RUNS.exists():
        RUNS.unlink()

    print("SEVEN AGENT SCENARIOS")
    all_results = []
    for _, row in scenarios.iterrows():
        mock_llm.reset()
        try:
            result = run_agent(row["message"], row["scenario_id"])
        except Exception as exc:
            print("Scenario error:", row["scenario_id"], str(exc))
            raise

        tool_names = [s["tool"] for s in result["steps"] if s.get("type") == "tool"]
        print(
            row["scenario_id"],
            "| expected:", row["expected_outcome"],
            "| actual:", result["status"],
            "| calls:", result["llm_calls"],
            "| tokens:", result["tokens_used"],
            "| tools:", ",".join(tool_names),
        )
        record = {
            "scenario_id": row["scenario_id"],
            "status": result["status"],
            "final_answer": result["final_answer"],
            "llm_calls": result["llm_calls"],
            "tokens_used": result["tokens_used"],
            "steps": result["steps"],
        }
        append_jsonl(RUNS, record)
        all_results.append((row, result))

    print("\nSTOP CONDITION DEMOS")
    mock_llm.reset()
    token_demo = run_agent(
        scenarios.iloc[0]["message"], "TOKEN_DEMO", token_limit=200
    )
    print("Token demo:", token_demo["status"])

    mock_llm.reset()
    timeout_demo = run_agent(
        scenarios.iloc[0]["message"], "TIME_DEMO", time_limit_seconds=0
    )
    print("Timeout demo:", timeout_demo["status"])

    print("\nCHECKPOINT / RESUME DEMO")
    row = scenarios[scenarios["scenario_id"] == "S1"].iloc[0]
    mock_llm.reset()
    resume_run_id = "S1_crash_resume_demo"
    try:
        run_agent(
            row["message"],
            "S1",
            run_id=resume_run_id,
            crash_after_tools=2,
        )
    except RuntimeError as exc:
        print("Simulated crash:", str(exc))
    saved = load_checkpoint(resume_run_id)
    before_tools = [s["tool"] for s in saved["steps"] if s.get("type") == "tool"]
    print("Checkpointed tools:", before_tools)

    resumed = resume_agent(resume_run_id)
    after_tools = [s["tool"] for s in resumed["steps"] if s.get("type") == "tool"]
    print("Resumed status:", resumed["status"])
    print("Final answer:", resumed["final_answer"])
    print("Tools across run + resume:", after_tools)

    assert len(before_tools) == 2
    assert after_tools[:2] == before_tools
    assert after_tools == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
    assert resumed["status"] == "REFUND"
    assert "128.25" in resumed["final_answer"]

    print("\nAGENT SCRIPT FINISHED")

if __name__ == "__main__":
    main()
