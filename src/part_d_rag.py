from pathlib import Path
import re
import sys

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
POLICY_DIR = ROOT / "data" / "policies"
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

def chunk_text(text, size=40, overlap=10):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap
    return chunks

documents = {}
for path in sorted(POLICY_DIR.glob("*.txt")):
    documents[path.name] = path.read_text(encoding="utf-8")

chunks = []
for source, text in documents.items():
    for index, chunk in enumerate(chunk_text(text), start=1):
        chunks.append({
            "source": source,
            "chunk_id": f"{source}:{index}",
            "text": chunk,
        })

corpus = [item["text"] for item in chunks]
vectorizer = TfidfVectorizer()
matrix = vectorizer.fit_transform(corpus)

def retrieve(query, k=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()
    order = scores.argsort()[::-1][:k]
    return [
        {
            "source": chunks[i]["source"],
            "score": float(scores[i]),
            "text": chunks[i]["text"],
        }
        for i in order
    ]

def answer(question):
    results = retrieve(question, k=3)
    context = "\n".join(
        f"[{item['source']}]\n{item['text']}" for item in results
    )
    messages = [
        {
            "role": "system",
            "content": (
                "POLICY ASSISTANT. Answer using only the supplied policy "
                "context. If the answer is not present, say exactly: "
                "'I could not find that information in the policy documents.'"
            ),
        },
        {
            "role": "user",
            "content": f"CONTEXT:\n{context}\nQUESTION:\n{question}",
        },
    ]
    response = mock_llm.chat(messages)
    return response.choices[0].message.content, results

sys.path.insert(0, str(ROOT))
import mock_llm

print("POLICY CHUNK COUNTS")
for name in sorted(documents):
    print(name, sum(c["source"] == name for c in chunks))
print("Total chunks:", len(chunks))

print("\nRETRIEVAL DEMO: refund policy late delivery")
for result in retrieve("refund policy late delivery"):
    print(
        result["source"],
        "score=", round(result["score"], 4),
        "|", result["text"][:140],
    )

evaluation = pd.read_csv(RAW_DIR / "policy_eval.csv", keep_default_na=False)
results = []
answerable_correct = 0
answerable_total = 0
unanswerable_correct = 0
unanswerable_total = 0
mock_llm.reset()

for _, row in evaluation.iterrows():
    question = row["question"]
    expected_doc = row["expected_doc"]
    expected_fact = row["expected_fact"]
    response, retrieved = answer(question)
    retrieved_sources = [item["source"] for item in retrieved]

    if expected_doc == "NONE":
        unanswerable_total += 1
        fallback_ok = "I could not find" in response
        unanswerable_correct += int(fallback_ok)
        doc_ok = fallback_ok
        fact_ok = fallback_ok
    else:
        answerable_total += 1
        doc_ok = expected_doc in retrieved_sources
        fact_ok = expected_fact.lower() in response.lower()
        answerable_correct += int(doc_ok and fact_ok)

    results.append({
        "question": question,
        "expected_doc": expected_doc,
        "retrieved_sources": " | ".join(retrieved_sources),
        "answer": response,
        "document_found": doc_ok,
        "expected_fact_found": fact_ok,
    })

result_df = pd.DataFrame(results)
result_df.to_csv(OUT / "policy_eval_results.csv", index=False)

print("\nPOLICY EVALUATION")
print("Answerable document + fact checks:",
      answerable_correct, "/", answerable_total)
print("Unanswerable fallback checks:",
      unanswerable_correct, "/", unanswerable_total)
print("Saved:", OUT / "policy_eval_results.csv")

# Repeat retrieval with no overlap to demonstrate chunk-boundary effects.
no_overlap_chunks = []
for source, text in documents.items():
    for index, chunk in enumerate(chunk_text(text, size=40, overlap=0), start=1):
        no_overlap_chunks.append({
            "source": source,
            "chunk_id": f"{source}:{index}",
            "text": chunk,
        })

no_overlap_vectorizer = TfidfVectorizer()
no_overlap_matrix = no_overlap_vectorizer.fit_transform(
    [item["text"] for item in no_overlap_chunks]
)

def retrieve_no_overlap(query, k=3):
    qv = no_overlap_vectorizer.transform([query])
    scores = cosine_similarity(qv, no_overlap_matrix).ravel()
    order = scores.argsort()[::-1][:k]
    return [no_overlap_chunks[i] for i in order]

no_overlap_hits = 0
for _, row in evaluation.iterrows():
    if row["expected_doc"] == "NONE":
        continue
    sources = [
        item["source"] for item in retrieve_no_overlap(row["question"], k=3)
    ]
    no_overlap_hits += int(row["expected_doc"] in sources)

print("No-overlap expected-document retrieval:",
      no_overlap_hits, "/", answerable_total)

print("\nPOLICY RETRIEVAL SCRIPT FINISHED")
