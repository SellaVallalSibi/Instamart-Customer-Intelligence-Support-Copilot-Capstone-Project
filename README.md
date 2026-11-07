# Instamart Customer Intelligence & Support Copilot

## Overview

This capstone combines customer analytics, machine learning, support-ticket triage, policy retrieval, and a tool-using refund agent. It runs locally using Python, SQLite, scikit-learn, and the supplied offline `mock_llm.py`. No hosted LLM or API key is required.

## Reproduce

Run these commands from the repository root in PowerShell. Use Python 3.11 or newer.

1. Create a virtual environment:

```powershell
python -m venv .venv
```

2. Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

3. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Generate the supplied input data:

```powershell
python .\generate_data.py
```

5. Clean the data and build the SQLite database:

```powershell
python .\src\part_a.py
```

6. Run Part A analysis and validation:

```powershell
python .\src\part_a_analysis.py
```

7. Execute the SQL queries:

```powershell
python -c "import sqlite3; c=sqlite3.connect('data/instamart.db'); s=open('sql/queries.sql',encoding='utf-8').read(); [print('Query',i+1,'rows:',len(c.execute(q).fetchall())) for i,q in enumerate(s.split(';')) if q.strip() and not q.strip().startswith('--')]; c.close()"
```

8. Build customer segments:

```powershell
python .\src\part_b_segments.py
```

9. Train and evaluate the lapse model:

```powershell
python .\src\part_b_lapse.py
```

10. Run support-ticket triage:

```powershell
python .\src\part_c_triage.py
```

11. Evaluate policy retrieval:

```powershell
python .\src\part_d_rag.py
```

12. Run the refund agent and checkpoint/resume demonstrations:

```powershell
python .\src\part_d_agent.py
```

The supplied generate_data.py and mock_llm.py are used without modification. Outputs are written beneath data/, outputs/, and state/.

## Part A
Data cleaning and database
- Cleaned customers, products, and orders into CSV files.
- Normalized city and gender values; validated and imputed customer ages.
- Removed duplicate order records and retained the latest record per order ID.
- Converted numeric fields, derived missing order amounts where possible, and imputed missing delivery times for completed orders.
- Preserved cancelled-order delivery times as missing.
- Created SQLite tables for customers, products, and orders.

Validation results:

|       Check             |    Result         |
|-------------------------|-------------------|
|Customers                |    1,200          |
|Products                 |       40          |
|Unique cleaned orders    |   12,309          |
|Completed orders         |   11,110          |
|Cancelled orders         |    1,199          |
|Pending orders remaining |        0          |
|Completed-order revenue  | INR 2,998,668.80  |


SQL and exploratory analysis:
sql/queries.sql contains seven queries covering customer order thresholds, revenue by city and category, above-average product prices, customers without completed orders, above-average spend, ranking functions, and cumulative daily revenue.

The seven queries executed successfully during validation. SQL revenue totals matched the corresponding pandas aggregation.

Completed-order revenue was highest in Hyderabad at INR 528,589.90, followed by Bengaluru at INR 523,643.40. Order amounts were right-skewed, with a small number of high-value orders.


## Part B
Customer segmentation:
src/part_b_segments.py calculates recency, frequency, and monetary value from completed orders, scales the features, and evaluates K-means values from K=2 through K=8. The selected K=4 solution produces these segments:

|Segment  |  Customers |   Mean recency (days) |  Mean frequency |  Mean monetary value (INR)  |
|---------|------------|-----------------------|-----------------|-----------------------------|
|Regular  |     436    |         28.86         |        3.77     |           1,057.62          |
|Premium  |     138    |         24.12         |        6.77     |           7,129.70          |
|Loyal    |     419    |          8.77         |       19.40     |           3,410.02          |
|At-Risk  |     182    |        109.04         |        2.20     |             685.99          |

PCA explains approximately 85.2% of the standardized feature variance in its first two components.

Output: outputs/customer_segments.csv.

Lapse prediction:
The lapse model uses pre-cutoff customer behavior to predict whether a customer has no completed order in September 2025. The data is split into stratified training and test sets, with preprocessing fitted on training data only. Logistic Regression and Random Forest are tuned using cross-validation.

|Metric                 | Logistic-Regression |  Random Forest |
|-----------------------|---------------------|----------------|
|Test accuracy          |        0.763        |      0.772     |
|Lapsed-class precision |        0.655        |      0.680     |
|Lapsed-class recall    |        0.809        |      0.764     |
|Lapsed-class F1        |        0.724        |      0.720     |

The baseline that predicts every customer will stay has 61.6% accuracy and zero F1 for the lapsed class. Logistic Regression was selected based on the measured test F1.

Outputs:
- outputs/lapse_model.joblib
- outputs/retention_list.csv
The retention list ranks Premium customers by predicted lapse probability and includes their actual lapse label for evaluation. Because the Premium label is derived from the full-window segmentation, this list is illustrative and is not a fully leakage-free evaluation of a real deployment campaign.

## Part C
The ticket triage pipeline redacts email addresses, phone numbers, and order IDs before sending ticket text to the offline mock model. It tests zero-shot and few-shot prompts, validates JSON responses, uses a keyword fallback for some invalid outputs, retries transient errors, and routes tickets according to category and customer segment.

|Measure                            |  Zero-shot   |  Few-shot  |
|-----------------------------------|--------------|------------|
|Classified tickets                 |   147 / 164  |  147 / 164 |
|Correct classified tickets         |   126 / 147  |  147 / 147 |
|Accuracy among classified tickets  |    85.71%    |    100%    |
|Tokens used                        |    9,556     |   19,540   |
|Retry events                       |      19      |     19     |

The final triage statuses were 135 valid JSON classifications, 12 keyword classifications, 9 invalid-output flags, 4 empty-ticket skips, and 4 long-ticket flags.

The budget demonstration uses a nominal 6,000-token threshold and marks subsequent tickets as budget-exhausted after the running total reaches the threshold. The observed total was 6,017 tokens because the budget is checked before a call; a single call can exceed the remaining budget.

Outputs:
- outputs/triage_results.csv
- outputs/triage_budget_demo.csv
- outputs/triage_retry_log.jsonl
These accuracy figures measure performance against this deterministic mock model and supplied synthetic tickets. They should not be interpreted as real-world LLM performance.

## Part D
Policy retrieval:
The five policy documents are split into 40-word chunks with a 10-word overlap. The implementation uses TfidfVectorizer and cosine similarity to retrieve the top three chunks. The offline policy assistant answers from the retrieved context and returns a fixed fallback when the answer is not found.

Validation:
- 15 total chunks across five documents.
- All three results for the query refund policy late delivery came from refund_policy.txt.
- 12 of 12 answerable questions passed the combined document and expected-fact checks.
- Both unanswerable questions triggered the expected fallback.
- The current no-overlap experiment retrieved the expected document for 12 of 12 answerable questions; it does not demonstrate a document-retrieval failure from removing overlap.

Output: outputs/policy_eval_results.csv.

Refund agent
The ReAct-style agent uses five tools:
1. get_order — looks up an order in SQLite.
2. search_policy — retrieves policy chunks.
3. calculate_refund — calculates full refunds for delivery times exceeding 45 minutes.
4. issue_refund — records an approved refund.
5. escalate_to_human — records an escalation for a refund above INR 1,000.

All seven supplied scenarios produced their expected outcome:

|Scenario |  Expected  |      Observed             |
|---------|------------|---------------------------|
|  S1     |   REFUND   |       REFUND              |
|  S2     |   REFUND   |       REFUND              |
|  S3     |  ESCALATE  |      ESCALATE             |
|  S4     |   DENIED   |       DENIED              |
|  S5     |   DENIED   |       DENIED              |
|  S6     |  NOT_FOUND |      NOT_FOUND            |
|  S7     |  FALLBACK  |   FALLBACK_MAX_ITERATIONS |

The stop-condition tests produced FALLBACK_TOKEN_LIMIT and FALLBACK_TIMEOUT. The loop scenario reached FALLBACK_MAX_ITERATIONS.

For crash recovery, the S1 run checkpointed after get_order and search_policy, then resumed with calculate_refund and issue_refund. The final answer was REFUND: INR 128.25 issued. No earlier tool was repeated after resuming.

Outputs and state:
- outputs/agent_runs.jsonl
- outputs/refund_ledger.jsonl
- state/
The mock model is deterministic and intentionally simplified. The agent is an educational workflow demonstration, not a production payment system.

## Limitations and Next Steps
- The project uses synthetic or supplied assessment data and a deterministic mock LLM; results may not generalize to production traffic.
- PII redaction relies on patterns and may miss unusual formats or other sensitive information.
- Customer segment names are interpretations of unsupervised clusters.
- Lapse prediction depends on the chosen lapse definition, feature window, and train/test split. Future work should use time-based validation and monitor leakage.
- The retention list is illustrative and should be evaluated with a leakage-free campaign design before operational use.
- TF-IDF retrieval is a lightweight baseline. Compare it with alternative retrieval methods and evaluate answer-level evidence coverage on a larger test set.
- The agent's refund rules are simplified demonstration logic. Production use would require authorization, idempotency, audit controls, policy versioning, robust exception handling, and human oversight.
- Improve observability, add automated tests, and validate all outputs against explicit acceptance criteria before deployment.
