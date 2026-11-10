"""Creates a self-contained sample dataset so the project runs out of the box:

  data/*.txt           three FICTIONAL company annual-report excerpts
  data/financials.csv  SYNTHETIC company-year table used to train the predictor
  eval_set.json        questions with known answers, matched to the documents

All companies, people and numbers are invented. Once the pipeline works, swap
data/ for real filings (SEC EDGAR 10-K PDFs are free) and write your own
eval_set.json questions: that is what makes the project yours.

Run:  python generate_sample_data.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

HEADER = "FICTIONAL SAMPLE DOCUMENT - generated for a learning project, not a real filing.\n\n"

LAKESHORE = HEADER + """LAKESHORE ROBOTICS INC.
ANNUAL REPORT - FISCAL YEAR 2025 (EXCERPTS)

Item 1. Business Overview
Lakeshore Robotics Inc. designs and sells industrial robots. The company is headquartered in Chicago, Illinois, and was founded in 2009. Our chief executive officer is Marisol Quenby. As of December 31, 2025, we employed 1,840 people worldwide. We sell through three segments: Warehouse Automation, which generated 61% of revenue; Surgical Assistance, which generated 27% of revenue; and Services, which generated 12% of revenue.

Item 1A. Risk Factors
Supplier concentration. We depend on a single supplier, Hartwell Components, for the servo motors used in nearly every robot we build. Hartwell Components accounted for 38% of our component spending in fiscal 2025. A disruption at this supplier could delay shipments for several quarters, because qualifying a replacement motor design takes roughly nine months.
Regulatory risk. Our Surgical Assistance products require regulatory clearance before sale, and clearance timelines are difficult to predict.
Customer timing. Warehouse customers often delay capital purchases when interest rates rise, which can make our quarterly results uneven.

Item 7. Management's Discussion
Revenue for fiscal 2025 was $412.6 million, an increase of 14.2% from $361.3 million in fiscal 2024. Growth was driven mainly by Warehouse Automation orders from e-commerce fulfillment customers. Cash and cash equivalents were $96.4 million at year end.

Outlook and Capital Plans
In 2026 we plan to open a new manufacturing facility in Reno, Nevada, to add production capacity and reduce shipping times to western customers. We expect capital expenditures of $48.0 million in 2026, most of it for the Reno facility. We also intend to qualify a second servo motor supplier during 2026 to reduce the supplier concentration described above.
"""

PRAIRIE = HEADER + """PRAIRIE HEALTH SYSTEMS
ANNUAL REPORT - FISCAL YEAR 2025 (EXCERPTS)

Item 1. Business Overview
Prairie Health Systems operates hospitals, outpatient clinics and a telehealth service in the Midwest. The company is headquartered in Madison, Wisconsin. Our chief executive officer is Dwight Ostrander. As of December 31, 2025, we employed 7,950 people. Revenue by segment in fiscal 2025 was: Hospitals, 72%; Outpatient Clinics, 20%; and Telehealth, 8%.

Item 1A. Risk Factors
Government reimbursement. Approximately 54% of our revenue in fiscal 2025 came from Medicare and Medicaid. Changes in reimbursement rates set by government payers could reduce our revenue materially, and we have limited ability to offset such changes through pricing.
Labor costs. Competition for nurses and physicians has increased wage costs, and staffing shortages can force us to reduce the number of available beds.
Cybersecurity. Our patient records are an attractive target for attackers, and an incident could disrupt operations and lead to penalties.

Item 7. Management's Discussion
Revenue for fiscal 2025 was $1,208.4 million, an increase of 6.1% from $1,138.9 million in fiscal 2024. Cash and cash equivalents were $310.2 million at year end. Telehealth visits grew faster than any other line of business, though from a small base.

Outlook and Capital Plans
During 2026 we plan to open six new outpatient clinics in suburban markets, because outpatient care costs less to deliver than inpatient care and patients increasingly prefer it. We will also invest in upgrading our electronic health record system.
"""

SUMMIT = HEADER + """SUMMIT CLOUD SOFTWARE, INC.
ANNUAL REPORT - FISCAL YEAR 2025 (EXCERPTS)

Item 1. Business Overview
Summit Cloud Software sells subscription software that helps finance teams automate reporting. The company is headquartered in Austin, Texas. Our chief executive officer is Priyanka Raman-Hale. As of December 31, 2025, we employed 962 people. Subscriptions generated 83% of revenue in fiscal 2025 and professional services generated 17%.

Item 1A. Risk Factors
Customer concentration. Our ten largest customers represented 44% of revenue in fiscal 2025. Losing one or more of them, or a significant reduction in their usage, would harm our results.
Competition. Larger software vendors can bundle competing features at low prices.
Security. A breach of our platform could damage customer trust and lead to contract terminations.

Item 7. Management's Discussion
Revenue for fiscal 2025 was $228.9 million, an increase of 31.5% from $174.1 million in fiscal 2024. Our net revenue retention rate was 118%, meaning customers that were with us a year ago spent 18% more with us this year on average. Cash and cash equivalents were $187.5 million at year end.

Outlook and Capital Plans
We plan to launch an AI assistant product in the second quarter of 2026. It will let finance teams ask questions about their reports in plain language. We expect to hire additional engineers to support this launch.
"""

DOCS = {
    "lakeshore_robotics_10k.txt": LAKESHORE,
    "prairie_health_10k.txt": PRAIRIE,
    "summit_cloud_10k.txt": SUMMIT,
}

# Each question has a known answer. Fields:
#   gold_source / gold_evidence : which file and which text snippet a good retriever should find
#   answer_keywords             : every item must appear in the answer (a list item = any-of)
#   expected_tools / expected_args : what the agent should call (calc questions only)
LK, PR, SC = "lakeshore_robotics_10k.txt", "prairie_health_10k.txt", "summit_cloud_10k.txt"


def q(id, type_, question, **kw):
    return {"id": id, "type": type_, "question": question, **kw}


EVAL_SET = [
    q("f01", "factual", "Where is Lakeshore Robotics headquartered?", gold_source=LK, gold_evidence="Chicago, Illinois", answer_keywords=["chicago"]),
    q("f02", "factual", "What was Lakeshore Robotics' revenue in fiscal 2025?", gold_source=LK, gold_evidence="$412.6 million", answer_keywords=["412.6"]),
    q("f03", "factual", "How many employees does Lakeshore Robotics have?", gold_source=LK, gold_evidence="1,840", answer_keywords=["1840"]),
    q("f04", "factual", "Which supplier does Lakeshore Robotics depend on for servo motors?", gold_source=LK, gold_evidence="Hartwell Components", answer_keywords=["hartwell"]),
    q("f05", "factual", "Where will Lakeshore Robotics open a new facility in 2026?", gold_source=LK, gold_evidence="Reno, Nevada", answer_keywords=["reno"]),
    q("f06", "factual", "Who is the CEO of Prairie Health Systems?", gold_source=PR, gold_evidence="Dwight Ostrander", answer_keywords=["ostrander"]),
    q("f07", "factual", "What share of Prairie Health's revenue came from Medicare and Medicaid?", gold_source=PR, gold_evidence="54%", answer_keywords=["54"]),
    q("f08", "factual", "How many new outpatient clinics does Prairie Health plan to open in 2026?", gold_source=PR, gold_evidence="six new outpatient clinics", answer_keywords=[["six", "6"]]),
    q("f09", "factual", "How much cash did Prairie Health Systems hold at year end?", gold_source=PR, gold_evidence="$310.2 million", answer_keywords=["310.2"]),
    q("f10", "factual", "What percentage of Prairie Health's revenue came from hospitals?", gold_source=PR, gold_evidence="Hospitals, 72%", answer_keywords=["72"]),
    q("f11", "factual", "What was Summit Cloud Software's net revenue retention rate?", gold_source=SC, gold_evidence="118%", answer_keywords=["118"]),
    q("f12", "factual", "Who is the CEO of Summit Cloud Software?", gold_source=SC, gold_evidence="Priyanka Raman-Hale", answer_keywords=["raman-hale"]),
    q("f13", "factual", "How much of Summit's revenue came from its ten largest customers?", gold_source=SC, gold_evidence="44%", answer_keywords=["44"]),
    q("f14", "factual", "When does Summit Cloud plan to launch its AI assistant?", gold_source=SC, gold_evidence="second quarter of 2026", answer_keywords=[["second quarter", "q2"]]),
    q("f15", "factual", "By what percentage did Summit Cloud's revenue grow in fiscal 2025?", gold_source=SC, gold_evidence="31.5%", answer_keywords=["31.5"]),
    q("m01", "multi", "Which of the three companies grew revenue fastest in fiscal 2025, and by how much?", gold_sources=[LK, PR, SC], answer_keywords=["summit", "31.5"]),
    q("m02", "multi", "Which of the three companies has the most employees?", gold_sources=[LK, PR, SC], answer_keywords=["prairie", "7950"]),
    q("u01", "unanswerable", "What dividend per share did Lakeshore Robotics pay in fiscal 2025?"),
    q("u02", "unanswerable", "Who is the chief financial officer of Summit Cloud Software?"),
    q("u03", "unanswerable", "What was Prairie Health Systems' net income in fiscal 2024?"),
    q("c01", "calc", "A company has $300 million in revenue and grew 10% last year. Use the growth predictor to project its growth next year.",
      expected_tools=["predict_growth"], expected_args={"revenue_musd": 300, "prior_growth": 0.10}),
    q("c02", "calc", "Use the growth predictor on Summit Cloud Software's fiscal 2025 revenue and growth rate from its annual report.",
      expected_tools=["search_documents", "predict_growth"], expected_args={"revenue_musd": 228.9, "prior_growth": 0.315}),
    q("c03", "calc", "Based on Lakeshore Robotics' fiscal 2025 results in its report, what growth does the predictor project for next year?",
      expected_tools=["search_documents", "predict_growth"], expected_args={"revenue_musd": 412.6, "prior_growth": 0.142}),
]


def make_financials(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """SYNTHETIC data with a known pattern (growth fades as companies get big,
    and carries over partly from last year) plus noise. The model should learn
    roughly that pattern; the held-out R^2 printed by predictor.py shows how well."""
    rng = np.random.default_rng(seed)
    revenue = rng.lognormal(mean=5.3, sigma=1.0, size=n).round(1)
    prior = rng.uniform(-0.05, 0.40, size=n).round(3)
    nxt = 0.015 + 0.5 * prior - 0.00002 * revenue + rng.normal(0, 0.02, size=n)
    return pd.DataFrame({"company_id": range(1, n + 1), "revenue_musd": revenue,
                         "prior_growth": prior, "next_growth": nxt.round(3)})


def main():
    for name, text in DOCS.items():
        (DATA / name).write_text(text, encoding="utf-8")
    make_financials().to_csv(DATA / "financials.csv", index=False)
    (ROOT / "eval_set.json").write_text(json.dumps(EVAL_SET, indent=2), encoding="utf-8")
    print(f"Wrote {len(DOCS)} documents, financials.csv, and eval_set.json ({len(EVAL_SET)} questions)")


if __name__ == "__main__":
    main()
