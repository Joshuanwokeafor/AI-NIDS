# AI-NIDS

### AI-Enhanced Intrusion Detection System using Machine Learning for Protocol Traffic Pattern

A hybrid, ensemble-based Network Intrusion Detection System combining an
RBF-kernel SVM, a 100-tree Random Forest, and an LSTM network (for
temporal/sequential protocol dependencies), fused via Fuzzy c-means
clustering with confidence-threshold routing to an analyst review queue.

Built around an OOAD five-stage pipeline:

```
DataIngestor -> Preprocessor -> FeatureExtractor -> [SVM | RandomForest | LSTM] -> FuzzyAggregator -> AlertManager
```

---

## 1. Architecture

| Class                    | File                                 | Responsibility                                                             |
| ------------------------ | ------------------------------------ | -------------------------------------------------------------------------- |
| `DataIngestor`           | `src/data_ingestor.py`               | Loads NSL-KDD CSV or PCAP/live traffic; rejects and logs malformed records |
| `Preprocessor`           | `src/preprocessor.py`                | Min-Max normalization, one-hot encoding, SMOTE balancing (train only)      |
| `FeatureExtractor`       | `src/feature_extractor.py`           | Mutual-information feature ranking, top-k selection                        |
| `SVMClassifier`          | `src/classifiers/svm_classifier.py`  | RBF-kernel SVM ensemble member                                             |
| `RandomForestClassifier` | `src/classifiers/rf_classifier.py`   | 100-tree Random Forest ensemble member                                     |
| `LSTMClassifier`         | `src/classifiers/lstm_classifier.py` | Sequence-aware LSTM ensemble member                                        |
| `FuzzyAggregator`        | `src/fuzzy_aggregator.py`            | Fuses 15-dim probability vector via fuzzy c-means, threshold routing       |
| `AlertManager`           | `src/alert_manager.py`               | Persists alerts, immutable audit log, CSV/PDF export                       |
| `DashboardController`    | `web/app.py`                         | Flask dashboard with RBAC (admin / analyst)                                |
| `NIDSPipeline`           | `pipeline.py`                        | Orchestrates all five stages end-to-end                                    |

Classification output: **Normal, DoS, Probe, R2L, U2R** (FR3).

Records whose maximum fuzzy-membership falls below the ambiguity
threshold **δ = 0.6** (configurable) are routed to the analyst review
queue instead of an automated alert.

---

## 2. Installation

Requires **Python 3.11+**.

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`scapy` (PCAP/live capture) is optional — the CSV training/evaluation
path works without it. Live capture additionally requires root /
`CAP_NET_RAW` privileges on the host OS.

---

## 3. Getting the dataset

Download the NSL-KDD dataset (`KDDTrain+.csv`, `KDDTest+.csv`) from the
Canadian Institute for Cybersecurity and place them in `data/`:

```
data/KDDTrain+.csv
data/KDDTest+.csv
```

The `DataIngestor` also accepts headerless NSL-KDD files (the classic
distribution ships without a header row) — it auto-detects either
format.

---

## 4. Training

```bash
pip install -U imbalanced-learn
```

```bash
python scripts/train.py --data data/KDDTrain+.csv --test-size 0.2
```

This runs the full pipeline (ingest → preprocess+SMOTE → feature
select → train SVM/RF/LSTM → fit+calibrate fuzzy aggregator), saves
all artifacts to `models/`, and immediately writes a held-out
evaluation report to `logs/evaluation_report.json` (see §6).

**Retraining (FR5):** re-run the same command against a new labelled
CSV at any time — it overwrites `models/` and logs a `MODEL_RETRAINED`
audit event. Retraining is also available from the dashboard
(Administrator Controls → Retrain).

---

## 5. Evaluation

```bash
python scripts/evaluate.py --data data/KDDTest+.csv
```

Loads the saved model and reports, on the held-out set:
Accuracy, Precision/Recall/F1 (macro), per-class and macro False
Positive Rate, full 5x5 confusion matrix, review-queue rate, and an
empirical single-record latency benchmark for the full hybrid pipeline
(NF1) — mean and p95 milliseconds, broken down per model.

**Cross-dataset validation (NF3):** to validate generalizability beyond
NSL-KDD, point `--data` at a UNSW-NB15 export that has been remapped to
the same 41+class column schema (`config.NSL_KDD_COLUMNS`,
`config.ATTACK_CATEGORY_MAP`) — no code changes are required, only a
schema-conforming CSV.

---

## 6. Running the dashboard (FR6)

```bash
python web/app.py
```

Visit `http://127.0.0.1:5000`. First run creates a default account
**admin / admin** — change this password immediately (Administrator
Controls → Create user, then remove/rotate the default account).

Roles (FR7):

- **admin** — full access: retrain, change the ambiguity threshold,
  create users, view the audit log.
- **analyst** — read-only: view real-time alerts, historical alerts,
  model metrics, acknowledge alerts, and export reports.

Export alert logs/reports as CSV or PDF from the dashboard or:

```python
from src.alert_manager import AlertManager
AlertManager().export_alerts_csv("alerts.csv")
AlertManager().export_alerts_pdf("alerts.pdf")
```

---

## 7. Real-time / live PCAP mode

Offline PCAP file:

```python
from src.data_ingestor import DataIngestor
df, report = DataIngestor().load_pcap("capture.pcap")
```

Live interface (Linux, requires root):

```bash
sudo python scripts/run_live.py --interface eth0
```

This sniffs the interface, aggregates packets into NSL-KDD-shaped flow
records (`PacketFlowAggregator`), and runs each completed flow through
the trained pipeline, printing and persisting alerts as they occur.

---

## 8. Testing (NF6)

```bash
pytest                      # full suite
pytest --cov=src --cov=pipeline   # with coverage
```

Unit tests cover `DataIngestor` (CSV parsing, malformed-record
rejection, PCAP error handling), `Preprocessor` (normalization
bounds, one-hot expansion, SMOTE balancing, unseen-category/label
handling), and `FuzzyAggregator` (fusion vector shape validation,
membership-sum invariants, threshold routing, cluster calibration).

---

## 9. Database

`db/schema.sql` defines `users`, `alerts`, `audit_log`, and
`model_performance` tables. Written to be SQLite-compatible out of the
box (default, via `AlertManager`); for MySQL, substitute
`INTEGER PRIMARY KEY AUTOINCREMENT` → `INT AUTO_INCREMENT PRIMARY KEY`
and adjust `TEXT`/`REAL` types as noted in the file header.

---

## 10. Project structure

```
ai_nids/
├── config.py                 # schema, hyperparameters, thresholds
├── pipeline.py                # NIDSPipeline orchestrator
├── requirements.txt
├── db/schema.sql
├── src/
│   ├── data_ingestor.py       # Stage 1
│   ├── preprocessor.py        # Stage 2
│   ├── feature_extractor.py   # Stage 3 (+ PCAP flow aggregation)
│   ├── classifiers/           # Stage 4
│   │   ├── base.py
│   │   ├── svm_classifier.py
│   │   ├── rf_classifier.py
│   │   └── lstm_classifier.py
│   ├── fuzzy_aggregator.py    # Stage 5
│   └── alert_manager.py       # alerts, audit trail, export
├── web/
│   ├── app.py                 # DashboardController (Flask)
│   └── templates/
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── run_live.py
└── tests/
```

---

## 11. Known limitations (in-scope disclosures for the report)

- `PacketFlowAggregator` is a lightweight, dependency-free flow
  featurizer, not a full CICFlowMeter-equivalent; it derives a reduced
  feature subset sufficient to drive the same trained pipeline used
  for CSV data, but PCAP-mode feature fidelity is lower than the
  original NSL-KDD feature set.
- Fuzzy c-means clusters are unordered by construction; `calibrate()`
  must be run once against a labelled batch before predictions are
  trustworthy as class labels (done automatically at the end of
  `NIDSPipeline.train()`).
- Host-based detection and physical security are explicitly out of
  scope, per the project constraints.
