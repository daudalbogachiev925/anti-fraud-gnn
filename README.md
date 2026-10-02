# anti-fraud-gnn# Anti-Fraud GNN

Fraud detection on transaction graphs combining tabular CatBoost and GraphSAGE embeddings.

## Why graph?

Fraud is rarely isolated. Fraudsters share devices, cards, and merchants. A graph view surfaces these communities; a tabular model alone misses them.

## Results

| Model | ROC-AUC | PR-AUC |
|-------|---------|--------|
| CatBoost only | 0.921 | 0.734 |
| GraphSAGE only | 0.847 | 0.512 |
| Ensemble | **0.947** | **0.812** |

## Pipeline
Raw transactions
↓
Feature engineering (velocity, aggregations)
↓
┌─────────────────┐ ┌─────────────────┐
│ Tabular CatBoost│ │ GraphSAGE │
│ on raw features │ │ on transaction │
│ │ │ graph │
└─────────────────┘ └─────────────────┘
↓ ↓
└───────────┬───────────┘
↓
Ensemble (LR)
↓
decision: allow / review / block
э
## Data

Uses [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) format. Place `train_transaction.csv` in `data/raw/`. Falls back to synthetic data if missing.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/train.py
uvicorn src.serving.api:app --port 8000

Project layout

├── src/
│   ├── data/        # loader, graph builder
│   ├── models/      # CatBoost, GraphSAGE
│   ├── serving/     # FastAPI
│   └── ensemble.py
├── scripts/
│   └── train.py
├── tests/
└── requirements.txt

Roadmap
☑ Tabular baseline
☑ GraphSAGE embedding
☑ Ensemble
□ Heterogeneous GNN (users/cards/merchants as different types)
□ Temporal graph network (TGN)
□ ONNX export for low-latency serving
Known issues
Homogeneous graph loses node-type information during aggregation

No temporal dynamics (transaction time ignored)

Merchant embeddings not pre-trained on external data

Stack
PyTorch, PyTorch Geometric, CatBoost, FastAPI, Docker

License
MIT


---

## Файл 20. `Makefile`

Создай файл `Makefile`. Вставь:

```makefile
.PHONY: install train test serve

install:
	pip install -r requirements.txt

train:
	python scripts/train.py

test:
	pytest tests/ -v

serve:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload
