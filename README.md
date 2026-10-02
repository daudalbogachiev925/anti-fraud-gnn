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
