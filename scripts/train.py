"""Training script. Run: python scripts/train.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.loader import load_transactions
from src.data.graph_builder import TransactionGraphBuilder
from src.models.tabular import TabularFraudModel
from src.models.gnn import FraudGNN
from src.config import MODELS_DIR
import torch


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Loading transactions...")
    df = load_transactions()
    print(f"  {len(df)} rows, fraud rate {df['is_fraud'].mean():.3%}")

    FEATURES = ["amount", "hour"]
    X = df[FEATURES].values
    y = df["is_fraud"].values

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y,
    )
    print(f"  Train: {X_tr.shape}, Test: {X_te.shape}")

    print("Training tabular CatBoost...")
    tab_model = TabularFraudModel()
    tab_model.fit(X_tr, y_tr, X_te, y_te)
    probs = tab_model.predict_proba(X_te)

    print(f"  ROC-AUC: {roc_auc_score(y_te, probs):.4f}")
    print(f"  PR-AUC:  {average_precision_score(y_te, probs):.4f}")

    tab_model.save(MODELS_DIR / "catboost_fraud.cbm")
    print(f"Model saved to {MODELS_DIR / 'catboost_fraud.cbm'}")

    print("Building transaction graph...")
    builder = TransactionGraphBuilder(df.head(20000))
    graph = builder.build()
    print(f"  Graph: {graph.x.shape[0]} nodes, {graph.edge_index.shape[1]} edges")

    print("Training GNN (unsupervised embedding warmup)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gnn = FraudGNN(in_dim=graph.x.shape[1]).to(device)
    x = graph.x.to(device)
    ei = graph.edge_index.to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)

    for epoch in range(20):
        gnn.train()
        out = gnn(x, ei)
        loss = out.norm(dim=1).mean() * 0.01
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
