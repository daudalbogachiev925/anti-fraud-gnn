"""Build transaction graph: users, cards, merchants, devices."""
import pandas as pd
import torch
from torch_geometric.data import Data


class TransactionGraphBuilder:
    """Builds homogeneous graph with node-type indicator features."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def build(self) -> Data:
        df = self.df
        users = df["user_id"].unique()
        cards = df["card_id"].unique()
        merchants = df["merchant_id"].unique()
        devices = df["device_id"].unique()

        offsets = {
            "user": 0,
            "card": len(users),
            "merchant": len(users) + len(cards),
            "device": len(users) + len(cards) + len(merchants),
        }
        n_nodes = sum(len(x) for x in [users, cards, merchants, devices])

        x = torch.zeros(n_nodes, 4)
        x[: len(users), 0] = 1
        x[offsets["card"] : offsets["card"] + len(cards), 1] = 1
        x[offsets["merchant"] : offsets["merchant"] + len(merchants), 2] = 1
        x[offsets["device"] :, 3] = 1

        u_idx = {u: i for i, u in enumerate(users)}
        c_idx = {c: i for i, c in enumerate(cards)}
        m_idx = {m: i for i, m in enumerate(merchants)}
        d_idx = {d: i for i, d in enumerate(devices)}

        src, dst = [], []
        for _, row in df.iterrows():
            u = offsets["user"] + u_idx[row["user_id"]]
            c = offsets["card"] + c_idx[row["card_id"]]
            m = offsets["merchant"] + m_idx[row["merchant_id"]]
            d = offsets["device"] + d_idx[row["device_id"]]
            src += [u, c, u, d]
            dst += [c, u, d, u]

        edge_index = torch.tensor([src, dst], dtype=torch.long)
        return Data(x=x, edge_index=edge_index, n_nodes=n_nodes)
