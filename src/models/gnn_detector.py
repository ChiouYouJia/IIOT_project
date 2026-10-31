"""GNN 邊分類 (E-GraphSAGE 架構概念 + PyG GATConv)。

- 依時間切成固定長度窗口，每個窗口一張圖。
- 節點 = sAddress / rAddress (ARP flow 沒有 IP，必須用 address 才不會漏掉 ip-scan)。
- 邊 = flow，邊特徵 = flow 特徵；另加反向邊讓訊息雙向傳遞 (以方向旗標區分)。
- 節點特徵 = 進 / 出邊特徵平均 + 進 / 出度數，不含任何位址識別資訊。
- 分類器輸入 = [h_src, h_dst, flow 特徵]。本資料集只有 19 個節點，若不接 flow 特徵，
  同一窗口內同一對主機的所有 flow 會得到相同預測 (原版 E-GraphSAGE 的作法)。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import scatter

from src.data.loader import sort_by_time

SPLIT_CODE = {"train": 0, "val": 1, "test": 2}


def mac_to_ip_map(data):
    """由 train 的 TCP flow 統計每個 MAC 最常對應的 IP (只取單一 MAC 的 flow)。

    MitM 期間受害 IP 會對應到攻擊者 MAC，因此取多數決而非第一次出現；只用 train 避免看到 val/test。
    """
    df = data["df"][data["split"] == "train"]
    pairs = []
    for ip_col, mac_col in (("sIPs", "sMACs"), ("rIPs", "rMACs")):
        t = df[[ip_col, mac_col]].dropna()
        t = t[~t[mac_col].str.contains("/")]
        pairs.append(t.set_axis(["ip", "mac"], axis=1))
    pairs = pd.concat(pairs)
    return pairs.groupby("mac")["ip"].agg(lambda x: x.value_counts().index[0]).to_dict()


def build_window_graphs(data, window=10.0, node_key="address"):
    """把 load_split() 的結果轉成一串窗口圖。

    node_key="address"：節點直接用 sAddress / rAddress (同一主機的 IP 與 MAC 是兩個節點)。
    node_key="address_merged"：ARP flow 的 MAC 位址依 mac_to_ip_map() 換成該主機的 IP，
    讓同一台主機只有一個節點；對應不到的 (廣播、多播) 維持原樣。

    每張圖的 flow 層級屬性 (flow_index / flow_x / flow_y / flow_split / flow_id) 長度相同，
    flow_index 為 [2, m]，PyG 批次化時會自動加上節點偏移。
    """
    df = sort_by_time(data["df"])
    fid = df.index.to_numpy()
    X = data["X_all"].loc[fid].to_numpy(np.float32)
    y = data["y_all"].loc[fid].to_numpy(np.int64)
    split = data["split"].loc[fid].map(SPLIT_CODE).to_numpy(np.int64)
    src_addr = df["sAddress"].to_numpy()
    dst_addr = df["rAddress"].to_numpy()
    if node_key == "address_merged":
        m2i = mac_to_ip_map(data)
        src_addr = np.array([m2i.get(a, a) for a in src_addr])
        dst_addr = np.array([m2i.get(a, a) for a in dst_addr])
    elif node_key != "address":
        raise ValueError(f"unknown node_key: {node_key}")

    gid = ((df["start"] - df["start"].iloc[0]) // window).astype(np.int64).to_numpy()
    bounds = np.flatnonzero(np.diff(gid)) + 1
    graphs = []
    for idx in np.split(np.arange(len(df)), bounds):
        m = len(idx)
        addrs, inv = np.unique(np.concatenate([src_addr[idx], dst_addr[idx]]), return_inverse=True)
        n = len(addrs)
        s, d = torch.from_numpy(inv[:m]), torch.from_numpy(inv[m:])
        ex = torch.from_numpy(X[idx])

        ones = torch.ones(m, 1)
        node_x = torch.cat([
            scatter(ex, s, dim=0, dim_size=n, reduce="mean"),
            scatter(ex, d, dim=0, dim_size=n, reduce="mean"),
            torch.log1p(scatter(ones, s, dim=0, dim_size=n, reduce="sum")),
            torch.log1p(scatter(ones, d, dim=0, dim_size=n, reduce="sum")),
        ], dim=1)

        edge_index = torch.cat([torch.stack([s, d]), torch.stack([d, s])], dim=1)
        edge_attr = torch.cat([torch.cat([ex, ones], 1), torch.cat([ex, torch.zeros(m, 1)], 1)], dim=0)

        graphs.append(Data(
            x=node_x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=n,
            flow_index=torch.stack([s, d]), flow_x=ex,
            flow_y=torch.from_numpy(y[idx]), flow_split=torch.from_numpy(split[idx]),
            flow_id=torch.from_numpy(fid[idx]),
        ))
    return graphs


class EdgeGAT(nn.Module):
    def __init__(self, node_dim, edge_dim, flow_dim, n_classes, hidden=64, heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        in_dim = node_dim
        for _ in range(n_layers):
            self.convs.append(GATConv(in_dim, hidden, heads=heads, edge_dim=edge_dim, dropout=dropout))
            in_dim = hidden * heads
        self.dropout = dropout
        self.classifier = nn.Sequential(
            nn.Linear(2 * in_dim + flow_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, batch):
        h = batch.x
        for conv in self.convs:
            h = F.elu(conv(h, batch.edge_index, batch.edge_attr))
            h = F.dropout(h, self.dropout, self.training)
        s, d = batch.flow_index
        return self.classifier(torch.cat([h[s], h[d], batch.flow_x], dim=1))
