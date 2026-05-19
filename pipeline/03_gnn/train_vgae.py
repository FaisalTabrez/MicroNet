#!/usr/bin/env python3
"""
train_vgae.py — Variational Graph Autoencoder for microbial interaction network
                reconstruction and link prediction.

Architecture:
    Input:  Co-occurrence graph (adjacency) + node feature matrix
            Node features = [CLR abundance profile | phylogenetic embedding |
                             functional gene content]
    Encoder: 2-layer GCN → (mu, log_std) latent vectors per node
    Decoder: Inner product of latent vectors → edge probability
    Loss:    ELBO = BCE reconstruction + KL divergence

    Additional head: Link type classifier (mutualism/competition/parasitism)
    using gLV-labeled edges as supervision.

Reference: Kipf & Welling (2016), "Variational Graph Autoencoders"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import negative_sampling
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import json
import matplotlib.pyplot as plt


# ── Model definition ─────────────────────────────────────────────────────────

class GCNEncoder(nn.Module):
    """Two-layer GCN encoder producing mean and log-variance of latent space."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logstd = GCNConv(hidden_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(hidden_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn(x)
        x = F.elu(x)
        x = self.dropout(x)
        mu = self.conv_mu(x, edge_index)
        log_std = self.conv_logstd(x, edge_index)
        return mu, log_std


class VGAE(nn.Module):
    """Variational Graph Autoencoder with reparameterization trick."""

    def __init__(self, encoder: GCNEncoder):
        super().__init__()
        self.encoder = encoder

    def reparameterize(self, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(log_std)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def encode(self, x, edge_index):
        self.mu, self.log_std = self.encoder(x, edge_index)
        return self.reparameterize(self.mu, self.log_std)

    def decode(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Inner product decoder — returns edge logits."""
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)

    def decode_all(self, z: torch.Tensor) -> torch.Tensor:
        """Full adjacency reconstruction (N×N). Use only for small graphs."""
        return torch.sigmoid(z @ z.T)

    def recon_loss(self, z: torch.Tensor,
                   pos_edge_index: torch.Tensor,
                   neg_edge_index: torch.Tensor = None) -> torch.Tensor:
        """Binary cross-entropy reconstruction loss."""
        pos_loss = -F.logsigmoid(self.decode(z, pos_edge_index)).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(
                pos_edge_index, num_nodes=z.size(0),
                num_neg_samples=pos_edge_index.size(1)
            )
        neg_loss = -F.logsigmoid(-self.decode(z, neg_edge_index)).mean()
        return pos_loss + neg_loss

    def kl_loss(self) -> torch.Tensor:
        """KL divergence: D_KL[q(z|x) || p(z)] where p(z) = N(0, I)."""
        return -0.5 * torch.mean(
            1 + 2 * self.log_std - self.mu.pow(2) - (2 * self.log_std).exp()
        )

    def elbo(self, z, pos_edge_index, neg_edge_index=None, beta=1.0):
        return self.recon_loss(z, pos_edge_index, neg_edge_index) + beta * self.kl_loss()


class InteractionTypeHead(nn.Module):
    """
    MLP classifier for labeling edge types using gLV-inferred interaction labels.
    Input: concatenated latent vectors of source and target nodes.
    Output: interaction type logits (Mutualism / Competition / Parasitism / Other)
    """
    TYPES = ["Mutualism", "Competition", "Parasitism", "Commensalism", "Amensalism"]

    def __init__(self, latent_dim: int, hidden: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, len(self.TYPES))
        )

    def forward(self, z, edge_index):
        src, dst = edge_index
        edge_features = torch.cat([z[src], z[dst]], dim=-1)
        return self.mlp(edge_features)


# ── Data preparation ─────────────────────────────────────────────────────────

def build_graph_data(abundance_path: str,
                     adj_path: str,
                     functional_path: str = None,
                     interaction_labels_path: str = None) -> Data:
    """
    Build a PyG Data object from:
    - abundance_path: CLR matrix (samples × taxa) → node features = mean profile
    - adj_path: SPIEC-EASI weighted adjacency TSV (taxa × taxa)
    - functional_path: HUMAnN3 pathway matrix (optional, appended to features)
    - interaction_labels_path: gLV classified interactions for edge label supervision
    """
    # Node features: mean CLR abundance profile per taxon (shape: N × S)
    clr = pd.read_csv(abundance_path, sep="\t", index_col=0)
    taxa = clr.columns.tolist()
    node_features = clr.T.values.astype(np.float32)  # N × S

    # Append functional features if available
    if functional_path:
        func = pd.read_csv(functional_path, sep="\t", index_col=0)
        func = func.reindex(taxa).fillna(0)
        node_features = np.hstack([node_features, func.values.astype(np.float32)])

    # Adjacency matrix
    adj_df = pd.read_csv(adj_path, sep="\t", index_col=0)
    adj_df = adj_df.reindex(index=taxa, columns=taxa).fillna(0)
    adj = adj_df.values.astype(np.float32)

    # Convert adjacency to edge_index + edge_weight
    rows, cols = np.where(adj != 0)
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(adj[rows, cols], dtype=torch.float)

    x = torch.tensor(node_features, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_weight,
                num_nodes=len(taxa))
    data.taxa_names = taxa

    # Edge labels from gLV (optional supervision)
    if interaction_labels_path:
        labels_df = pd.read_csv(interaction_labels_path, sep="\t")
        type_map = {t: i for i, t in enumerate(InteractionTypeHead.TYPES)}
        # Map taxon indices
        taxa_idx = {t: i for i, t in enumerate(taxa)}
        labeled_edges = []
        labeled_types = []
        for _, row in labels_df.iterrows():
            i = taxa_idx.get(row.get("taxon_i_name"))
            j = taxa_idx.get(row.get("taxon_j_name"))
            t = type_map.get(row.get("interaction_type"), 4)
            if i is not None and j is not None:
                labeled_edges.append([i, j])
                labeled_types.append(t)
        if labeled_edges:
            data.labeled_edge_index = torch.tensor(labeled_edges, dtype=torch.long).T
            data.edge_labels = torch.tensor(labeled_types, dtype=torch.long)

    return data


# ── Training ─────────────────────────────────────────────────────────────────

def train_epoch(model, head, data, optimizer, device, beta=1.0):
    model.train()
    optimizer.zero_grad()

    x = data.x.to(device)
    # FIX C2: train_pos_edge_index was the old train_test_split_edges API.
    # RandomLinkSplit stores training message-passing edges in data.edge_index.
    edge_index = data.edge_index.to(device)

    z = model.encode(x, edge_index)
    loss = model.elbo(z, edge_index, beta=beta)

    # Optional: add interaction type classification loss
    if hasattr(data, "labeled_edge_index"):
        le = data.labeled_edge_index.to(device)
        logits = head(z, le)
        labels = data.edge_labels.to(device)
        clf_loss = F.cross_entropy(logits, labels)
        loss = loss + 0.5 * clf_loss

    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def evaluate(model, val_data, device):
    model.eval()
    x          = val_data.x.to(device)
    edge_index  = val_data.edge_index.to(device)
    z = model.encode(x, edge_index)

    from sklearn.metrics import roc_auc_score, average_precision_score

    pos_edge = val_data.edge_label_index[:, val_data.edge_label == 1].to(device)
    neg_edge = val_data.edge_label_index[:, val_data.edge_label == 0].to(device)

    pos_pred = torch.sigmoid(model.decode(z, pos_edge)).cpu().numpy()
    neg_pred = torch.sigmoid(model.decode(z, neg_edge)).cpu().numpy()

    y_true = np.concatenate([np.ones(len(pos_pred)), np.zeros(len(neg_pred))])
    y_pred = np.concatenate([pos_pred, neg_pred])

    auc = roc_auc_score(y_true, y_pred)
    ap  = average_precision_score(y_true, y_pred)
    return auc, ap, z.cpu()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train VGAE for microbial network reconstruction")
    parser.add_argument("--abundance",    required=True)
    parser.add_argument("--adjacency",    required=True)
    parser.add_argument("--functional",   default=None)
    parser.add_argument("--interactions", default=None,
                        help="gLV classified interactions for edge label supervision")
    parser.add_argument("--hidden",   type=int, default=128)
    parser.add_argument("--latent",   type=int, default=64)
    parser.add_argument("--epochs",   type=int, default=300)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--beta",     type=float, default=1.0, help="KL weight")
    parser.add_argument("--outdir",   default="../../results/gnn")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data — FIX C5: train_test_split_edges was removed in PyG ≥ 2.3.
    # Use RandomLinkSplit which returns three separate Data objects.
    data = build_graph_data(args.abundance, args.adjacency,
                            args.functional, args.interactions)
    transform = RandomLinkSplit(
        num_val=0.1, num_test=0.1,
        is_undirected=True,
        add_negative_train_samples=False,
    )
    train_data, val_data, test_data = transform(data)
    # Attach labeled edges to train_data for the classification head
    if hasattr(data, "labeled_edge_index"):
        train_data.labeled_edge_index = data.labeled_edge_index
        train_data.edge_labels        = data.edge_labels
    print(f"Graph: {data.num_nodes} nodes, in_features={data.x.shape[1]}")

    # Model
    encoder = GCNEncoder(data.x.shape[1], args.hidden, args.latent)
    model   = VGAE(encoder).to(device)
    head    = InteractionTypeHead(args.latent).to(device)

    # FIX M6: only include head parameters in the optimizer when labeled edges
    # exist. When there are none, the head receives zero gradients but still
    # accumulates optimizer state (momentum, weight decay), causing numeric drift.
    params = list(model.parameters())
    if hasattr(train_data, "labeled_edge_index"):
        params += list(head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    # FIX M7: record loss every epoch so the loss curve is dense and accurate.
    # Validation AUC is still computed every 10 epochs (expensive), but the best
    # model is checkpointed whenever a new best is found at those eval points.
    history = {"loss": [], "val_auc": [], "val_ap": []}
    best_auc = 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, head, train_data, optimizer, device, beta=args.beta)
        scheduler.step()
        history["loss"].append(loss)   # Every epoch — not just eval epochs

        if epoch % 10 == 0:
            auc, ap, z = evaluate(model, val_data, device)
            history["val_auc"].append(auc)
            history["val_ap"].append(ap)
            print(f"Epoch {epoch:4d} | Loss {loss:.4f} | AUC {auc:.4f} | AP {ap:.4f}")

            if auc > best_auc:
                best_auc = auc
                torch.save(model.state_dict(), outdir / "best_vgae.pt")
                np.save(outdir / "best_embeddings.npy", z.numpy())

    # Save training history
    with open(outdir / "training_history.json", "w") as f:
        json.dump(history, f)

    # Plot learning curves
    # FIX I2: after the M7 fix, history["loss"] has one entry per epoch (not per
    # 10 epochs), so "Epoch (×10)" was misleading. AUC/AP are still sampled every
    # 10 epochs, so their x-axis is labelled separately for clarity.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(history["loss"])
    ax1.set_title("ELBO Loss")
    ax1.set_xlabel("Epoch")
    ax2.plot(history["val_auc"], label="AUC")
    ax2.plot(history["val_ap"],  label="AP")
    ax2.set_title("Validation Metrics")
    ax2.set_xlabel("Evaluation checkpoint (every 10 epochs)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training_curves.png", dpi=150)

    # Final link prediction on test set
    # FIX C6: weights_only=True prevents arbitrary code execution from
    # tampered .pt files and suppresses the deprecation warning in PyTorch ≥ 2.1.
    model.load_state_dict(torch.load(outdir / "best_vgae.pt", weights_only=True))
    model.eval()
    with torch.no_grad():
        z = model.encode(train_data.x.to(device), train_data.edge_index.to(device))
        z = z.cpu()

    # Reconstruct full probability matrix
    prob_matrix = torch.sigmoid(z @ z.T).numpy()
    prob_df = pd.DataFrame(prob_matrix, index=data.taxa_names, columns=data.taxa_names)
    prob_df.to_csv(outdir / "predicted_edge_probabilities.tsv", sep="\t")

    print(f"\nBest validation AUC: {best_auc:.4f}")
    print(f"Outputs saved → {outdir}")


if __name__ == "__main__":
    main()
