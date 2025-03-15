# -*- coding: utf-8 -*-
import dgl
import dgl.data.fraud
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import random

# Constants
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16
NUM_TEMPORAL_BASES = 4
NUM_RELATIONS = 3


class DataLoader:
    @staticmethod
    def load_yelpchi_data():
        print("Loading YelpChi dataset...")
        dataset = dgl.data.fraud.FraudYelpDataset()
        graph = dataset[0]
        num_classes = dataset.num_classes
        features = graph.ndata['feature'].float()
        labels = graph.ndata['label'].long()
        print(f"Dataset loaded: {graph.num_nodes()} nodes, {graph.num_edges()} edges, "
              f"{features.shape[1]} features, {num_classes} classes")
        return graph, features, labels, num_classes


class GraphProcessor:
    @staticmethod
    def convert_to_pyg_format(graph, features):
        print("Converting DGL graph to PyTorch Geometric format...")
        if graph.is_homogeneous:
            edge_index = torch.stack(graph.edges(), dim=0)
            print("Graph is homogeneous")
        else:
            edge_types = graph.etypes
            print(f"Graph is heterogeneous. Available edge types: {edge_types}")
            edge_type = edge_types[0]  # Try changing this if needed
            edge_index = torch.stack(graph.edges(etype=edge_type), dim=0)
            print(f"Using edge type: {edge_type}, {edge_index.shape[1]} edges selected")
        data = Data(x=features, edge_index=edge_index)
        print(f"PyG Data created: {data.num_nodes} nodes, {data.num_edges} edges")
        return data

    @staticmethod
    def prepare_masks(labels, train_ratio=0.8):
        print("Preparing train/test masks...")
        labeled_idx = torch.where(labels != -1)[0].tolist()
        print(f"Found {len(labeled_idx)} labeled nodes")
        train_mask, test_mask = train_test_split(labeled_idx,
                                                 train_size=train_ratio,
                                                 random_state=42)
        print(f"Train mask: {len(train_mask)} nodes, Test mask: {len(test_mask)} nodes")
        return train_mask, test_mask


class TemporalDependencyModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_temporal_bases):
        super().__init__()
        self.attribute_embedding = nn.Linear(embedding_size, hidden_size)
        self.temporal_encoding = TemporalEncoding(hidden_size, num_temporal_bases)

    def forward(self, x, timestamp):
        attr_emb = F.relu(self.attribute_embedding(x))
        temp_enc = self.temporal_encoding(timestamp)
        return attr_emb + temp_enc


class TemporalEncoding(nn.Module):
    def __init__(self, embedding_size, num_temporal_bases):
        super().__init__()
        self.num_temporal_bases = num_temporal_bases
        self.linear = nn.Linear(num_temporal_bases * 2, embedding_size)

    def forward(self, tvi):
        bases = torch.zeros(self.num_temporal_bases, 2)
        for i in range(self.num_temporal_bases):
            bases[i, 0] = torch.sin(tvi / (10000.0 ** (2 * i / self.num_temporal_bases)))
            bases[i, 1] = torch.cos(tvi / (10000.0 ** (2 * i + 1 / self.num_temporal_bases)))
        return F.relu(self.linear(bases.view(1, -1)))


class TransformerBlock(nn.Module):
    def __init__(self, input_size, num_heads, adj_matrix):
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.head_dim = input_size // num_heads
        assert input_size % num_heads == 0, "input_size must be divisible by num_heads"

        print(f"Initializing TransformerBlock: input_size={input_size}, num_heads={num_heads}, "
              f"head_dim={self.head_dim}")

        self.qkv_proj = nn.Linear(input_size, input_size * 3)
        self.adj_matrix = adj_matrix
        self.src, self.tgt = adj_matrix.nonzero(as_tuple=True)
        print(f"TransformerBlock edges: {self.src.size(0)} edges loaded from full graph")

        self.ffn = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.ReLU(),
            nn.Linear(4 * input_size, input_size)
        )
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, node_indices):
        batch_size, num_nodes, _ = x.size()
        print(f"TransformerBlock forward: input shape={x.shape}, node_indices={len(node_indices)}")
        if batch_size != 1:
            raise ValueError("Sparse attention currently assumes batch_size=1")

        index_map = {idx.item(): i for i, idx in enumerate(node_indices)}
        print(f"Index map created with {len(index_map)} entries, max original index={max(index_map.keys())}")

        mask = torch.tensor([idx.item() in index_map and self.tgt[i].item() in index_map
                             for i, idx in enumerate(self.src)], dtype=torch.bool)
        src_sub = self.src[mask]
        tgt_sub = self.tgt[mask]
        print(f"Edges after filtering: {src_sub.size(0)} (before mapping)")

        src_mapped = torch.tensor([index_map[idx.item()] for idx in src_sub], dtype=torch.long)
        tgt_mapped = torch.tensor([index_map[idx.item()] for idx in tgt_sub], dtype=torch.long)
        print(f"Filtered and mapped edges: {src_mapped.size(0)}, src_max={src_mapped.max().item()}, "
              f"tgt_max={tgt_mapped.max().item()}")

        qkv = self.qkv_proj(x).reshape(num_nodes, 3 * self.input_size)
        q, k, v = qkv.split(self.input_size, dim=-1)
        print(f"QKV projected: q={q.shape}, k={k.shape}, v={v.shape}")

        q = q.view(num_nodes, self.num_heads, self.head_dim)
        k = k.view(num_nodes, self.num_heads, self.head_dim)
        v = v.view(num_nodes, self.num_heads, self.head_dim)
        print(f"Multi-head reshape: q={q.shape}, k={k.shape}, v={v.shape}")

        num_edges = src_mapped.size(0)
        print(f"Computing attention for {num_edges} edges")

        if num_edges == 0:
            print("Warning: No edges found in subset, returning input with FFN only")
            x = self.norm1(x.squeeze(0))
            ffn_out = self.ffn(x)
            return self.norm2(x + ffn_out).unsqueeze(0)

        q_edges = q[src_mapped]
        k_edges = k[tgt_mapped]
        print(f"Edge Q/K: q_edges={q_edges.shape}, k_edges={k_edges.shape}")

        attn_scores = (q_edges * k_edges).sum(dim=-1) / (self.head_dim ** 0.5)
        print(f"Attention scores: {attn_scores.shape}")

        attn_weights = torch.zeros_like(attn_scores)
        for head in range(self.num_heads):
            scores_head = attn_scores[:, head]
            weights_head = torch.zeros(num_nodes, device=x.device)
            weights_head.index_add_(0, src_mapped, F.softmax(scores_head, dim=0))
            attn_weights[:, head] = weights_head[src_mapped]
        print(f"Attention weights: {attn_weights.shape}")

        v_edges = v[tgt_mapped]
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        for head in range(self.num_heads):
            weighted_v = attn_weights[:, head].unsqueeze(-1) * v_edges[:, head, :]
            out[:, head, :].index_add_(0, src_mapped, weighted_v)
        print(f"Attention output before reshape: {out.shape}")

        out = out.view(num_nodes, self.input_size)
        out = self.dropout(out)
        print(f"Attention output after reshape: {out.shape}")

        x = self.norm1(x.squeeze(0) + out)
        print(f"After norm1: {x.shape}")

        ffn_out = self.ffn(x)
        out = self.norm2(x + ffn_out).unsqueeze(0)
        print(f"TransformerBlock output: {out.shape}")
        return out


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))


class SpatialDependencyModel(nn.Module):
    def __init__(self, num_relations, hidden_size, mlp_hidden_size, adj_matrix):
        super().__init__()
        self.transformer = TransformerBlock(hidden_size, num_heads=4, adj_matrix=adj_matrix)
        self.mlp = MLP(hidden_size, mlp_hidden_size)
        self.adj_matrix = adj_matrix

    def forward(self, node_embeddings, timestamp, relations, index_obtains):
        print(f"SpatialDependencyModel forward: {len(index_obtains)} nodes to process")
        aggregated_reps = []
        valid_indices = []

        for vi in index_obtains:
            neighbors = torch.where(self.adj_matrix[vi] > 0)[0]
            # print(f"Node {vi}: {len(neighbors)} neighbors")
            if len(neighbors) == 0:
                continue

            neighbor_emb = node_embeddings[neighbors]
            weights = self.adj_matrix[vi, neighbors]
            intra_agg = torch.mean(weights.view(-1, 1) * neighbor_emb, dim=0)
            diff = node_embeddings[vi] - neighbor_emb
            inter_agg = torch.mean(diff, dim=0)

            combined = intra_agg + inter_agg
            if combined.dim() > 0:
                aggregated_reps.append(combined)
                valid_indices.append(vi)
            else:
                print(f"Node {vi}: Combined dim invalid: {combined.dim()}")

        if not aggregated_reps:
            print("No valid aggregations found, falling back to node embeddings")
            fused = node_embeddings[index_obtains].unsqueeze(0)
            prediction = self.mlp(fused.squeeze(0))
            return prediction, fused, None, index_obtains

        fused = torch.stack(aggregated_reps).unsqueeze(0)
        print(f"Aggregated representations: {fused.shape}")
        fused = self.transformer(fused, torch.tensor(valid_indices, dtype=torch.long))
        prediction = self.mlp(fused.squeeze(0))
        print(f"Spatial output: fused={fused.shape}, prediction={prediction.shape}")
        return prediction, fused, None, valid_indices


class TemporalSpatialModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_temporal_bases, num_relations,
                 mlp_hidden_size, adj_matrix):
        super().__init__()
        self.temporal = TemporalDependencyModel(embedding_size, hidden_size, num_temporal_bases)
        self.spatial = SpatialDependencyModel(num_relations, hidden_size, mlp_hidden_size, adj_matrix)

    def forward(self, x, node_indices, timestamp, relations, index_obtains):
        temporal_emb = self.temporal(x, timestamp)
        print(f"TemporalSpatialModel: temporal_emb={temporal_emb.shape}")
        return self.spatial(temporal_emb, timestamp, relations, index_obtains)


class CombinedModel(nn.Module):
    def __init__(self, input_dim, embedding_size, hidden_size, num_temporal_bases,
                 num_relations, mlp_hidden_size, adj_matrix):
        super().__init__()
        print(f"Initializing CombinedModel: input_dim={input_dim}, embedding_size={embedding_size}")
        self.initial_conv = GCNConv(input_dim, embedding_size)
        self.temporal_spatial = TemporalSpatialModel(
            embedding_size, hidden_size, num_temporal_bases, num_relations,
            mlp_hidden_size, adj_matrix
        )

    def forward(self, data, node_indices, timestamp, relations, index_obtains):
        x = F.relu(self.initial_conv(data.x, data.edge_index))
        print(f"CombinedModel after GCN: {x.shape}")
        return self.temporal_spatial(x, node_indices, timestamp, relations, index_obtains)


def train_model(model, data, labels, train_mask, test_mask, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()

        pred, _, _, idx = model(data, torch.arange(data.num_nodes),
                                torch.tensor(123.45), [1], train_mask.copy())

        if pred is None:
            print(f"Epoch {epoch + 1}: No predictions generated")
            continue

        masked_pred = pred.squeeze()
        masked_labels = torch.tensor([labels[i] for i in idx], dtype=torch.float)
        print(f"Epoch {epoch + 1}: pred={masked_pred.shape}, labels={masked_labels.shape}")

        loss = loss_fn(masked_pred, masked_labels)
        loss.backward()
        optimizer.step()

        # if epoch % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred, _, _, idx = model(data, torch.arange(data.num_nodes),
                                torch.tensor(123.45), [1], test_mask.copy())

        if pred is None:
            print("No predictions generated during evaluation")
            return

        masked_labels = torch.tensor([labels[i] for i in idx])
        predicted_labels = (pred.squeeze() > 0).long()

        accuracy = (predicted_labels == masked_labels).float().mean().item()
        f1 = f1_score(masked_labels.numpy(), predicted_labels.numpy(), average='macro')
        probs = torch.sigmoid(pred).squeeze().numpy()
        auc = roc_auc_score(masked_labels.numpy(), probs)

        cm = confusion_matrix(masked_labels.numpy(), predicted_labels.numpy())
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        gmeans = (sensitivity * specificity) ** 0.5

        print(f"Evaluation - Accuracy: {accuracy:.4f}, F1-macro: {f1:.4f}, "
              f"AUC: {auc:.4f}, G-means: {gmeans:.4f}")


def main():
    graph, features, labels, num_classes = DataLoader.load_yelpchi_data()
    data = GraphProcessor.convert_to_pyg_format(graph, features)
    train_mask, test_mask = GraphProcessor.prepare_masks(labels)

    adj_matrix = torch.zeros((data.num_nodes, data.num_nodes))
    adj_matrix[data.edge_index[0], data.edge_index[1]] = 1
    print(f"Adjacency matrix created: {adj_matrix.shape}, {adj_matrix.sum().item()} edges")

    model = CombinedModel(
        input_dim=features.shape[1],
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_temporal_bases=NUM_TEMPORAL_BASES,
        num_relations=NUM_RELATIONS,
        mlp_hidden_size=EMBEDDING_SIZE,
        adj_matrix=adj_matrix
    )

    train_model(model, data, labels, train_mask, test_mask)


if __name__ == "__main__":
    main()