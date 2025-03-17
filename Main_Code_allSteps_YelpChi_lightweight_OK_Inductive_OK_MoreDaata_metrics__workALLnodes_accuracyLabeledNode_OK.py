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
from sklearn.metrics import (f1_score, roc_auc_score, confusion_matrix,
                             precision_score, recall_score, classification_report)
import random
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix

# Constants
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16
NUM_TEMPORAL_BASES = 4
NUM_RELATIONS = 3


# Step 1: Compute Only Degree Centrality (Lightweight)
def compute_centrality_features(graph):
    print("Step 1: Computing degree centrality (only)...")
    G_nx = nx.DiGraph()
    src, tgt = graph.edges(etype=graph.etypes[0])
    # Use all edges instead of limiting to 5000
    edge_list = list(zip(src.numpy(), tgt.numpy()))
    G_nx.add_edges_from(edge_list)
    degree = nx.degree_centrality(G_nx)
    centrality_embeddings = {}
    for node in range(graph.num_nodes()):  # Ensure all nodes are covered
        centrality_embeddings[node] = [degree.get(node, 0), 0, 0, 0, 0, 0, 0, 0]  # 8D
    print(f"Degree centrality computed for {len(centrality_embeddings)} nodes.")
    return centrality_embeddings


# Step 2: Probabilistic FraudWalk Embeddings with Inductive Pooling
def compute_fraudwalk_embeddings(graph, train_mask=None, inductive_nodes=None):
    print("Step 2: Computing FraudWalk embeddings with Inductive Pooling...")
    src, tgt = graph.edges(etype=graph.etypes[0])
    edge_list = list(zip(src.numpy(), tgt.numpy()))  # Use all edges
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    def simple_random_walk(start_node, walk_length=5):
        walk = [str(start_node)]
        current = start_node
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(str(current))
        return walk

    num_walks = 50
    nodes = list(G.nodes())
    if train_mask is not None:
        train_nodes = [n for n in nodes if n in train_mask]
    else:
        train_nodes = nodes
    walks = [simple_random_walk(random.choice(train_nodes)) for _ in range(num_walks)]
    model = Word2Vec(sentences=walks, vector_size=5, window=3, sg=1, min_count=1, workers=4)
    base_embeddings = {int(node): model.wv[node] for node in model.wv.index_to_key if str(node).isdigit()}
    print(f"Base FraudWalk embeddings trained for {len(base_embeddings)} nodes.")

    fraudwalk_embeddings = {}
    for node in range(graph.num_nodes()):  # Ensure all nodes get embeddings
        if node not in G:
            fraudwalk_embeddings[node] = np.zeros(5)
        elif node not in base_embeddings:
            neighbors = list(G.neighbors(node))
            if not neighbors:
                fraudwalk_embeddings[node] = np.zeros(5)
            else:
                neighbor_embs = [base_embeddings.get(n, np.zeros(5)) for n in neighbors if n in base_embeddings]
                fraudwalk_embeddings[node] = np.mean(neighbor_embs, axis=0) if neighbor_embs else np.zeros(5)
        else:
            fraudwalk_embeddings[node] = base_embeddings[node]
    print(f"Total FraudWalk embeddings generated for {len(fraudwalk_embeddings)} nodes.")
    return fraudwalk_embeddings


# Step 3: Fraud Subgraph, GAE, and GCN (Lightweight)
def reconstruct_adj_and_gcn(graph, labels, train_mask):
    print("Step 3: Reconstructing adjacency matrix with GAE and GCN (optimized)...")
    G_nx = nx.DiGraph()
    src, tgt = graph.edges(etype=graph.etypes[0])
    edge_list = list(zip(src.numpy(), tgt.numpy()))
    G_nx.add_edges_from(edge_list)
    valid_nodes = set(G_nx.nodes())
    node_list = list(G_nx.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}

    all_labeled_idx = torch.where(labels != -1)[0].tolist()
    fraud_nodes = [idx for idx in all_labeled_idx if labels[idx] == 1 and idx in valid_nodes][:50]
    if not fraud_nodes:
        print("Warning: No fraud nodes found in reduced graph. Using first 50 valid labeled nodes.")
        fraud_nodes = [idx for idx in all_labeled_idx if idx in valid_nodes][:50]
    num_fraud_nodes = len(fraud_nodes)
    print(f"Processing {num_fraud_nodes} fraud nodes.")

    adj_matrix_sparse = nx.adjacency_matrix(G_nx)
    fraud_indices = np.array([node_to_idx[node] for node in fraud_nodes if node in node_to_idx])
    if len(fraud_indices) == 0:
        print("Error: No valid fraud indices after mapping. Using first 50 indices.")
        fraud_indices = np.arange(min(50, adj_matrix_sparse.shape[0]))
    sub_adj_matrix = adj_matrix_sparse[fraud_indices, :][:, fraud_indices].tocsr()

    class GraphAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, latent_dim))
            self.decoder = nn.Sequential(nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, input_dim))

        def forward(self, x):
            latent = self.encoder(x)
            return self.decoder(latent), latent

    features = torch.eye(num_fraud_nodes, dtype=torch.float)
    integrated_features = torch.sparse.mm(torch.from_numpy(sub_adj_matrix.toarray()).float(), features)
    gae = GraphAutoencoder(input_dim=num_fraud_nodes, hidden_dim=10, latent_dim=5)
    optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    for epoch in range(20):
        optimizer.zero_grad()
        reconstructed, latent = gae(integrated_features)
        loss = criterion(reconstructed, integrated_features)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"GAE Epoch {epoch}, Loss: {loss.item():.4f}")

    latent_reps = latent.detach().numpy()
    pairwise_similarity = cosine_similarity(latent_reps)
    row_indices, col_indices = np.where(pairwise_similarity > 0.7)
    values = pairwise_similarity[row_indices, col_indices]
    reconstructed_sub_adj = csr_matrix((values, (row_indices, col_indices)), shape=(num_fraud_nodes, num_fraud_nodes))
    full_shape = (graph.num_nodes(), graph.num_nodes())
    full_row_indices = np.array(fraud_nodes)[row_indices]
    full_col_indices = np.array(fraud_nodes)[col_indices]
    reconstructed_adj = csr_matrix((values, (full_row_indices, full_col_indices)), shape=full_shape)
    reconstructed_adj_coo = reconstructed_adj.tocoo()
    indices = torch.tensor([reconstructed_adj_coo.row, reconstructed_adj_coo.col], dtype=torch.long)
    values = torch.tensor(reconstructed_adj_coo.data, dtype=torch.float)
    reconstructed_adj_torch = torch.sparse_coo_tensor(indices, values, full_shape).coalesce()

    edge_index = torch.stack(graph.edges(etype=graph.etypes[0]), dim=0)
    data_gcn = Data(x=torch.zeros(graph.num_nodes(), 8, dtype=torch.float), edge_index=edge_index)

    class GCN(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.conv1 = GCNConv(input_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, output_dim)

        def forward(self, data):
            x = F.relu(self.conv1(data.x, data.edge_index))
            return self.conv2(x, data.edge_index)

    gcn = GCN(input_dim=8, hidden_dim=16, output_dim=10)
    optimizer = torch.optim.Adam(gcn.parameters(), lr=0.01)
    for epoch in range(3):
        optimizer.zero_grad()
        embeddings = gcn(data_gcn)
        loss = torch.mean(embeddings)
        loss.backward()
        optimizer.step()
        if epoch % 1 == 0:
            print(f"GCN Epoch {epoch}, Loss: {loss.item():.4f}")

    gcn_embeddings = embeddings.detach().numpy()
    print(f"GCN embeddings generated for {len(gcn_embeddings)} nodes.")
    return gcn_embeddings, reconstructed_adj_torch


# DataLoader
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


# GraphProcessor with All Data
class GraphProcessor:
    @staticmethod
    def convert_to_pyg_format(graph, features):
        print("Converting DGL graph to PyTorch Geometric format...")
        edge_types = graph.etypes
        print(f"Graph is heterogeneous. Available edge types: {edge_types}")
        edge_type = edge_types[0]
        edge_index = torch.stack(graph.edges(etype=edge_type), dim=0)  # Use all edges
        print(f"Using edge type: {edge_type}, {edge_index.shape[1]} edges selected")
        data = Data(x=features, edge_index=edge_index)
        print(f"PyG Data created: {data.num_nodes} nodes, {data.num_edges} edges")
        return data

    @staticmethod
    def prepare_masks(labels, train_ratio=0.8):
        print("Preparing train/test masks using all data...")
        all_nodes = list(range(len(labels)))  # All 45,954 nodes
        train_mask, test_mask = train_test_split(all_nodes, train_size=train_ratio, random_state=42)

        # Identify labeled nodes within train and test sets
        labeled_idx = torch.where(labels != -1)[0].tolist()
        train_labeled = [i for i in train_mask if i in labeled_idx]
        test_labeled = [i for i in test_mask if i in labeled_idx]

        train_labels = [labels[i].item() for i in train_labeled]
        test_labels = [labels[i].item() for i in test_labeled]
        print(f"Total nodes: {len(all_nodes)}, Train mask: {len(train_mask)} nodes, Test mask: {len(test_mask)} nodes")
        print(f"Labeled train nodes: {len(train_labeled)}, Label distribution: {np.bincount(train_labels)}")
        print(f"Labeled test nodes: {len(test_labeled)}, Label distribution: {np.bincount(test_labels)}")
        return train_mask, test_mask, train_labeled, test_labeled


# Model Classes
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
        self.qkv_proj = nn.Linear(input_size, input_size * 3)
        if torch.is_tensor(adj_matrix) and adj_matrix.is_sparse:
            indices = adj_matrix.indices()
            self.src = indices[0]
            self.tgt = indices[1]
        else:
            self.src, self.tgt = adj_matrix.nonzero(as_tuple=True)
        self.ffn = nn.Sequential(nn.Linear(input_size, 4 * input_size), nn.ReLU(),
                                 nn.Linear(4 * input_size, input_size))
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, node_indices):
        batch_size, num_nodes, _ = x.size()
        if batch_size != 1:
            raise ValueError("Sparse attention assumes batch_size=1")
        index_map = {idx.item(): i for i, idx in enumerate(node_indices)}
        mask = torch.tensor(
            [idx.item() in index_map and self.tgt[i].item() in index_map for i, idx in enumerate(self.src)],
            dtype=torch.bool)
        src_sub, tgt_sub = self.src[mask], self.tgt[mask]
        src_mapped = torch.tensor([index_map[idx.item()] for idx in src_sub], dtype=torch.long)
        tgt_mapped = torch.tensor([index_map[idx.item()] for idx in tgt_sub], dtype=torch.long)
        qkv = self.qkv_proj(x).reshape(num_nodes, 3 * self.input_size)
        q, k, v = qkv.split(self.input_size, dim=-1)
        q = q.view(num_nodes, self.num_heads, self.head_dim)
        k = k.view(num_nodes, self.num_heads, self.head_dim)
        v = v.view(num_nodes, self.num_heads, self.head_dim)
        if src_mapped.size(0) == 0:
            x = self.norm1(x.squeeze(0))
            ffn_out = self.ffn(x)
            return self.norm2(x + ffn_out).unsqueeze(0)
        q_edges, k_edges = q[src_mapped], k[tgt_mapped]
        attn_scores = (q_edges * k_edges).sum(dim=-1) / (self.head_dim ** 0.5)
        attn_weights = torch.zeros_like(attn_scores)
        for head in range(self.num_heads):
            scores_head = attn_scores[:, head]
            weights_head = torch.zeros(num_nodes, device=x.device)
            weights_head.index_add_(0, src_mapped, F.softmax(scores_head, dim=0))
            attn_weights[:, head] = weights_head[src_mapped]
        v_edges = v[tgt_mapped]
        out = torch.zeros(num_nodes, self.num_heads, self.head_dim, device=x.device)
        for head in range(self.num_heads):
            weighted_v = attn_weights[:, head].unsqueeze(-1) * v_edges[:, head, :]
            out[:, head, :].index_add_(0, src_mapped, weighted_v)
        out = out.view(num_nodes, self.input_size)
        out = self.dropout(out)
        x = self.norm1(x.squeeze(0) + out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out).unsqueeze(0)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class SpatialDependencyModel(nn.Module):
    def __init__(self, num_relations, hidden_size, mlp_hidden_size, adj_matrix):
        super().__init__()
        self.transformer = TransformerBlock(hidden_size, num_heads=4, adj_matrix=adj_matrix)
        self.mlp = MLP(hidden_size, mlp_hidden_size)
        self.adj_matrix = adj_matrix
        self.neighbors = {}
        if torch.is_tensor(adj_matrix) and adj_matrix.is_sparse:
            indices = adj_matrix.indices()
            values = adj_matrix.values()
            for i in range(indices.size(1)):
                src = indices[0, i].item()
                tgt = indices[1, i].item()
                weight = values[i].item()
                if src not in self.neighbors:
                    self.neighbors[src] = []
                self.neighbors[src].append((tgt, weight))

    def forward(self, node_embeddings, timestamp, relations, index_obtains):
        aggregated_reps = []
        valid_indices = []
        for vi in index_obtains:
            neighbor_list = self.neighbors.get(vi, [])
            if not neighbor_list:
                aggregated_reps.append(node_embeddings[vi])
                valid_indices.append(vi)
                continue
            neighbors, weights = zip(*neighbor_list)
            neighbors = torch.tensor(neighbors, dtype=torch.long)
            weights = torch.tensor(weights, dtype=torch.float)
            neighbor_emb = node_embeddings[neighbors]
            weights = weights.view(-1, 1)
            intra_agg = torch.mean(weights * neighbor_emb, dim=0)
            diff = node_embeddings[vi] - neighbor_emb
            inter_agg = torch.mean(diff, dim=0)
            combined = intra_agg + inter_agg
            if combined.dim() > 0:
                aggregated_reps.append(combined)
                valid_indices.append(vi)

        fused = torch.stack(aggregated_reps)
        fused = fused.unsqueeze(0)
        fused = self.transformer(fused, torch.tensor(valid_indices, dtype=torch.long))
        fused = fused.squeeze(0)
        prediction = self.mlp(fused)
        return prediction, fused, None, valid_indices


class TemporalSpatialModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_temporal_bases, num_relations, mlp_hidden_size, adj_matrix):
        super().__init__()
        self.temporal = TemporalDependencyModel(embedding_size, hidden_size, num_temporal_bases)
        self.spatial = SpatialDependencyModel(num_relations, hidden_size, mlp_hidden_size, adj_matrix)

    def forward(self, x, node_indices, timestamp, relations, index_obtains):
        temporal_emb = self.temporal(x, timestamp)
        return self.spatial(temporal_emb, timestamp, relations, index_obtains)


class CombinedModel(nn.Module):
    def __init__(self, input_dim, embedding_size, hidden_size, num_temporal_bases, num_relations, mlp_hidden_size,
                 adj_matrix):
        super().__init__()
        self.initial_conv = GCNConv(input_dim, embedding_size)
        self.temporal_spatial = TemporalSpatialModel(embedding_size, hidden_size, num_temporal_bases, num_relations,
                                                     mlp_hidden_size, adj_matrix)

    def forward(self, data, node_indices, timestamp, relations, index_obtains):
        x = F.relu(self.initial_conv(data.x, data.edge_index))
        return self.temporal_spatial(x, node_indices, timestamp, relations, index_obtains)


# Updated Training Function with All Data
def train_model(model, data, labels, train_mask, test_mask, train_labeled, test_labeled, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    train_subset = train_mask  # All training nodes (labeled + unlabeled)
    test_subset = test_mask    # All testing nodes (labeled + unlabeled)
    train_labels = torch.tensor([labels[i] for i in train_labeled], dtype=torch.float)
    print(f"Training on {len(train_subset)} nodes, Labeled nodes: {len(train_labeled)}, "
          f"Label distribution: {np.bincount(train_labels.int())}")

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred, _, _, idx = model(data, torch.tensor(train_subset), torch.tensor(123.45), [1], train_subset)
        if pred is None:
            print(f"Epoch {epoch + 1}: No predictions generated")
            continue
        # Compute loss only on labeled nodes
        labeled_mask = torch.tensor([i in train_labeled for i in train_subset], dtype=torch.bool)
        pred_labeled = pred[labeled_mask]
        loss = loss_fn(pred_labeled.squeeze(), train_labels)
        loss.backward()
        optimizer.step()
        accuracy = ((pred_labeled.squeeze() > 0).float() == train_labels).float().mean().item()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Accuracy (Labeled): {accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        pred, _, _, idx = model(data, torch.tensor(test_subset), torch.tensor(123.45), [1], test_subset)
        if pred is None:
            print("No predictions generated during evaluation")
            return
        test_labels = torch.tensor([labels[i] for i in test_labeled], dtype=torch.float)
        labeled_mask = torch.tensor([i in test_labeled for i in test_subset], dtype=torch.bool)
        pred_labeled = pred[labeled_mask]
        pred_binary = (pred_labeled.squeeze() > 0).float()
        test_labels_np = test_labels.numpy()
        pred_binary_np = pred_binary.numpy()

        # Calculate metrics
        accuracy = (pred_binary == test_labels).float().mean().item()
        precision = precision_score(test_labels_np, pred_binary_np)
        recall = recall_score(test_labels_np, pred_binary_np)
        f1_micro = f1_score(test_labels_np, pred_binary_np, average='micro')
        f1_macro = f1_score(test_labels_np, pred_binary_np, average='macro')
        unique_classes = np.unique(test_labels_np)
        auc = roc_auc_score(test_labels_np, torch.sigmoid(pred_labeled.squeeze()).numpy()) if len(unique_classes) > 1 else None
        cm = confusion_matrix(test_labels_np, pred_binary_np)

        # Print detailed evaluation
        print("\nEvaluation Metrics (on labeled test nodes):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score (Micro): {f1_micro:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        print(f"AUC: {auc:.4f}" if auc is not None else "AUC: Not defined (only one class present)")
        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(classification_report(test_labels_np, pred_binary_np, target_names=['Non-Fraud', 'Fraud']))


# Main Function
def main():
    graph, features, labels, num_classes = DataLoader.load_yelpchi_data()
    centrality_embeddings = compute_centrality_features(graph)
    train_mask, test_mask, train_labeled, test_labeled = GraphProcessor.prepare_masks(labels)

    fraudwalk_embeddings = compute_fraudwalk_embeddings(graph, train_mask=train_mask, inductive_nodes=test_mask)
    gcn_embeddings, reconstructed_adj = reconstruct_adj_and_gcn(graph, labels, train_mask)
    print("Step 4: Combining embeddings...")
    num_nodes = graph.num_nodes()
    combined_embeddings = np.zeros((num_nodes, 23))
    for i in range(num_nodes):
        centrality = centrality_embeddings.get(i, [0] * 8)
        fraudwalk = fraudwalk_embeddings.get(i, np.zeros(5))
        gcn = gcn_embeddings[i] if i < len(gcn_embeddings) else np.zeros(10)
        combined_embeddings[i] = np.concatenate([centrality, fraudwalk, gcn])

    data = GraphProcessor.convert_to_pyg_format(graph, torch.from_numpy(combined_embeddings).float())
    adj_matrix = reconstructed_adj
    print(f"Using reconstructed adjacency matrix: {adj_matrix.shape}, {adj_matrix._nnz()} edges")

    model = CombinedModel(
        input_dim=23,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_temporal_bases=NUM_TEMPORAL_BASES,
        num_relations=NUM_RELATIONS,
        mlp_hidden_size=EMBEDDING_SIZE,
        adj_matrix=adj_matrix
    )
    train_model(model, data, labels, train_mask, test_mask, train_labeled, test_labeled)


if __name__ == "__main__":
    main()