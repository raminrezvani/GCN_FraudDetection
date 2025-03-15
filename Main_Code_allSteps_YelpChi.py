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
import networkx as nx
import pandas as pd
from datetime import datetime, timedelta
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import random

import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity


from collections import defaultdict
from datetime import datetime, timedelta
from gensim.models import Word2Vec
from multiprocessing import Pool
# Constants
EMBEDDING_SIZE = 16
HIDDEN_SIZE = 16
NUM_TEMPORAL_BASES = 4
NUM_RELATIONS = 3


# Step 1: Compute Only Degree Centrality
def compute_centrality_features(graph):
    print("Step 1: Computing degree centrality (only)...")
    G_nx = nx.DiGraph()
    src, tgt = graph.edges(etype=graph.etypes[0])
    edge_list = list(zip(src.numpy(), tgt.numpy()))
    G_nx.add_edges_from(edge_list)

    # Compute only degree centrality
    degree = nx.degree_centrality(G_nx)

    # Create embeddings with degree only (pad with zeros for consistency)
    centrality_embeddings = {}
    for node in G_nx.nodes():
        centrality_embeddings[node] = [degree.get(node, 0), 0, 0, 0, 0, 0, 0, 0]  # 8D for compatibility
    print(f"Degree centrality computed for {len(centrality_embeddings)} nodes.")
    return centrality_embeddings


# Step 2: Probabilistic FraudWalk Embeddings (Unchanged)
# Define the walk function at the module level to make it picklable
# Global variable to share the graph (will be set in the main process)


def compute_fraudwalk_embeddings(graph):
    print("Step 2: Computing FraudWalk embeddings (simplified)...")

    # Convert DGL graph to NetworkX
    src, tgt = graph.edges(etype=graph.etypes[0])
    edge_list = list(zip(src.numpy(), tgt.numpy()))
    G = nx.DiGraph()
    G.add_edges_from(edge_list)

    # Simple random walk function
    def simple_random_walk(start_node, walk_length=10):
        walk = [start_node]
        current = start_node
        for _ in range(walk_length - 1):
            neighbors = list(G.neighbors(current))
            if not neighbors:
                break
            current = random.choice(neighbors)
            walk.append(current)
        return walk

    # Generate a small number of walks
    num_walks = 100  # Reduced from 500
    nodes = list(G.nodes())
    walks = [simple_random_walk(random.choice(nodes)) for _ in range(num_walks)]

    # Train lightweight Word2Vec model
    model = Word2Vec(
        sentences=walks,
        vector_size=5,  # Very small embedding size
        window=3,
        sg=1,  # Skip-gram
        min_count=1,
        workers=4  # Use multiple CPU cores
    )

    # Extract embeddings
    fraudwalk_embeddings = {node: model.wv[node] for node in model.wv.index_to_key if str(node) != 'nan'}
    print(f"FraudWalk embeddings generated for {len(fraudwalk_embeddings)} nodes.")
    return fraudwalk_embeddings


# Step 3: Fraud Subgraph, GAE, and GCN (Unchanged)
def reconstruct_adj_and_gcn(graph, labels, train_mask):
    print("Step 3: Reconstructing adjacency matrix with GAE and GCN (optimized)...")

    # Identify fraud nodes from train_mask
    fraud_nodes = [idx for idx in train_mask if labels[idx] == 1]
    num_fraud_nodes = len(fraud_nodes)
    print(f"Processing {num_fraud_nodes} fraud nodes.")

    # Convert to NetworkX and get sparse adjacency matrix
    G_nx = nx.DiGraph()
    src, tgt = graph.edges(etype=graph.etypes[0])
    edge_list = list(zip(src.numpy(), tgt.numpy()))
    G_nx.add_edges_from(edge_list)
    adj_matrix_sparse = nx.adjacency_matrix(G_nx)  # Keep as sparse (csr_matrix)

    # Extract fraud subgraph as sparse matrix
    fraud_indices = np.array(fraud_nodes)
    sub_adj_matrix = adj_matrix_sparse[fraud_indices, :][:, fraud_indices].tocsr()  # Sparse fraud subgraph

    # GAE with sparse input
    class GraphAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            )

        def forward(self, x):
            latent = self.encoder(x)
            return self.decoder(latent), latent

    # Use identity matrix as features for fraud nodes
    features = torch.eye(num_fraud_nodes, dtype=torch.float)
    integrated_features = torch.sparse.mm(
        torch.from_numpy(sub_adj_matrix.toarray()).float(), features
    )  # Temporary dense conversion for simplicity

    gae = GraphAutoencoder(input_dim=num_fraud_nodes, hidden_dim=10, latent_dim=5)
    optimizer = torch.optim.Adam(gae.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # Train GAE
    for epoch in range(100):
        optimizer.zero_grad()
        reconstructed, latent = gae(integrated_features)
        loss = criterion(reconstructed, integrated_features)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"GAE Epoch {epoch}, Loss: {loss.item():.4f}")

    # Get latent representations and compute similarity
    latent_reps = latent.detach().numpy()
    pairwise_similarity = cosine_similarity(latent_reps)  # Shape: (num_fraud_nodes, num_fraud_nodes)

    # Reconstruct sparse adjacency matrix
    row_indices, col_indices = np.where(pairwise_similarity > 0.5)
    values = pairwise_similarity[row_indices, col_indices]
    reconstructed_sub_adj = csr_matrix((values, (row_indices, col_indices)), shape=(num_fraud_nodes, num_fraud_nodes))

    # Map back to full graph size (sparse)
    full_shape = (graph.num_nodes(), graph.num_nodes())
    full_row_indices = fraud_indices[row_indices]
    full_col_indices = fraud_indices[col_indices]
    reconstructed_adj = csr_matrix((values, (full_row_indices, full_col_indices)), shape=full_shape)

    # Convert CSR to COO for PyTorch sparse tensor
    reconstructed_adj_coo = reconstructed_adj.tocoo()
    indices = torch.tensor([reconstructed_adj_coo.row, reconstructed_adj_coo.col], dtype=torch.long)
    values = torch.tensor(reconstructed_adj_coo.data, dtype=torch.float)
    reconstructed_adj_torch = torch.sparse_coo_tensor(indices, values, full_shape)

    # GCN with sparse input
    edge_index = indices  # Use reconstructed edges directly
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
        loss = torch.mean(embeddings)  # Dummy loss
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f"GCN Epoch {epoch}, Loss: {loss.item():.4f}")

    gcn_embeddings = embeddings.detach().numpy()
    print(f"GCN embeddings generated for {len(gcn_embeddings)} nodes.")
    return gcn_embeddings, reconstructed_adj_torch
# Existing Classes (Unchanged)
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
        edge_types = graph.etypes
        print(f"Graph is heterogeneous. Available edge types: {edge_types}")
        edge_type = edge_types[0]
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
        train_mask, test_mask = train_test_split(labeled_idx, train_size=train_ratio, random_state=42)
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
        self.qkv_proj = nn.Linear(input_size, input_size * 3)
        self.src, self.tgt = adj_matrix.nonzero(as_tuple=True)
        self.ffn = nn.Sequential(
            nn.Linear(input_size, 4 * input_size),
            nn.ReLU(),
            nn.Linear(4 * input_size, input_size)
        )
        self.norm1 = nn.LayerNorm(input_size)
        self.norm2 = nn.LayerNorm(input_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, node_indices):
        batch_size, num_nodes, _ = x.size()
        if batch_size != 1:
            raise ValueError("Sparse attention assumes batch_size=1")

        index_map = {idx.item(): i for i, idx in enumerate(node_indices)}
        mask = torch.tensor([idx.item() in index_map and self.tgt[i].item() in index_map
                             for i, idx in enumerate(self.src)], dtype=torch.bool)
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

    def forward(self, node_embeddings, timestamp, relations, index_obtains):
        aggregated_reps = []
        valid_indices = []
        for vi in index_obtains:
            neighbors = torch.where(self.adj_matrix[vi] > 0)[0]
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

        if not aggregated_reps:
            fused = node_embeddings[torch.tensor(index_obtains)].unsqueeze(0)
            prediction = self.mlp(fused.squeeze(0))
            return prediction, fused, None, index_obtains

        fused = torch.stack(aggregated_reps).unsqueeze(0)
        fused = self.transformer(fused, torch.tensor(valid_indices, dtype=torch.long))
        prediction = self.mlp(fused.squeeze(0))
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


def train_model(model, data, labels, train_mask, test_mask, num_epochs=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
    train_labels = [labels[i] for i in train_mask]
    print(f"Training label distribution: {np.bincount(train_labels)}")

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        pred, _, _, idx = model(data, torch.arange(data.num_nodes), torch.tensor(123.45), [1], train_mask.copy())
        if pred is None:
            print(f"Epoch {epoch + 1}: No predictions generated")
            continue
        masked_pred = pred.squeeze()
        masked_labels = torch.tensor([labels[i] for i in idx], dtype=torch.float)
        loss = loss_fn(masked_pred, masked_labels)
        predicted_labels = (masked_pred > 0).long()
        accuracy = (predicted_labels == masked_labels).float().mean().item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}, Train Accuracy: {accuracy:.4f}")

    model.eval()
    with torch.no_grad():
        pred, _, _, idx = model(data, torch.arange(data.num_nodes), torch.tensor(123.45), [1], test_mask.copy())
        if pred is None:
            print("No predictions generated during evaluation")
            return
        masked_labels = torch.tensor([labels[i] for i in idx])
        masked_pred = pred.squeeze()
        predicted_labels = (masked_pred > 0).long()
        accuracy = (predicted_labels == masked_labels).float().mean().item()
        f1 = f1_score(masked_labels.numpy(), predicted_labels.numpy(), average='macro')
        probs = torch.sigmoid(masked_pred).numpy()
        auc = roc_auc_score(masked_labels.numpy(), probs)
        cm = confusion_matrix(masked_labels.numpy(), predicted_labels.numpy())
        sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        gmeans = (sensitivity * specificity) ** 0.5
        print(f"Evaluation - Accuracy: {accuracy:.4f}, F1-macro: {f1:.4f}, AUC: {auc:.4f}, G-means: {gmeans:.4f}")

def main():
    graph, features, labels, num_classes = DataLoader.load_yelpchi_data()

    # Step 1: Compute Degree Centrality Only
    centrality_embeddings = compute_centrality_features(graph)

    # Step 2: FraudWalk Embeddings
    fraudwalk_embeddings = compute_fraudwalk_embeddings(graph)

    # Step 3: Fraud Subgraph, GAE, and GCN
    train_mask, test_mask = GraphProcessor.prepare_masks(labels)
    gcn_embeddings, reconstructed_adj = reconstruct_adj_and_gcn(graph, labels, train_mask)

    # Step 4: Combine Embeddings
    print("Step 4: Combining embeddings...")
    num_nodes = graph.num_nodes()
    combined_embeddings = np.zeros((num_nodes, 23))  # 8 (centrality) + 5 (fraudwalk) + 10 (gcn)
    for i in range(num_nodes):
        centrality = centrality_embeddings.get(i, [0] * 8)  # 8D
        fraudwalk = fraudwalk_embeddings.get(i, np.zeros(5))  # 5D
        gcn = gcn_embeddings[i] if i < len(gcn_embeddings) else np.zeros(10)  # 10D
        combined_embeddings[i] = np.concatenate([centrality, fraudwalk, gcn])

    data = GraphProcessor.convert_to_pyg_format(graph, torch.from_numpy(combined_embeddings).float())
    adj_matrix = reconstructed_adj
    print(f"Using reconstructed adjacency matrix: {adj_matrix.shape}, {adj_matrix._nnz()} edges")  # Use _nnz() for sparse tensor

    model = CombinedModel(
        input_dim=23,  # Corrected to match combined embedding size
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