import pandas as pd
import networkx as nx
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn import Linear, ReLU, MSELoss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam
import os
from collections.abc import Mapping

def autoencoder_pairings(input_ingredient, file_path = 'non_duplicate_ingredients.csv',model_path='gnn_autoencoder.pkl', top_n=5):

    data = pd.read_csv(file_path)
    grouped_data = data.groupby('Recipe ID')['Ingredient'].apply(list).tolist()
    G = nx.Graph()
    for recipe in grouped_data:
        for i, ingredient in enumerate(recipe):
            for j in range(i + 1, len(recipe)):
                G.add_edge(ingredient, recipe[j])
                
    if not os.path.exists(model_path):
        return f"Model file '{model_path}' not found. Train the model first."

    checkpoint = torch.load(model_path)
    node_mapping = checkpoint['node_mapping']
    label_encoder = checkpoint['label_encoder']

    if input_ingredient not in node_mapping:
        return f"Ingredient '{input_ingredient}' not found in the dataset."

    num_nodes = len(node_mapping)
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges], dtype=torch.long).t()
    x = torch.eye(num_nodes, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index)
    class GNN_AutoEncoder(torch.nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(GNN_AutoEncoder, self).__init__()
            self.encoder_conv = GCNConv(input_dim, hidden_dim)
            self.decoder_conv = GCNConv(hidden_dim, output_dim)
            self.output_layer = Linear(output_dim, input_dim)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = self.encoder_conv(x, edge_index)
            x = ReLU()(x)
            x = self.decoder_conv(x, edge_index)
            x = ReLU()(x)
            x = self.output_layer(x)
            return x

    model = GNN_AutoEncoder(x.shape[1], 64, 32)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    with torch.no_grad():
        embeddings = model.encoder_conv(graph_data.x, graph_data.edge_index)
    input_id = node_mapping[input_ingredient]
    input_embedding = embeddings[input_id].unsqueeze(0)
    distances = torch.norm(embeddings - input_embedding, dim=1)
    closest_indices = torch.argsort(distances)[:top_n + 1]  # Include the input itself
    closest_indices = closest_indices[closest_indices != input_id]  # Exclude the input itself
    recommendations = [list(node_mapping.keys())[i] for i in closest_indices]

    return recommendations

# print(autoencoder_pairings('butter'))