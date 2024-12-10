import csv
import pandas as pd
from itertools import combinations
from collections import defaultdict
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import networkx as nx
import torch
import pandas as pd
import networkx as nx
import torch
from sklearn.preprocessing import LabelEncoder
from torch.nn import Linear, ReLU, MSELoss
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch.optim import Adam
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import Counter
import pyfpgrowth



def find_similar_ingredients(input_ingredient, min_support=0.02, top_n=5):
    df_ingredients = pd.read_csv('non_duplicate_ingredients.csv')

    recipes = df_ingredients.groupby('Recipe ID')['Ingredient'].apply(set).tolist()

    min_count = int(min_support * len(recipes))
    
    def count_itemsets(recipes, itemsets):
        itemset_counts = defaultdict(int)
        for recipe in recipes:
            for itemset in combinations(recipe, len(itemsets[0])):
                sorted_itemset = tuple(sorted(itemset))
                itemset_counts[sorted_itemset] += 1
        return itemset_counts

    def filter_itemsets(itemset_counts, min_count):
        return {itemset: count for itemset, count in itemset_counts.items() if count >= min_count}

    item_counts = defaultdict(int)
    for recipe in recipes:
        for item in recipe:
            item_counts[item] += 1

    # Filter out infrequent items
    frequent_itemsets_1 = {item: count for item, count in item_counts.items() if count >= min_count}
    frequent_items_1 = set(frequent_itemsets_1.keys())

    # Filter recipes to include only frequent items
    filtered_recipes = [recipe.intersection(frequent_items_1) for recipe in recipes]

    # Count frequent itemsets of size 2
    itemset_counts_2 = count_itemsets(filtered_recipes, list(combinations(frequent_items_1, 2)))
    frequent_itemsets_2 = filter_itemsets(itemset_counts_2, min_count)

    # Count frequent itemsets of size 3
    itemset_counts_3 = count_itemsets(filtered_recipes, list(combinations(frequent_items_1, 3)))
    frequent_itemsets_3 = filter_itemsets(itemset_counts_3, min_count)

    # Dictionary to store frequency of similar ingredients
    similar_ingredient_counts = defaultdict(int)

    # Check size 1 itemsets
    if input_ingredient in frequent_items_1:
        for item in frequent_items_1:
            if item != input_ingredient:
                similar_ingredient_counts[item] += item_counts[item]

    # Check size 2 itemsets
    for itemset, count in frequent_itemsets_2.items():
        if input_ingredient in itemset:
            for item in itemset:
                if item != input_ingredient:
                    similar_ingredient_counts[item] += count

    # Check size 3 itemsets
    for itemset, count in frequent_itemsets_3.items():
        if input_ingredient in itemset:
            for item in itemset:
                if item != input_ingredient:
                    similar_ingredient_counts[item] += count

    # Sort similar ingredients by frequency and return the top N
    sorted_similar_ingredients = sorted(similar_ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # Return the top N similar ingredients
    return [ingredient for ingredient, _ in sorted_similar_ingredients]


def cosine_similarity_top(input_ingredient):
    file_path = 'non_duplicate_ingredients.csv'
    data = pd.read_csv(file_path)

    grouped_data = data.groupby('Recipe ID')['Ingredient'].apply(list)

    mlb = MultiLabelBinarizer()
    ingredient_matrix = mlb.fit_transform(grouped_data)
    ingredient_df = pd.DataFrame(ingredient_matrix, columns=mlb.classes_, index=grouped_data.index)

    ingredient_similarity = cosine_similarity(ingredient_df.T)  
    ingredient_similarity_df = pd.DataFrame(ingredient_similarity,
                                            index=ingredient_df.columns,
                                            columns=ingredient_df.columns)
    
    if input_ingredient not in ingredient_df.columns:
        print(f"Ingredient {input_ingredient} not found in the dataset.")
        return []

    similar_scores = ingredient_similarity_df[input_ingredient].sort_values(ascending=False)
    similar_ingredients = similar_scores.iloc[1:6].index.tolist()  # Exclude the ingredient itself
    return similar_ingredients


def get_knn_similar_ingredients(ingredient, top_n=5):
    file_path = 'non_duplicate_ingredients.csv'
    data = pd.read_csv(file_path)
    grouped_data = data.groupby('Recipe ID')['Ingredient'].apply(list)

    mlb = MultiLabelBinarizer()
    ingredient_matrix = mlb.fit_transform(grouped_data)
    ingredient_df = pd.DataFrame(ingredient_matrix, columns=mlb.classes_, index=grouped_data.index)
    knn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    knn.fit(ingredient_df.T)  

    ingredient_index = mlb.classes_.tolist().index(ingredient)
    distances, indices = knn.kneighbors([ingredient_df.T.iloc[ingredient_index].values], n_neighbors=top_n+1)
    similar_ingredients = [mlb.classes_[i] for i in indices[0][1:]]  # Exclude the ingredient itself
    return similar_ingredients


def autoencoder_pairings(input_ingredient, file_path = 'non_duplicate_ingredients.csv',model_path='gnn_autoencoder.pkl', top_n=5):
    print("inside ae")
    data = pd.read_csv(file_path)
    grouped_data = data.groupby('Recipe ID')['Ingredient'].apply(list).tolist()

    G = nx.Graph()
    for recipe in grouped_data:
        for i, ingredient in enumerate(recipe):
            for j in range(i + 1, len(recipe)):
                G.add_edge(ingredient, recipe[j])
    import os
    if not os.path.exists(model_path):
        return f"Model file '{model_path}' not found. Train the model first."

    # Load the saved model and metadata
    checkpoint = torch.load(model_path)
    node_mapping = checkpoint['node_mapping']
    label_encoder = checkpoint['label_encoder']

    # Check if the ingredient exists in the node mapping
    if input_ingredient not in node_mapping:
        return f"Ingredient '{input_ingredient}' not found in the dataset."

    # Rebuild the graph
    num_nodes = len(node_mapping)
    edge_index = torch.tensor([[node_mapping[u], node_mapping[v]] for u, v in G.edges], dtype=torch.long).t()
    x = torch.eye(num_nodes, dtype=torch.float)
    graph_data = Data(x=x, edge_index=edge_index)

    # Define and load the model
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

    # Extract embeddings
    with torch.no_grad():
        embeddings = model.encoder_conv(graph_data.x, graph_data.edge_index)

    # Get recommendations
    input_id = node_mapping[input_ingredient]
    input_embedding = embeddings[input_id].unsqueeze(0)
    distances = torch.norm(embeddings - input_embedding, dim=1)
    closest_indices = torch.argsort(distances)[:top_n + 1]  # Include the input itself
    closest_indices = closest_indices[closest_indices != input_id]  # Exclude the input itself
    recommendations = [list(node_mapping.keys())[i] for i in closest_indices]

    return recommendations


def find_similar_ingredients_svd(input_ingredient, top_n=5):
    df_ingredients = pd.read_csv('non_duplicate_ingredients.csv')

    unique_ingredients = sorted(df_ingredients['Ingredient'].unique())
    ingredient_to_idx = {ingredient: idx for idx, ingredient in enumerate(unique_ingredients)}
    
    # Group recipes by 'Recipe ID' and create a matrix
    recipes = df_ingredients.groupby('Recipe ID')['Ingredient'].apply(list)
    co_occurrence_matrix = np.zeros((len(unique_ingredients), len(unique_ingredients)))
    
    for recipe in recipes:
        for i in range(len(recipe)):
            for j in range(i + 1, len(recipe)):
                idx_i = ingredient_to_idx[recipe[i]]
                idx_j = ingredient_to_idx[recipe[j]]
                co_occurrence_matrix[idx_i][idx_j] += 1
                co_occurrence_matrix[idx_j][idx_i] += 1
    
    svd = TruncatedSVD(n_components=50, random_state=42)
    reduced_matrix = svd.fit_transform(co_occurrence_matrix)
    
    if input_ingredient not in ingredient_to_idx:
        raise ValueError(f"Ingredient '{input_ingredient}' not found in the dataset.")
    
    input_idx = ingredient_to_idx[input_ingredient]
    
    
    similarities = cosine_similarity([reduced_matrix[input_idx]], reduced_matrix)[0]
    similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]  # Exclude the input ingredient itself
    similar_ingredients = [(unique_ingredients[idx], similarities[idx]) for idx in similar_indices]
    
    return [ingredient for ingredient, _ in similar_ingredients]

def find_similar_ingredients_fp_growth(input_ingredient, min_support=0.02, top_n=5):
    df_ingredients = pd.read_csv('non_duplicate_ingredients.csv')
    transactions = df_ingredients.groupby('Recipe ID')['Ingredient'].apply(list).tolist()
    min_count = int(min_support * len(transactions))
    patterns = pyfpgrowth.find_frequent_patterns(transactions, min_count)
    rules = pyfpgrowth.generate_association_rules(patterns, confidence_threshold = 0.5)
    similar_ingredient_counts = defaultdict(int)
    for pattern, count in patterns.items():
        if input_ingredient in pattern:
            for ingredient in pattern:
                if ingredient != input_ingredient:
                    similar_ingredient_counts[ingredient] += count
    sorted_similar_ingredients = sorted(similar_ingredient_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return [ingredient for ingredient, _ in sorted_similar_ingredients]



def recommend_similar_ingredients_dbscan(input_ingredient, file_path='non_duplicate_ingredients.csv', top_n=5):
    """
    Recommends the top N similar ingredients to the input ingredient based on co-occurrence in recipes.

    Parameters:
    - input_ingredient (str): The ingredient to find pairings for.
    - file_path (str): Path to the CSV file containing ingredients data.
    - top_n (int): Number of top similar ingredients to return.

    Returns:
    - List of tuples containing similar ingredients and their occurrence counts,
      or a message string if no pairings are found.
    """
    try:
        data = pd.read_csv(file_path)
    except FileNotFoundError:
        return f"File not found at path: {file_path}"
    except Exception as e:
        return f"An error occurred while reading the file: {e}"

    if not {'Recipe ID', 'Ingredient'}.issubset(data.columns):
        return "Input CSV must contain 'Recipe ID' and 'Ingredient' columns."

    grouped_data = data.groupby('Recipe ID')['Ingredient'].apply(lambda x: ' '.join(x)).reset_index()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(grouped_data['Ingredient'])
    dbscan = DBSCAN(eps=0.5, min_samples=5, metric='cosine')
    grouped_data['Cluster'] = dbscan.fit_predict(tfidf_matrix)

    def extract_cluster_ingredients(cluster_id):
        return grouped_data[grouped_data['Cluster'] == cluster_id]['Ingredient']

    clusters_with_ingredient = grouped_data[grouped_data['Ingredient'].str.contains(input_ingredient, case=False, na=False)]

    if clusters_with_ingredient.empty:
        return f"Ingredient '{input_ingredient}' not found in any cluster."

    cluster_ids = clusters_with_ingredient['Cluster'].unique()

    all_cluster_ingredients = pd.Series(dtype=str)

    for cluster_id in cluster_ids:
        cluster_ingredients = extract_cluster_ingredients(cluster_id)
        all_cluster_ingredients = pd.concat([all_cluster_ingredients, cluster_ingredients], ignore_index=True)

    all_ingredients = ' '.join(all_cluster_ingredients)
    individual_ingredients = all_ingredients.split()

    ingredient_combinations = Counter(
        comb for recipe in all_cluster_ingredients for comb in combinations(set(recipe.split()), 2)
    )

    relevant_pairings = {
        pair: count for pair, count in ingredient_combinations.items()
        if input_ingredient.lower() in (pair[0].lower(), pair[1].lower())
    }

    if not relevant_pairings:
        return f"No pairings found for ingredient '{input_ingredient}'."

    sorted_pairings = sorted(relevant_pairings.items(), key=lambda x: x[1], reverse=True)[:top_n]

    similar_ingredients = []
    for pair, count in sorted_pairings:
        ingredient = pair[0] if pair[1].lower() == input_ingredient.lower() else pair[1]
        similar_ingredients.append((ingredient, count))
    a = []
    for i in similar_ingredients:
        a.append(i[0])
    return a


# input_ingredient = input("Enter an ingredient for pairing recommendations: ").strip()
# pairings = recommend_similar_ingredients(input_ingredient)

# if isinstance(pairings, str):
#     print(pairings)
# else:
#     print(pairings)

# # Display results
# if isinstance(pairings, str):
#     print(pairings)
# else:
#     print(f"Top {len(pairings)} similar ingredients to '{input_ingredient}':")
#     for ingredient, count in pairings:
#         print(f"{ingredient}: {count} occurrences")
