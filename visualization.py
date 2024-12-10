import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import io
import base64

def plot_top_cooccurring_pairs(file_path, top_n=20):
    """
    Plots a graph of the top N co-occurring ingredient pairs from a CSV file,
    ensuring that the graph contains exactly N nodes corresponding to the top N ingredients.

    Parameters:
    - file_path (str): Path to the CSV file containing recipe data.
                       The CSV should have at least two columns: 'Recipe ID' and 'Ingredient'.
    - top_n (int): Number of top ingredients and their co-occurring pairs to visualize.

    Returns:
    - str: Base64-encoded PNG image of the graph.
    """
    
    # ----------------------- #
    #       Load Data         #
    # ----------------------- #
    
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except pd.errors.EmptyDataError:
        print("Error: The provided CSV file is empty.")
        return None
    except pd.errors.ParserError:
        print("Error: The provided CSV file is malformed.")
        return None

    # Check if required columns exist
    required_columns = {'Recipe ID', 'Ingredient'}
    if not required_columns.issubset(df.columns):
        print(f"Error: CSV file must contain the following columns: {required_columns}")
        return None

    # ----------------------- #
    #   Identify Top N Ingredients #
    # ----------------------- #
    
    # Count the frequency of each ingredient across all recipes
    ingredient_counts = df['Ingredient'].value_counts()
    
    if ingredient_counts.empty:
        print("No ingredients found in the CSV file.")
        return None
    
    # Select the top_n ingredients
    top_ingredients = ingredient_counts.nlargest(top_n).index.tolist()
    
    if len(top_ingredients) < 2:
        print("Not enough unique ingredients to form pairs.")
        return None
    
    # ----------------------- #
    #   Process Co-occurrence #
    # ----------------------- #
    
    # Group ingredients by Recipe ID and filter to include only top ingredients
    recipes = df[df['Ingredient'].isin(top_ingredients)].groupby('Recipe ID')['Ingredient'].apply(set)
    
    # Initialize a counter for co-occurrences
    co_occurrence = Counter()
    
    # Iterate over each recipe and count ingredient pairs
    for ingredients in recipes:
        if len(ingredients) < 2:
            continue  # Need at least two ingredients to form a pair
        # Generate all unique pairs of top ingredients in the recipe
        pairs = combinations(sorted(ingredients), 2)
        co_occurrence.update(pairs)
    
    if not co_occurrence:
        print("No co-occurring ingredient pairs found among the top ingredients.")
        return None
    
    # Get all pairs sorted by their counts
    top_pairs = co_occurrence.most_common()
    
    # ----------------------- #
    #      Build Graph        #
    # ----------------------- #
    
    # Create a NetworkX graph
    G = nx.Graph()
    
    # Add nodes for top ingredients
    G.add_nodes_from(top_ingredients)
    
    # Add edges with weights based on co-occurrence counts
    for pair, count in co_occurrence.items():
        ingredient1, ingredient2 = pair
        G.add_edge(ingredient1, ingredient2, weight=count)
    
    # ----------------------- #
    #     Visualize Graph     #
    # ----------------------- #
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white')
    
    pos = nx.spring_layout(G, k=0.5, seed=42)
    
    edges = G.edges(data=True)
    edge_weights = [data['weight'] for _, _, data in edges]
    
    if not edge_weights:
        print("No edges to display in the graph.")
        return None
    
    # Normalize edge weights for better visualization
    max_weight = max(edge_weights)
    min_weight = min(edge_weights)
    if max_weight != min_weight:
        normalized_weights = [(weight - min_weight) / (max_weight - min_weight) for weight in edge_weights]
    else:
        normalized_weights = [1 for _ in edge_weights]
    
    # Define edge widths based on normalized weights
    edge_widths = [1 + weight * 4 for weight in normalized_weights]  # Base width + scaled weight
    
    # Define edge colors (optional customization)
    edge_colors = [weight for weight in normalized_weights]
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        ax=ax,
        width=edge_widths,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Blues,
        alpha=0.7
    )
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        ax=ax,
        node_size=700,
        node_color='skyblue',
        edgecolors='black',
        linewidths=1.5
    )
    
    # Draw labels
    nx.draw_networkx_labels(
        G, pos,
        ax=ax,
        font_size=12,
        font_family='sans-serif',
        font_weight='bold'
    )
    
    # Add a colorbar for edge weights
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=min_weight, vmax=max_weight))
    sm.set_array(edge_weights)  # Associate with edge weights
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.05)
    cbar.set_label('Co-occurrence Count', fontsize=14)
    
    # Set plot title
    ax.set_title(f'Top {top_n} Ingredients and Their Co-occurring Pairs', fontsize=20, fontweight='bold')
    
    # Remove axes
    ax.axis('off')
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode the bytes to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return image_base64

def plot_top_n_cooccurring_ingredients_heatmap(csv_file, n=20, recipe_id_col='Recipe ID', ingredient_col='Ingredient', title=None):
    """
    Reads a CSV file containing recipes and their ingredients, identifies the top n ingredients by frequency,
    calculates their co-occurrence, and visualizes them using a heatmap.

    Parameters:
    - csv_file (str): Path to the CSV file.
    - n (int): Number of top ingredients to visualize.
    - recipe_id_col (str): Column name for Recipe IDs in the CSV. Default is 'Recipe ID'.
    - ingredient_col (str): Column name for Ingredients in the CSV. Default is 'Ingredient'.
    - title (str): Title for the heatmap. If None, a default title is used.

    Returns:
    - str: Base64-encoded PNG image of the heatmap.
    """

    # Step 1: Load the CSV file
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
        return None
    except pd.errors.ParserError:
        print(f"Error: Could not parse the file '{csv_file}'. Please check the delimiter and file format.")
        return None

    # Step 2: Validate required columns
    if recipe_id_col not in df.columns or ingredient_col not in df.columns:
        print(f"Error: The CSV file must contain the columns '{recipe_id_col}' and '{ingredient_col}'.")
        return None

    # Step 3: Identify the top n ingredients by frequency
    ingredient_counts = Counter(df[ingredient_col].dropna())
    top_n_ingredients = [ingredient for ingredient, count in ingredient_counts.most_common(n)]

    if not top_n_ingredients:
        print("No ingredients found in the data.")
        return None

    # Step 4: Filter the recipes to include only top n ingredients
    df_top = df[df[ingredient_col].isin(top_n_ingredients)]

    # Step 5: Create a dictionary mapping each Recipe ID to its list of top ingredients
    recipe_dict = df_top.groupby(recipe_id_col)[ingredient_col].apply(set).to_dict()

    # Step 6: Initialize a co-occurrence matrix
    co_matrix = pd.DataFrame(0, index=top_n_ingredients, columns=top_n_ingredients)

    # Step 7: Count co-occurrences
    for ingredients in recipe_dict.values():
        if len(ingredients) < 2:
            continue  # Need at least two ingredients to form a pair
        pairs = combinations(sorted(ingredients), 2)
        for ing1, ing2 in pairs:
            co_matrix.loc[ing1, ing2] += 1
            co_matrix.loc[ing2, ing1] += 1  # Ensure symmetry

    # Step 8: Set diagonal to zero (no self-co-occurrence)
    for ingredient in top_n_ingredients:
        co_matrix.loc[ingredient, ingredient] = 0

    # Step 9: Plot the heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(co_matrix, annot=True, fmt='d', cmap='YlGnBu', linewidths=.5)

    # Set title
    if title:
        plt.title(title)
    else:
        plt.title(f'Co-occurrence Heatmap of Top {n} Ingredients')

    plt.xlabel('Ingredient')
    plt.ylabel('Ingredient')
    plt.tight_layout()

    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    # Encode the bytes to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return image_base64

def generate_ingredient_wordcloud(
    file_path,
    image_width=800,
    image_height=400,
    background_color='white',
    colormap='viridis',
    title='Ingredient Word Cloud',
    stopwords_list=None,
    remove_collocations=True,
    save_path=None
):
    """
    Generates and returns a word cloud from a CSV file containing recipe ingredients.

    Parameters:
    - file_path (str): Path to the CSV file.
    - image_width (int): Width of the word cloud image in pixels (default is 800).
    - image_height (int): Height of the word cloud image in pixels (default is 400).
    - background_color (str): Background color for the word cloud (default is 'white').
    - colormap (str): Colormap for the word cloud (default is 'viridis').
    - title (str): Title of the word cloud plot (default is 'Ingredient Word Cloud').
    - stopwords_list (list or set, optional): Additional words to exclude from the word cloud.
    - remove_collocations (bool): If False, allows repeated phrases (default is True).
    - save_path (str, optional): If provided, saves the word cloud image to the specified path.

    Returns:
    - str: Base64-encoded PNG image of the word cloud.
    """
    
    # Read the CSV file
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return None
    
    # Check if 'Ingredient' column exists
    if 'Ingredient' not in df.columns:
        print("The CSV file must contain an 'Ingredient' column.")
        return None
    
    # Combine all ingredients into a single string
    text = ' '.join(df['Ingredient'].astype(str))
    
    # Initialize stopwords
    stopwords = set(STOPWORDS)
    if stopwords_list:
        stopwords.update(stopwords_list)
    
    # Create a word cloud object
    wordcloud = WordCloud(
        width=image_width,
        height=image_height,
        background_color=background_color,
        colormap=colormap,
        stopwords=stopwords,
        collocations=not remove_collocations
    ).generate(text)
    
    # Set up the matplotlib figure
    plt.figure(figsize=(image_width/100, image_height/100))
    
    # Display the word cloud
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # Remove axes
    plt.title(title, fontsize=20)
    
    # Save the word cloud image if save_path is provided
    if save_path:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Word cloud image saved to {save_path}")
    
    # Save the plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    
    # Encode the bytes to base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    return image_base64
