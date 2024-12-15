
from flask import Flask, render_template, request
from apriori import find_similar_ingredients as apriori_func
from apriori import cosine_similarity_top as cosine_similarity_top_func
from apriori import get_knn_similar_ingredients as knn_func
from cosinesimilarity import autoencoder_pairings as autoencoders_func
from apriori import find_similar_ingredients_fp_growth as fp_growth_func
from apriori import find_similar_ingredients_svd as svd_func
from apriori import recommend_similar_ingredients_dbscan as dbscan_func
from visualization import (
    plot_top_cooccurring_pairs,
    plot_top_n_cooccurring_ingredients_heatmap,
    generate_ingredient_wordcloud,
    plot_tsne_for_ingredients
)
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

app = Flask(__name__)

ALGORITHMS = {
    "apriori": apriori_func,
    "cosine": cosine_similarity_top_func,
    "autoencoder": autoencoders_func,
    "fp_growth": fp_growth_func,
    "knn": knn_func,
    "svd": svd_func,
    "dbscan": dbscan_func
}

@app.route('/', methods=['GET', 'POST'])
def index():
    substitutes = []
    selected_algorithm = "apriori"  # Default algorithm

    if request.method == 'POST':
        print("request.form: ", request.form)
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*torch.load.*")

        ingredient = request.form['ingredient']
        selected_algorithm = request.form['algorithm']  
        algorithm_func = ALGORITHMS.get(selected_algorithm) 

        if algorithm_func:
            substitutes = algorithm_func(ingredient)[:5]  
            print("algorithm_func: ", algorithm_func) 
            print("substitutes are: ", substitutes)
            if(len(substitutes) == 0):
                substitutes = ["No pairings found"]
        
    return render_template('index.html', substitutes=substitutes, selected_algorithm=selected_algorithm)


@app.route('/visualization', methods=['GET', 'POST'])
def visualization():
    error_message = None
    plot_image_graph = None
    plot_image_heatmap = None
    plot_image_wordcloud = None
    plot_image_tsne = None  # Renamed to avoid shadowing
    num_ingredients = 10  # Default value

    if request.method == 'POST':
        try:
            num_ingredients = int(request.form.get('num_ingredients', 10))
            if num_ingredients < 1 or num_ingredients > 100:
                raise ValueError("Number of ingredients must be between 1 and 100.")
        except ValueError as ve:
            num_ingredients = 10  # Reset to default
            error_message = f"Invalid input for the number of ingredients. Defaulting to 10. ({ve})"

    # Path to the CSV file
    csv_file_path = 'non_duplicate_ingredients.csv'  # Ensure this path is correct

    # Generate the NetworkX graph as a base64 string
    plot_image_graph = plot_top_cooccurring_pairs(file_path=csv_file_path, top_n=num_ingredients)

    # Generate the seaborn heatmap as a base64 string
    plot_image_heatmap = plot_top_n_cooccurring_ingredients_heatmap(
        csv_file=csv_file_path,
        n=num_ingredients
    )

    # Generate the Word Cloud as a base64 string
    plot_image_wordcloud = generate_ingredient_wordcloud(
        file_path=csv_file_path,
        image_width=800,
        image_height=400,
        background_color='white',
        colormap='viridis',
        title='Ingredient Word Cloud',
        remove_collocations=True,  # Set to False if you want to include collocations
        save_path=None  # Set to a path if you want to save the image on the server
    )

    # Corrected function call with appropriate parameter names and variable naming
    plot_image_tsne = plot_tsne_for_ingredients(
        num_ingredients=num_ingredients
    )

    # Check for plot generation failures and set error messages accordingly
    if not plot_image_graph and not plot_image_heatmap and not plot_image_wordcloud and not plot_image_tsne:
        if not error_message:
            error_message = "Unable to generate any plots. Please check the CSV file and try again."
    else:
        if not plot_image_graph and not plot_image_heatmap and not plot_image_wordcloud:
            if not error_message:
                error_message = "Unable to generate the NetworkX graph, heatmap, and word cloud. Please check the CSV file and try again."
        elif not plot_image_graph and not plot_image_heatmap and not plot_image_tsne:
            if not error_message:
                error_message = "Unable to generate the NetworkX graph, heatmap, and t-SNE plot. Please check the CSV file and try again."
        elif not plot_image_graph and not plot_image_wordcloud and not plot_image_tsne:
            if not error_message:
                error_message = "Unable to generate the NetworkX graph, word cloud, and t-SNE plot. Please check the CSV file and try again."
        elif not plot_image_heatmap and not plot_image_wordcloud and not plot_image_tsne:
            if not error_message:
                error_message = "Unable to generate the heatmap, word cloud, and t-SNE plot. Please check the CSV file and try again."
        else:
            if not plot_image_graph:
                if not error_message:
                    error_message = "Unable to generate the NetworkX graph. Please check the CSV file and try again."
            if not plot_image_heatmap:
                if not error_message:
                    error_message = "Unable to generate the heatmap. Please check the CSV file and try again."
            if not plot_image_wordcloud:
                if not error_message:
                    error_message = "Unable to generate the word cloud. Please check the CSV file and try again."
            if not plot_image_tsne:
                if not error_message:
                    error_message = "Unable to generate the t-SNE plot. Please check the CSV file and try again."

    return render_template(
        'visualization.html',
        plot_image_graph=plot_image_graph,
        plot_image_heatmap=plot_image_heatmap,
        plot_image_wordcloud=plot_image_wordcloud,
        plot_image_tsne=plot_image_tsne,  # Updated variable name
        num_ingredients=num_ingredients,
        error=error_message
    )


@app.route('/information')
def information():
    return render_template('information.html')


if __name__ == '__main__':
    app.run(debug=True)
