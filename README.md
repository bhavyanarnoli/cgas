# Ingredient Pairing Tool

## Overview

The **Ingredient Pairing Tool** is an intuitive web-based platform designed to assist users in finding the best ingredient pairings based on various advanced algorithms. Leveraging concepts like molecular compatibility and data-driven recommendations, this tool helps enhance culinary experiences by suggesting substitutes or complementary ingredients.

The application is highly customizable, allowing users to select their desired algorithm for pairing generation, ensuring flexibility and versatility.

## Features


- **Interactive Interface**: A responsive, visually appealing interface for user input and result display.
- **Algorithm Selection**: Choose from multiple algorithms such as Apriori, KNN, SVD, FP-Growth, Cosine Similarity, Autoencoder, and DBSCAN for pairing generation.
- **Ingredient Substitution Suggestions**: Provides a list of compatible ingredient pairings.
- **Visualization Tool**: Generate visual representations of ingredient relationships, including graphs, heatmaps, and word clouds.
- **Image Carousel**: A visual component displaying various food-related images to enhance user engagement.
- **Mobile-Friendly**: Fully responsive design, ensuring a seamless experience across devices.
- **New Multi-Ingredient Pairing Feature**: Input multiple ingredients at once to generate pairings that consider combined compatibility.

## Data Generate of non_duplicate_ingredients.csv and recipes.csv 

# Recipe Scraping and Ingredient Processing
This project scrapes recipes from the AllRecipes website, cleans up the extracted ingredients, and processes them to generate meaningful statistics such as frequency counts and recipe size distributions.


## Requirements

Install the required libraries:

```bash
pip install requests beautifulsoup4 pandas spacy matplotlib
python -m spacy download en_core_web_sm
```

## Functions

- **`get_proxies(file_path)`**: Loads proxies from a file.
- **`rotate_proxy(proxy_counter, proxies)`**: Rotates proxies during scraping.
- **`parse_category_urls(s, proxy_counter, proxies, max_categories, use_proxy=False)`**: Scrapes category URLs.
- **`get_category(s, category_url, proxy_counter, proxies, use_proxy=False)`**: Fetches recipe category data.
- **`get_category_recipes_urls(category)`**: Extracts recipe URLs from a category.
- **`get_recipe(s, recipe_url, proxy_counter, proxies, use_proxy=False)`**: Scrapes recipe data and saves to CSV.
- **`scrape_recipes(s, recipe_categories, max_categories, max_recipes)`**: Main scraping function for collecting recipes.

## Ingredient Processing

- **`clean_ingredient(ingredient)`**: Cleans ingredient text.
- **`extract_ingredients_ner(ingredient_text)`**: Extracts ingredients using spaCy NER.

## Data Analysis

- **Recipe Size Distribution**: Histogram of ingredients per recipe (saved as `recipe_size_distribution_probability.png`).
- **Cumulative Recipe Size Distribution**: Cumulative distribution of recipe sizes (saved as `cumulative_recipe_size_distribution.png`).

## Files

- **`recipes.csv`**: Scraped recipe data.
- **`non_duplicate_ingredients.csv`**: Cleaned, non-duplicate ingredients.
- **`recipes_ingredients.txt`**: 100 random recipes with ingredients.
- **`unique_ingredient_frequencies.csv`**: Ingredient frequency counts.
- **`recipe_size_distribution_probability.png`**: Recipe size distribution histogram.
- **`cumulative_recipe_size_distribution.png`**: Cumulative size distribution plot.

## How to Run

1. Run the `scrape_recipes` function to scrape recipes.
2. Use `clean_ingredient` and `extract_ingredients_ner` for ingredient processing.
3. Analyze data with the provided visualization functions.

## Troubleshooting

- Install missing dependencies.
- Check proxy settings if facing issues with scraping.

--- 

You can copy this directly into your README. Let me know if you'd like any changes!


## File Structure


### 1. `index.html`

This file serves as the front-end interface of the tool, providing the following functionalities:

#### Key Features

- **Navbar**: Contains navigation links (Home, Visualization, Information) to ensure easy navigation.
- **Introductory Section**: Describes the purpose and usage of the tool.
- **Image Carousel**: Displays a carousel of ingredient and food-related images.
- **Form**: Allows users to input an ingredient and choose an algorithm for pairing generation.
- **Suggested Pairings Section**: Dynamically displays pairing suggestions returned by the backend.

#### Highlights

- **CSS Styling**: Includes styles for a professional and user-friendly interface.
- **Responsive Design**: Adapts layout for various screen sizes using media queries.
- **Dynamic Content**: Uses Jinja2 placeholders for displaying results (`{% if substitutes %}`, `{% for substitute in substitutes %}`).

### 2. Backend Application (e.g., Flask App)


The backend application processes user inputs and generates ingredient pairings. Here's how it works:

#### Input Handling:

- The form in `index.html` sends a POST request to the backend containing:
  - `ingredient`: The ingredient entered by the user.
  - `algorithm`: The selected algorithm for pairing generation.

#### Processing Logic:

- The backend receives the input and processes it using the selected algorithm.
- Algorithms like Apriori, KNN, SVD, and others analyze predefined ingredient data and determine the best matches.

#### Generating Results:

- The processed pairings are returned as a list of substitutes, formatted for display on the front-end.

#### Rendering the Response:

- The backend renders the `index.html` template, passing the substitutes to be displayed dynamically in the "Suggested Pairings" section.

### 3. `visualization.html`

This file provides an interactive platform to visualize ingredient pairings. It leverages Python libraries like NetworkX, Seaborn, and WordCloud for graph-based, heatmap-based, and word cloud representations.

#### Features

- **Navbar**: Contains links to Home, Visualization, and Information sections.
- **Form**: Users can input the number of ingredients to visualize relationships.
- **Graph Visualization**: Generates NetworkX graphs to showcase ingredient pairings.
- **Heatmap Visualization**: Displays a Seaborn heatmap for better insights into pairing relationships.
- **Word Cloud**: Generates a word cloud highlighting frequently paired ingredients.
- **T-SNE**: The plot_tsne_for_ingredients function visualizes the similarity between ingredients by applying t-SNE after vectorizing them using TF-IDF. It generates a 2D scatter plot where similar ingredients cluster together.

## Substitution Use Case:
Since the function groups similar ingredients together, the plot can help identify substitutions. For instance, ingredients that appear close together in the t-SNE visualization can be considered similar and potentially interchangeable in recipes. This clustering could be useful for recommending ingredient substitutions based on similarity, dietary preferences, or availability.

### 3. `information.html`

It has three plots showing the recipe size of the data, frequence rank curve and Cumulative Distribution of Recipe Sizes





#### Working

1. **Input Handling**:
   - Users enter the number of ingredients they want to analyze in the provided form.
2. **Data Processing**:
   - The backend generates visualizations (graph, heatmap, word cloud) using the input data.
3. **Rendering Results**:
   - The generated visualizations are returned to the `visualization.html` page and displayed dynamically.

#### HTML Highlights

- **CSS Styling**: Incorporates a visually appealing design using custom styles.
- **Responsive Design**: Ensures compatibility with both desktop and mobile screens.
- **Dynamic Visualizations**: Uses Jinja2 to render images generated by the backend (`{% if plot_image_graph %}`, etc.).

---

## How to Run the Project

### #HowToRun

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/yourusername/ingredient-pairing-tool.git
   cd ingredient-pairing-tool
   ```

2. **Install Dependencies**: Ensure you have Python installed. Then, install the required libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Backend Application**: Start the Flask server:

   ```bash
   python app.py
   ```

   The server will run on [http://127.0.0.1:5000](http://127.0.0.1:5000) by default.

4. **Access the Application**: Open your browser and navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000).

5. **Explore Features**:

   - Use the Ingredient Pairing Tool on the main page to generate ingredient suggestions.
   - Navigate to the Visualization tab to explore graphs, heatmaps, and word clouds.
   - Experiment with the new multi-ingredient pairing feature.

## Screenshots


- **Main Interface (index.html)**
- **Visualization Page (visualization.html)**
- **Multi-Ingredient Pairing Feature**

## Future Enhancements


- **User Accounts**: Add functionality for users to save their pairing preferences.
- **Ingredient Database Expansion**: Include more ingredients and their molecular data.
- **Real-Time Data**: Enable real-time data updates from external APIs.
- **Machine Learning Models**: Integrate advanced ML models for even better pairing recommendations.
- **Multi-Language Support**: Allow users to interact with the tool in multiple languages.

## Contributing

We welcome contributions from the community! If you'd like to contribute:

1. **Fork the repository.**
2. **Create a feature branch**:
   
   ```bash
   git checkout -b feature-name
   ```

3. **Commit your changes**:
   
   ```bash
   git commit -m 'Add feature'
   ```

4. **Push to the branch**:
   
   ```bash
   git push origin feature-name
   ```

5. **Open a pull request**.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For queries or feedback, feel free to reach out:

- **GitHub**: [bhavyanarnoli](https://github.com/bhavyanarnoli)

