import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from scipy.stats import norm

def load_data(file_path):
    """Load data from an Excel file."""
    purchase_data = pd.read_excel(file_path, sheet_name="Purchase Data")
    stock_price_data = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
    thyroid_data = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
    return purchase_data, stock_price_data, thyroid_data

def segregate_matrices(purchase_data):
    """Segregate data into matrices A and C."""
    A = purchase_data.iloc[:, :-1].values  # All columns except the last
    C = purchase_data.iloc[:, -1].values   # Last column
    return A, C

def calculate_matrix_properties(A):
    """Calculate dimensionality, number of vectors, and rank of matrix A."""
    dimensions = A.shape
    num_vectors = dimensions[1]
    rank = np.linalg.matrix_rank(A)
    return dimensions, num_vectors, rank

def calculate_pseudo_inverse(A):
    """Calculate pseudo-inverse of matrix A."""
    return np.linalg.pinv(A)

def compute_product_costs(A, C):
    """Calculate the cost of each product using pseudo-inverse."""
    A_pseudo_inv = calculate_pseudo_inverse(A)
    return A_pseudo_inv @ C

def classify_customers(purchase_data):
    """Classify customers as RICH or POOR based on their payments."""
    purchase_data['Class'] = purchase_data['Payment'].apply(lambda x: 'RICH' if x > 200 else 'POOR')
    return purchase_data

def analyze_stock_data(stock_price_data):
    """Analyze stock price data."""
    price_data = stock_price_data['Price'].dropna()
    mean_price = price_data.mean()
    variance_price = price_data.var()
   
    wednesdays = stock_price_data[stock_price_data['Day'] == 'Wednesday']['Price'].dropna()
    apr_prices = stock_price_data[stock_price_data['Month'] == 'Apr']['Price'].dropna()
   
    mean_wednesday = wednesdays.mean()
    mean_apr = apr_prices.mean()
   
    chg_percent = stock_price_data['Chg%'].dropna()
    prob_loss = (chg_percent < 0).mean()
    prob_profit_wednesday = (chg_percent[stock_price_data['Day'] == 'Wednesday'] > 0).mean()
    prob_profit_given_wednesday = prob_profit_wednesday / (chg_percent[stock_price_data['Day'] == 'Wednesday'].count() / len(chg_percent))
   
    return mean_price, variance_price, mean_wednesday, mean_apr, prob_loss, prob_profit_wednesday, prob_profit_given_wednesday

def plot_scatter_chg_day(stock_price_data):
    """Create a scatter plot of Chg% against the day of the week."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=stock_price_data, x='Day', y='Chg%')
    plt.title('Change Percentage vs Day of the Week')
    plt.show()

def explore_thyroid_data(thyroid_data):
    """Explore and preprocess the thyroid data."""
    # Handle categorical attributes
    le = LabelEncoder()
    ohe = OneHotEncoder()
   
    categorical_columns = thyroid_data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        thyroid_data[col] = le.fit_transform(thyroid_data[col])
   
    # Handle missing values
    for col in thyroid_data.columns:
        if thyroid_data[col].isnull().any():
            if thyroid_data[col].dtype == 'object':
                thyroid_data[col].fillna(thyroid_data[col].mode()[0], inplace=True)
            else:
                thyroid_data[col].fillna(thyroid_data[col].median(), inplace=True)
   
    return thyroid_data

def calculate_similarity_measures(vector1, vector2):
    """Calculate Jaccard and Simple Matching Coefficients."""
    f11 = np.sum((vector1 == 1) & (vector2 == 1))
    f00 = np.sum((vector1 == 0) & (vector2 == 0))
    f01 = np.sum((vector1 == 0) & (vector2 == 1))
    f10 = np.sum((vector1 == 1) & (vector2 == 0))
   
    jc = f11 / (f01 + f10 + f11)
    smc = (f11 + f00) / (f00 + f01 + f10 + f11)
   
    return jc, smc

def calculate_cosine_similarity(vector1, vector2):
    """Calculate cosine similarity between two vectors."""
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    return dot_product / (norm1 * norm2)

def create_similarity_heatmap(vectors):
    """Create a heatmap of similarity measures."""
    num_vectors = len(vectors)
    jc_matrix = np.zeros((num_vectors, num_vectors))
    smc_matrix = np.zeros((num_vectors, num_vectors))
    cosine_matrix = np.zeros((num_vectors, num_vectors))
   
    for i in range(num_vectors):
        for j in range(num_vectors):
            jc_matrix[i, j], smc_matrix[i, j] = calculate_similarity_measures(vectors[i], vectors[j])
            cosine_matrix[i, j] = calculate_cosine_similarity(vectors[i], vectors[j])
   
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    sns.heatmap(jc_matrix, annot=True, cmap='Blues')
    plt.title('Jaccard Coefficient')
   
    plt.subplot(1, 3, 2)
    sns.heatmap(smc_matrix, annot=True, cmap='Greens')
    plt.title('Simple Matching Coefficient')
   
    plt.subplot(1, 3, 3)
    sns.heatmap(cosine_matrix, annot=True, cmap='Reds')
    plt.title('Cosine Similarity')
   
    plt.show()

# Main program section
def main(file_path):
    # Load the data
    purchase_data, stock_price_data, thyroid_data = load_data(file_path)
   
    # Process Purchase Data
    A, C = segregate_matrices(purchase_data)
    dimensions, num_vectors, rank = calculate_matrix_properties(A)
    product_costs = compute_product_costs(A, C)
   
    # Print results
    print(f"Dimensions of matrix A: {dimensions}")
    print(f"Number of vectors: {num_vectors}")
    print(f"Rank of matrix A: {rank}")
    print(f"Product costs: {product_costs}")
   
    # Classify Customers
    classified_data = classify_customers(purchase_data)
    print("Customer classification:\n", classified_data[['Customer', 'Class']])
   
    # Analyze Stock Price Data
    mean_price, variance_price, mean_wednesday, mean_apr, prob_loss, prob_profit_wednesday, prob_profit_given_wednesday = analyze_stock_data(stock_price_data)
    print(f"Mean Price: {mean_price}")
    print(f"Variance Price: {variance_price}")
    print(f"Mean Price on Wednesdays: {mean_wednesday}")
    print(f"Mean Price in April: {mean_apr}")
    print(f"Probability of making a loss: {prob_loss}")
    print(f"Probability of making a profit on Wednesday: {prob_profit_wednesday}")
    print(f"Conditional probability of making a profit given Wednesday: {prob_profit_given_wednesday}")
   
    # Plot Stock Data
    plot_scatter_chg_day(stock_price_data)
   
    # Explore Thyroid Data
    processed_thyroid_data = explore_thy
