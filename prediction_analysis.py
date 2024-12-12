import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Function to load predictions from a CSV file
def load_predictions(csv_path):
    return pd.read_csv(csv_path)

# Function to compute statistics on actual and predicted values
def compute_statistics(predictions):
    mean_actual = predictions['actual'].mean()
    std_actual = predictions['actual'].std()
    mean_predicted = predictions['predicted'].mean()
    std_predicted = predictions['predicted'].std()
    correlation = predictions['actual'].corr(predictions['predicted'])

    return mean_actual, std_actual, mean_predicted, std_predicted, correlation

# Function to save the computed statistics to a text file
def save_statistics(stats, stats_file):
    mean_actual, std_actual, mean_predicted, std_predicted, correlation = stats
    with open(stats_file, 'w') as f:
        f.write(f'Mean (Actual): {mean_actual}\n')
        f.write(f'Std (Actual): {std_actual}\n')
        f.write(f'Mean (Predicted): {mean_predicted}\n')
        f.write(f'Std (Predicted): {std_predicted}\n')
        f.write(f'Correlation: {correlation}\n')

# Function to plot and save the scatter plot and linear fit to a PDF
def plot_actual_vs_predicted(predictions, pdf_file, stats_text):
    with PdfPages(pdf_file) as pdf:
        # Plot the scatter plot with a linear fit using seaborn's regplot
        plt.figure(figsize=(8, 6))
        sns.regplot(x='actual', y='predicted', data=predictions, scatter_kws={'color': 'blue'}, line_kws={'color': 'green', 'linewidth': 2}, label='Linear Fit')

        # Plot the perfect prediction line
        plt.plot([predictions['actual'].min(), predictions['actual'].max()],
                 [predictions['actual'].min(), predictions['actual'].max()],
                 color='red', linestyle='--', label='Perfect Prediction')

        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted with Linear Fit')
        plt.legend()
        pdf.savefig()
        plt.close()

        # Add the statistics text to a new page in the PDF
        plt.figure()
        plt.axis('off')
        plt.text(0.01, 0.99, stats_text, verticalalignment='top', horizontalalignment='left', wrap=True)
        pdf.savefig()
        plt.close()

# Function to read the statistics from a file
def read_statistics_from_file(stats_file):
    with open(stats_file, 'r') as f:
        return f.read()

# Callable function that combines the operations
def generate_predictions_report(csv_path):
    # Define paths
    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.splitext(os.path.basename(csv_path))[0]
    stats_file = os.path.join(csv_dir, f'computed_statistics_{csv_name}.txt')
    pdf_file = os.path.join(csv_dir, f'actual_vs_predicted_{csv_name}.pdf')

    # Load predictions from CSV
    predictions = load_predictions(csv_path)

    # Compute statistics
    stats = compute_statistics(predictions)

    # Save statistics to a text file
    save_statistics(stats, stats_file)

    # Read the statistics text
    stats_text = read_statistics_from_file(stats_file)

    # Plot the actual vs predicted values and save to PDF
    plot_actual_vs_predicted(predictions, pdf_file, stats_text)
