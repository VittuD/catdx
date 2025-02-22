import os
import time
import pandas as pd
import seaborn as sns
import torch
from utils import compute_r2, compute_mae
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Set the backend to Agg
matplotlib.use('Agg')
# Turn off interactive mode
plt.ioff()

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
    predicted_tensor = torch.tensor(predictions['predicted'].values).cuda()
    actual_tensor = torch.tensor(predictions['actual'].values).cuda()
    r2 = compute_r2(predicted_tensor, actual_tensor)
    mae = compute_mae(predicted_tensor, actual_tensor)

    return mean_actual, std_actual, mean_predicted, std_predicted, correlation, r2, mae

# Function to save the computed statistics to a text file
def save_statistics(stats, stats_file):
    mean_actual, std_actual, mean_predicted, std_predicted, correlation, r2, mae = stats
    with open(stats_file, 'w') as f:
        f.write(f'Mean (Actual): {mean_actual}\n')
        f.write(f'Std (Actual): {std_actual}\n')
        f.write(f'Mean (Predicted): {mean_predicted}\n')
        f.write(f'Std (Predicted): {std_predicted}\n')
        f.write(f'Correlation: {correlation}\n')
        f.write(f'R2 Score: {r2}\n')
        f.write(f'Mean Absolute Error: {mae}\n')

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
        # Split of interest is last word in the title before .pdf
        split = pdf_file.split('_')[-1].split('.')[0]
        plt.title(f'Actual vs Predicted Linear Fit ({split})')
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

    return pdf_file

# Test main with timing
if __name__ == "__main__":
    start_time = time.time()
    csv_path = 'apical4_none_112_32_22/checkpoint-154/predictions_train.csv'
    generate_predictions_report(csv_path)
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")