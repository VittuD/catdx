import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

val_path = "apical4_none_420p_amp_wd_lr/checkpoint-1014/predictions_validation.csv"
test_path = "apical4_none_420p_amp_wd_lr/checkpoint-1014/predictions_test.csv"
parent_dir = "apical4_none_420p_amp_wd_lr/checkpoint-1014"

# Load validation and test data
val_data = pd.read_csv(val_path)
test_data = pd.read_csv(test_path)

# Extract actual and predicted values
y_val_actual = val_data["actual"].values
y_val_pred = val_data["predicted"].values

y_test_actual = test_data["actual"].values
y_test_pred = test_data["predicted"].values

############################################
# Method 3: Direct Linear Mapping (adjusted = (predicted - b) * a)
############################################
A = np.vstack([y_val_pred, np.ones(len(y_val_pred))]).T
a, b = np.linalg.lstsq(A, y_val_actual, rcond=None)[0]

y_test_adjusted_linear = (y_test_pred - b) / a

mae_original = mean_absolute_error(y_test_actual, y_test_pred)
r2_original = r2_score(y_test_actual, y_test_pred)
corr_original = pearsonr(y_test_actual, y_test_pred)

mae_linear = mean_absolute_error(y_test_actual, y_test_adjusted_linear)
r2_linear = r2_score(y_test_actual, y_test_adjusted_linear)
corr_linear = pearsonr(y_test_actual, y_test_adjusted_linear)

print("=== Direct Linear Mapping ===")
print(f"a = {a:.3f}, b = {b:.3f}")
print(f"Original: MAE={mae_original:.3f}, R2={r2_original:.3f}, Corr={corr_original[0]:.3f}")
print(f"Linear Adjusted: MAE={mae_linear:.3f}, R2={r2_linear:.3f}, Corr={corr_linear[0]:.3f}")

############################################
# Method 4: Matching Means and Standard Deviations
############################################
mean_pred = np.mean(y_val_pred)
std_pred = np.std(y_val_pred)
mean_actual = np.mean(y_val_actual)
std_actual = np.std(y_val_actual)

y_test_adjusted_meanstd = ((y_test_pred - mean_pred) / std_pred) * std_actual + mean_actual

mae_meanstd = mean_absolute_error(y_test_actual, y_test_adjusted_meanstd)
r2_meanstd = r2_score(y_test_actual, y_test_adjusted_meanstd)
corr_meanstd = pearsonr(y_test_actual, y_test_adjusted_meanstd)

print("=== Mean/Std Matching ===")
print(f"mean_pred={mean_pred:.3f}, std_pred={std_pred:.3f}, mean_actual={mean_actual:.3f}, std_actual={std_actual:.3f}")
print(f"Original: MAE={mae_original:.3f}, R2={r2_original:.3f}, Corr={corr_original[0]:.3f}")
print(f"Mean/Std Adjusted: MAE={mae_meanstd:.3f}, R2={r2_meanstd:.3f}, Corr={corr_meanstd[0]:.3f}")

############################################
# (Optional) Save adjusted predictions to file
############################################
test_data["adjusted_linear"] = y_test_adjusted_linear
test_data["adjusted_meanstd"] = y_test_adjusted_meanstd
adjusted_predictions_path = os.path.join(parent_dir, "adjusted_test_predictions_comparison.csv")
test_data.to_csv(adjusted_predictions_path, index=False)

############################################
# (Optional) Plot actual vs adjusted for each method
############################################
def plot_actual_vs_adjusted(y_actual, y_adjusted, title, pdf_file, stats_text):
    with PdfPages(pdf_file) as pdf:
        plt.figure(figsize=(8, 6))
        plot_df = pd.DataFrame({"actual": y_actual, "adjusted": y_adjusted})
        sns.regplot(x='actual', y='adjusted', data=plot_df,
                    scatter_kws={'color': 'blue'},
                    line_kws={'color': 'green', 'linewidth': 2},
                    label='Linear Fit')

        # Perfect prediction line
        plt.plot([y_actual.min(), y_actual.max()],
                 [y_actual.min(), y_actual.max()],
                 color='red', linestyle='--', label='Perfect Fit')

        plt.xlabel('Actual')
        plt.ylabel('Adjusted')
        plt.title(title)
        plt.legend()
        pdf.savefig()
        plt.close()

        # Add stats text
        plt.figure()
        plt.axis('off')
        plt.text(0.01, 0.99, stats_text, verticalalignment='top', horizontalalignment='left', wrap=True)
        pdf.savefig()
        plt.close()

# Plot linear adjusted
stats_text_linear = (f"Original: MAE={mae_original:.3f}, R2={r2_original:.3f}, Corr={corr_original[0]:.3f}\n"
                     f"Linear Adjusted: MAE={mae_linear:.3f}, R2={r2_linear:.3f}, Corr={corr_linear[0]:.3f}")
plot_actual_vs_adjusted(y_test_actual, y_test_adjusted_linear, "Actual vs Linear Adjusted", 
                        os.path.join(parent_dir, "actual_vs_linear_adjusted_test.pdf"), stats_text_linear)

# Plot mean/std adjusted
stats_text_meanstd = (f"Original: MAE={mae_original:.3f}, R2={r2_original:.3f}, Corr={corr_original[0]:.3f}\n"
                      f"Mean/Std Adjusted: MAE={mae_meanstd:.3f}, R2={r2_meanstd:.3f}, Corr={corr_meanstd[0]:.3f}")
plot_actual_vs_adjusted(y_test_actual, y_test_adjusted_meanstd, "Actual vs Mean/Std Adjusted", 
                        os.path.join(parent_dir, "actual_vs_meanstd_adjusted_test.pdf"), stats_text_meanstd)
