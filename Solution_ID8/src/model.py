import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
import os
from datetime import datetime
import torch.optim as optim
import pandas as pd
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import savemat
from matplotlib import gridspec
from matplotlib.patches import ConnectionPatch

plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 12,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linestyle': '--'
    })

def train_model(model, train_loader, num_epochs=20, learning_rate=0.001, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for data in train_loader:
            data = data.view(data.size(0), -1).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(data)
            loss = vae_loss_function(recon_x, data, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.8f}")

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = F.mse_loss(recon_x, x, reduction='sum') / x.size(0)
    KLD = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1), dim=0)
    return 0.5*BCE + 0.5*KLD

def calculate_re(model, data_loader, device='cpu'):
    model.eval()
    hi_scores = []

    with torch.no_grad():
        for data in data_loader:
            data = data.view(data.size(0), -1).to(device)
            reconstructed, _, _ = model(data)
            loss = nn.MSELoss(reduction='none')(reconstructed, data)
            reconstruction_error = torch.mean(loss, dim=1).cpu()
            hi_scores.extend(reconstruction_error.tolist())
    return hi_scores

def apply_ewma(hi_scores, alpha=0.2):
    hi_series = pd.Series(hi_scores)
    smoothed_hi = hi_series.ewm(alpha=alpha, adjust=False).mean()
    return smoothed_hi.tolist()


def calculate_non_overlapping_average(scores: np.ndarray, window_size: int = 60) -> np.ndarray:
    """Calculate non-overlapping window averages of scores.
    Args:
        scores: Input array of scores
        window_size: Size of non-overlapping windows
    Returns:
        Array of averaged scores per window
    """
    scores = np.asarray(scores)
    return np.mean(scores[:len(scores) - len(scores) % window_size].reshape(-1, window_size), axis=1)


def process_health_indices(train_scores: np.ndarray, val_scores: np.ndarray, test_scores: np.ndarray,
                           window_size: int = 60, alpha: float = 0.2):
    """Process health indices with non-overlapping averaging and EWMA smoothing."""
    # Calculate non-overlapping averages
    averages = [
        calculate_non_overlapping_average(scores, window_size)
        for scores in (train_scores, val_scores, test_scores)
    ]

    # Combine and smooth
    all_scores = pd.concat([pd.Series(avg) for avg in averages])
    smoothed = apply_ewma(all_scores, alpha)

    # Split back into train/val/test
    train_len, val_len = len(averages[0]), len(averages[1])
    train_smoothed = smoothed[:train_len]
    val_smoothed = smoothed[train_len:train_len + val_len]
    test_smoothed = smoothed[train_len + val_len:]

    # Calculate threshold
    threshold = max(np.max(train_smoothed), np.max(val_smoothed))

    return train_smoothed, val_smoothed, test_smoothed, threshold

def plot_hi(train_hi, val_hi, icing_hi, threshold, fault_record_index,
            all_record_index, dates_for_legend, all_datetimes, output_dir):
    dates_for_legend = [datetime.strptime(date, '%Y-%m-%d') for date in dates_for_legend]

    all_hi = np.concatenate([train_hi, val_hi, icing_hi])
    segments = [0, len(train_hi), len(train_hi) + len(val_hi), len(train_hi) + len(val_hi)]

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    ax1.plot(train_hi, color='#2ecc71', label="Training", linewidth=1)
    ax1.plot(range(len(train_hi), len(train_hi) + len(val_hi)),
             val_hi, color='#34495e', label="Validation", linewidth=1)
    ax1.plot(range(len(train_hi) + len(val_hi), len(all_hi)),
             icing_hi, color='#3498db', label="Testing", linewidth=1)
    ax1.axhline(y=threshold, color='#e74c3c', linestyle='--',
                alpha=0.7, label='Threshold', linewidth=1)

    # Mark fault occurrence
    if fault_record_index is not None and fault_record_index < len(all_hi):
        fault_datetime = all_datetimes.iloc[fault_record_index]
        plt.scatter(fault_record_index, all_hi[fault_record_index],
                    color='#c0392b', s=50, marker='X',
                    label=f'Fault ({fault_datetime.strftime("%Y-%m-%d %H:%M")})')
        plt.annotate(f'Fault ({fault_datetime.strftime("%Y-%m-%d %H:%M")})',
                     (fault_record_index, all_hi[fault_record_index]),
                     textcoords="offset points", xytext=(10, 10),
                     ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

    # Mark first anomaly
    anomaly_indices = np.where(all_hi > threshold)[0]
    if len(anomaly_indices) > 0:
        first_anomaly_index = anomaly_indices[0]
        first_anomaly_datetime = all_datetimes.iloc[first_anomaly_index]
        plt.scatter(first_anomaly_index, all_hi[first_anomaly_index],
                    color='#e74c3c', s=50, marker='D', facecolors='none',
                    label=f'Alarm ({first_anomaly_datetime.strftime("%Y-%m-%d %H:%M")})')
        plt.annotate(f'Alarm ({first_anomaly_datetime.strftime("%Y-%m-%d %H:%M")})',
                     (first_anomaly_index, all_hi[first_anomaly_index]),
                     textcoords="offset points", xytext=(-10, 25),
                     ha='right', va='top',
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.3))

    ax1.set_xticks([idx for idx in all_record_index if idx < len(all_hi)])
    ax1.set_xticklabels([date.strftime('%Y-%m-%d') for date in dates_for_legend if date], rotation=45, ha='right')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
               ncol=5, frameon=True, edgecolor='gray')
    ax1.set_xlabel("Date")
    ax1.set_ylabel("EWMA")

    # Zoomed plot
    ax2 = plt.subplot(gs[1])
    if len(anomaly_indices) > 0:
        first_anomaly_index = anomaly_indices[0]
        zoom_start = max(0, first_anomaly_index - 100)  # Show 100 points before alarm
        zoom_end = min(len(all_hi), first_anomaly_index + 5)  # Show 50 points after alarm

        zoom_times = all_datetimes.iloc[zoom_start:zoom_end]
        tick_step = max(1, len(zoom_times) // 5)  # Show ~5 ticks
        ax2.set_xticks(range(zoom_start, zoom_end, tick_step))
        ax2.set_xticklabels([dt.strftime('%Y-%m-%d\n%H:%M') for dt in zoom_times[::tick_step]],
                            rotation=45, ha='right')

        # Plot zoomed section
        if zoom_start < len(train_hi):
            ax2.plot(range(zoom_start, min(len(train_hi), zoom_end)),
                     train_hi[zoom_start:min(len(train_hi), zoom_end)],
                     color='#2ecc71', linewidth=1.5)

        val_start = max(zoom_start, len(train_hi))
        val_end = min(zoom_end, len(train_hi) + len(val_hi))
        if val_start < val_end:
            ax2.plot(range(val_start, val_end),
                     val_hi[val_start - len(train_hi):val_end - len(train_hi)],
                     color='#34495e', linewidth=1.5)

        test_start = max(zoom_start, len(train_hi) + len(val_hi))
        if test_start < zoom_end:
            ax2.plot(range(test_start, zoom_end),
                     icing_hi[test_start - (len(train_hi) + len(val_hi)):zoom_end - (len(train_hi) + len(val_hi))],
                     color='#3498db', linewidth=1.5)

        ax2.axhline(y=threshold, color='#e74c3c', linestyle='--', alpha=0.7)
        ax2.scatter(first_anomaly_index, all_hi[first_anomaly_index],
                    color='#e74c3c', s=80, marker='D', facecolors='none')

        # Add detailed time annotation in zoomed plot
        ax2.annotate(f'Alarm\n{first_anomaly_datetime.strftime("%Y-%m-%d %H:%M")}',
                     (first_anomaly_index, all_hi[first_anomaly_index]),
                     textcoords="offset points", xytext=(10, 10),
                     ha='left', va='bottom',
                     bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='gray', alpha=0.8))

        # Set zoom plot labels and limits
        ax2.set_xlim(zoom_start, zoom_end)
        y_margin = (max(all_hi[zoom_start:zoom_end]) - min(all_hi[zoom_start:zoom_end])) * 0.1
        ax2.set_ylim(min(all_hi[zoom_start:zoom_end]) - y_margin,
                     max(all_hi[zoom_start:zoom_end]) + y_margin)

        # Add connecting lines between plots
        con1 = ConnectionPatch(xyA=(zoom_start, ax2.get_ylim()[1]),
                               xyB=(zoom_start, ax1.get_ylim()[0]),
                               coordsA="data", coordsB="data",
                               axesA=ax2, axesB=ax1, color="gray", linestyle="--")
        con2 = ConnectionPatch(xyA=(zoom_end, ax2.get_ylim()[1]),
                               xyB=(zoom_end, ax1.get_ylim()[0]),
                               coordsA="data", coordsB="data",
                               axesA=ax2, axesB=ax1, color="gray", linestyle="--")
        ax2.add_artist(con1)
        ax2.add_artist(con2)

    ax2.set_xlabel("Date")
    ax2.set_ylabel("EWMA")

    plt.tight_layout()
    fig_path = os.path.join(output_dir, "health_index_plot.svg")
    plt.savefig(fig_path, format='svg', dpi=600, bbox_inches='tight')

    matlab_data = {
        'train_hi': train_hi,
        'val_hi': val_hi,
        'icing_hi': icing_hi,
        'threshold': threshold,
        'fault_record_index': fault_record_index,
        'all_record_index': np.array(all_record_index),
        'dates_for_legend': np.array([date.strftime('%Y-%m-%d') for date in dates_for_legend]),
        'all_datetimes': np.array([dt.strftime('%Y-%m-%d %H:%M') for dt in all_datetimes]),
        'all_hi': all_hi,
        'segments': segments
    }
    mat_file_path = os.path.join(output_dir, "health_index_data.mat")
    savemat(mat_file_path, matlab_data)
    print(f"Data saved to {mat_file_path}")


class AutoEncoderVAE(nn.Module):
    def __init__(self, input_dim):
        super(AutoEncoderVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.mu = nn.Linear(32, 16)
        self.log_var = nn.Linear(32, 16)

        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu(encoded)
        log_var = self.log_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(z)
        return decoded, mu, log_var

def evaluate_test_performance(normal_scores, fault_scores, threshold, output_dir):
    predictions = np.concatenate([
        (normal_scores >= threshold).astype(int),
        (fault_scores >= threshold).astype(int)
    ])
    true_labels = np.concatenate([
        np.zeros(len(normal_scores)),
        np.ones(len(fault_scores))
    ]).astype(int)

    metrics = {
        'accuracy': accuracy_score(true_labels, predictions),
        'recall': recall_score(true_labels, predictions),
        'precision': precision_score(true_labels, predictions),
        'f1': f1_score(true_labels, predictions)
    }
    print("\nModel Performance Metrics:")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name.capitalize():>9}: {value:.4f}")
    print("-" * 50)

    conf_matrix = confusion_matrix(true_labels, predictions)
    conf_pct = conf_matrix / conf_matrix.sum(axis=1)[:, np.newaxis]
    labels = np.array([
        f"{val:,}\n({pct:.1%})"
        for val, pct in zip(conf_matrix.flatten(), conf_pct.flatten())
    ]).reshape(2, 2)

    plt.figure(figsize=(5, 5))
    sns.heatmap(
        conf_matrix,
        annot=labels,
        fmt='',
        cmap='YlOrRd',
        cbar=True,
        xticklabels=['Normal', 'Fault'],
        yticklabels=['Normal', 'Fault'],
        annot_kws={'size': 12, 'weight': 'bold'},
        square=True
    )

    plt.title('Confusion Matrix', pad=20, size=14, weight='bold')
    plt.xlabel('Predicted Labels', labelpad=10)
    plt.ylabel('True Labels', labelpad=10)
    plt.grid(True, which='minor', color='white', linewidth=0.5)
    plt.tight_layout()

    #
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()

    return predictions, true_labels, metrics

def save_predictions_and_labels(predictions, true_labels, file_path, output_dir = None):
    if output_dir:
        file_path = os.path.join(output_dir, file_path)
    with open(file_path, 'w') as f:
        for pred, true in zip(predictions, true_labels):
            f.write(f"{pred},{true}\n")