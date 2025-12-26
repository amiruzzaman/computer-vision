import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from sklearn.datasets import make_moons, make_circles, make_blobs
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def generate_mode_collapse_demo():
    """
    Generate a visual demonstration of mode collapse in GANs
    """
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Mode Collapse in Generative Adversarial Networks', fontsize=16, fontweight='bold', y=0.98)
    
    # Generate different types of data distributions
    n_samples = 1000
    
    # 1. 8 Gaussians (multi-modal distribution)
    centers = [(0, 0), (2, 2), (2, -2), (-2, 2), (-2, -2), (4, 0), (0, 4), (-4, 0)]
    gaussian_data = []
    for center in centers:
        points = np.random.randn(n_samples // 8, 2) * 0.3 + np.array(center)
        gaussian_data.append(points)
    gaussian_data = np.vstack(gaussian_data)
    
    # 2. Mixture of 4 Gaussians
    centers4 = [(1.5, 1.5), (1.5, -1.5), (-1.5, 1.5), (-1.5, -1.5)]
    gaussian_data4 = []
    for center in centers4:
        points = np.random.randn(n_samples // 4, 2) * 0.4 + np.array(center)
        gaussian_data4.append(points)
    gaussian_data4 = np.vstack(gaussian_data4)
    
    # 3. Circle data
    circle_data, _ = make_circles(n_samples=n_samples, factor=0.5, noise=0.05)
    circle_data = circle_data * 3
    
    # Plot the target distributions
    axes[0, 0].scatter(gaussian_data[:, 0], gaussian_data[:, 1], alpha=0.6, s=10, color='blue')
    axes[0, 0].set_title('Target Distribution: 8 Gaussians', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlim(-6, 6)
    axes[0, 0].set_ylim(-6, 6)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')
    
    axes[0, 1].scatter(gaussian_data4[:, 0], gaussian_data4[:, 1], alpha=0.6, s=10, color='green')
    axes[0, 1].set_title('Target Distribution: 4 Gaussians', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_aspect('equal')
    
    axes[0, 2].scatter(circle_data[:, 0], circle_data[:, 1], alpha=0.6, s=10, color='red')
    axes[0, 2].set_title('Target Distribution: Concentric Circles', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlim(-4, 4)
    axes[0, 2].set_ylim(-4, 4)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_aspect('equal')
    
    # Simulate mode collapse scenarios
    # 1. Partial mode collapse (captures only 3 of 8 modes)
    partial_collapse_centers = [(0, 0), (2, 2), (-2, -2)]
    partial_collapse_data = []
    for center in partial_collapse_centers:
        points = np.random.randn(n_samples // 3, 2) * 0.3 + np.array(center)
        partial_collapse_data.append(points)
    partial_collapse_data = np.vstack(partial_collapse_data)
    
    # 2. Severe mode collapse (captures only 1 of 4 modes)
    severe_collapse_centers = [(-1.5, 1.5)]
    severe_collapse_data = np.random.randn(n_samples, 2) * 0.4 + np.array(severe_collapse_centers[0])
    
    # 3. Mode collapse on circles (only captures outer circle)
    theta = np.linspace(0, 2*np.pi, n_samples)
    r = 2.5  # Only outer circle
    collapse_circle_data = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
    collapse_circle_data += np.random.randn(n_samples, 2) * 0.1
    
    # Plot the mode collapse scenarios
    axes[1, 0].scatter(partial_collapse_data[:, 0], partial_collapse_data[:, 1], 
                       alpha=0.6, s=10, color='orange')
    
    # Highlight the captured modes
    for center in partial_collapse_centers:
        axes[1, 0].scatter(center[0], center[1], s=200, marker='*', color='gold', edgecolor='black', linewidth=1)
    
    axes[1, 0].set_title('Partial Mode Collapse\n(Captures 3 of 8 modes)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlim(-6, 6)
    axes[1, 0].set_ylim(-6, 6)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_aspect('equal')
    
    axes[1, 1].scatter(severe_collapse_data[:, 0], severe_collapse_data[:, 1], 
                       alpha=0.6, s=10, color='orange')
    
    # Highlight the single captured mode
    axes[1, 1].scatter(-1.5, 1.5, s=200, marker='*', color='gold', edgecolor='black', linewidth=1)
    
    axes[1, 1].set_title('Severe Mode Collapse\n(Captures 1 of 4 modes)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim(-4, 4)
    axes[1, 1].set_ylim(-4, 4)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_aspect('equal')
    
    axes[1, 2].scatter(collapse_circle_data[:, 0], collapse_circle_data[:, 1], 
                       alpha=0.6, s=10, color='orange')
    
    # Highlight that only outer circle is captured
    circle = plt.Circle((0, 0), 2.5, color='gold', fill=False, linewidth=3, linestyle='--', alpha=0.7)
    axes[1, 2].add_artist(circle)
    
    axes[1, 2].set_title('Mode Collapse on Circles\n(Only outer circle captured)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlim(-4, 4)
    axes[1, 2].set_ylim(-4, 4)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_aspect('equal')
    
    # Add labels
    for i, ax in enumerate(axes[0]):
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
    
    for i, ax in enumerate(axes[1]):
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('y', fontsize=10)
    
    # Add descriptive text
    plt.figtext(0.02, 0.02, 
                "Mode Collapse: Generator learns to produce only a limited subset of the true data distribution,\n"
                "ignoring other modes. This results in reduced diversity and failure to capture the full data manifold.",
                fontsize=11, style='italic', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/mode_collapse.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def create_gan_training_visualization():
    """
    Create a visualization of GAN training progression with/without mode collapse
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('GAN Training Dynamics and Mode Collapse', fontsize=16, fontweight='bold')
    
    # Simulate training progression
    epochs = 50
    n_samples = 100
    
    # 1. Good training - captures all modes
    np.random.seed(42)
    
    # Target distribution: 4 Gaussians
    target_centers = [(2, 2), (2, -2), (-2, 2), (-2, -2)]
    
    # Simulate generator outputs at different epochs
    epochs_to_show = [5, 25, 50]
    colors = ['lightblue', 'blue', 'darkblue']
    
    axes[0].set_title('Ideal Training: Captures All Modes', fontsize=12, fontweight='bold')
    
    # Plot target distribution
    for center in target_centers:
        points = np.random.randn(n_samples, 2) * 0.5 + np.array(center)
        axes[0].scatter(points[:, 0], points[:, 1], alpha=0.2, s=10, color='gray', label='Target' if center == target_centers[0] else "")
    
    # Simulate generator learning progression
    for epoch, color in zip(epochs_to_show, colors):
        # As training progresses, generator captures more modes
        if epoch == 5:
            # Early training - captures 1 mode poorly
            gen_data = np.random.randn(n_samples, 2) * 1.0
        elif epoch == 25:
            # Mid training - captures 2 modes
            centers = [(2, 2), (-2, -2)]
            gen_data = []
            for center in centers:
                points = np.random.randn(n_samples // 2, 2) * 0.7 + np.array(center)
                gen_data.append(points)
            gen_data = np.vstack(gen_data)
        else:  # epoch 50
            # Final - captures all 4 modes
            gen_data = []
            for center in target_centers:
                points = np.random.randn(n_samples // 4, 2) * 0.5 + np.array(center)
                gen_data.append(points)
            gen_data = np.vstack(gen_data)
        
        axes[0].scatter(gen_data[:, 0], gen_data[:, 1], alpha=0.6, s=20, color=color, 
                       label=f'Epoch {epoch}', edgecolors='black', linewidth=0.5)
    
    axes[0].set_xlim(-5, 5)
    axes[0].set_ylim(-5, 5)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[0].set_aspect('equal')
    
    # 2. Mode collapse training - captures only 1 mode
    axes[1].set_title('Mode Collapse: Captures Only 1 Mode', fontsize=12, fontweight='bold')
    
    # Plot target distribution
    for center in target_centers:
        points = np.random.randn(n_samples, 2) * 0.5 + np.array(center)
        axes[1].scatter(points[:, 0], points[:, 1], alpha=0.2, s=10, color='gray', label='Target' if center == target_centers[0] else "")
    
    # Simulate generator with mode collapse
    for epoch, color in zip(epochs_to_show, colors):
        # Generator collapses to single mode
        if epoch == 5:
            gen_data = np.random.randn(n_samples, 2) * 1.0 + np.array([0, 0])
        elif epoch == 25:
            # Collapses to (-2, 2)
            gen_data = np.random.randn(n_samples, 2) * 0.5 + np.array([-2, 2])
        else:  # epoch 50
            # Still stuck at (-2, 2)
            gen_data = np.random.randn(n_samples, 2) * 0.3 + np.array([-2, 2])
        
        axes[1].scatter(gen_data[:, 0], gen_data[:, 1], alpha=0.6, s=20, color=color, 
                       label=f'Epoch {epoch}', edgecolors='black', linewidth=0.5)
    
    # Highlight the collapsed mode
    axes[1].scatter(-2, 2, s=300, marker='*', color='red', edgecolor='black', linewidth=2, 
                   label='Collapsed Mode', zorder=5)
    
    axes[1].set_xlim(-5, 5)
    axes[1].set_ylim(-5, 5)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].set_aspect('equal')
    
    # 3. Oscillating mode collapse
    axes[2].set_title('Mode Dropping: Oscillates Between Modes', fontsize=12, fontweight='bold')
    
    # Plot target distribution
    for center in target_centers:
        points = np.random.randn(n_samples, 2) * 0.5 + np.array(center)
        axes[2].scatter(points[:, 0], points[:, 1], alpha=0.2, s=10, color='gray', label='Target' if center == target_centers[0] else "")
    
    # Simulate mode dropping/oscillation
    modes_sequence = [(2, 2), (-2, -2), (2, -2), (-2, 2)]  # Oscillating sequence
    
    for epoch, color in zip(epochs_to_show, colors):
        if epoch == 5:
            current_mode = modes_sequence[0]
        elif epoch == 25:
            current_mode = modes_sequence[1]
        else:  # epoch 50
            current_mode = modes_sequence[2]
        
        gen_data = np.random.randn(n_samples, 2) * 0.5 + np.array(current_mode)
        axes[2].scatter(gen_data[:, 0], gen_data[:, 1], alpha=0.6, s=20, color=color, 
                       label=f'Epoch {epoch}: Mode {modes_sequence.index(current_mode)+1}', 
                       edgecolors='black', linewidth=0.5)
        
        # Highlight current mode
        axes[2].scatter(current_mode[0], current_mode[1], s=200, marker='s', 
                       color=color, edgecolor='black', linewidth=2, alpha=0.8)
    
    axes[2].set_xlim(-5, 5)
    axes[2].set_ylim(-5, 5)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    axes[2].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('images/gan_training_dynamics.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_comparison_figure():
    """
    Create a comparison figure showing good vs bad GAN training
    """
    fig = plt.figure(figsize=(14, 10))
    
    # Create a grid for the figure
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Main title
    fig.suptitle('GAN Performance: Mode Coverage and Collapse', fontsize=18, fontweight='bold', y=0.95)
    
    # 1. Left: Good GAN training
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax1.set_title('Well-Trained GAN\n(Good Mode Coverage)', fontsize=14, fontweight='bold', pad=15)
    
    # Generate 25 Gaussian mixture
    n_modes = 25
    grid_size = 5
    x_positions = np.linspace(-4, 4, grid_size)
    y_positions = np.linspace(-4, 4, grid_size)
    
    # Plot target distribution (all modes)
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            if np.random.random() > 0.3:  # Skip some for visual clarity
                center = (x, y)
                points = np.random.randn(30, 2) * 0.15 + np.array(center)
                ax1.scatter(points[:, 0], points[:, 1], alpha=0.4, s=15, color='lightgray')
    
    # Simulate generator covering most modes
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            if np.random.random() > 0.2:  # Cover most modes
                center = (x, y)
                points = np.random.randn(40, 2) * 0.2 + np.array(center)
                ax1.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25, color='blue')
    
    ax1.set_xlim(-5, 5)
    ax1.set_ylim(-5, 5)
    ax1.set_xlabel('Latent Dimension 1', fontsize=10)
    ax1.set_ylabel('Latent Dimension 2', fontsize=10)
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal')
    
    # 2. Right: Mode collapsed GAN
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    ax2.set_title('Mode-Collapsed GAN\n(Poor Diversity)', fontsize=14, fontweight='bold', pad=15)
    
    # Plot target distribution (all modes)
    for i, x in enumerate(x_positions):
        for j, y in enumerate(y_positions):
            if np.random.random() > 0.3:
                center = (x, y)
                points = np.random.randn(30, 2) * 0.15 + np.array(center)
                ax2.scatter(points[:, 0], points[:, 1], alpha=0.4, s=15, color='lightgray')
    
    # Simulate generator covering only few modes
    collapsed_modes = [(0, 0), (2, 2), (-2, -2)]  # Only 3 modes captured
    for center in collapsed_modes:
        points = np.random.randn(200, 2) * 0.25 + np.array(center)
        ax2.scatter(points[:, 0], points[:, 1], alpha=0.7, s=25, color='red')
        # Highlight collapsed mode
        ax2.scatter(center[0], center[1], s=300, marker='*', color='gold', 
                   edgecolor='black', linewidth=2, zorder=5)
    
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-5, 5)
    ax2.set_xlabel('Latent Dimension 1', fontsize=10)
    ax2.set_ylabel('Latent Dimension 2', fontsize=10)
    ax2.grid(True, alpha=0.2)
    ax2.set_aspect('equal')
    
    # 3. Bottom: Metrics comparison
    ax3 = fig.add_subplot(gs[2, :])
    
    # Simulate training metrics
    epochs = np.arange(0, 100, 2)
    
    # Good GAN metrics
    good_fid = 100 * np.exp(-epochs/30) + np.random.randn(len(epochs)) * 2 + 10
    good_is = 5 * (1 - np.exp(-epochs/20)) + np.random.randn(len(epochs)) * 0.2 + 2
    
    # Bad GAN metrics (with mode collapse)
    bad_fid = 100 * np.exp(-epochs/15) + 30  # Stops improving
    bad_fid[40:] = bad_fid[40] + np.random.randn(len(epochs)-40) * 3
    bad_is = 5 * (1 - np.exp(-epochs/10)) + 1  # Lower final score
    bad_is[30:] = bad_is[30] + np.random.randn(len(epochs)-30) * 0.1
    
    # Plot FID
    ax3.plot(epochs, good_fid, 'b-', linewidth=2, label='Well-trained GAN (FID)')
    ax3.plot(epochs, bad_fid, 'r--', linewidth=2, label='Mode-collapsed GAN (FID)')
    
    # Add Inception Score on secondary axis
    ax3_secondary = ax3.twinx()
    ax3_secondary.plot(epochs, good_is, 'b-', linewidth=2, alpha=0.5, label='Well-trained GAN (IS)')
    ax3_secondary.plot(epochs, bad_is, 'r--', linewidth=2, alpha=0.5, label='Mode-collapsed GAN (IS)')
    
    ax3.set_xlabel('Training Epochs', fontsize=11)
    ax3.set_ylabel('FID (lower is better)', fontsize=11, color='black')
    ax3_secondary.set_ylabel('Inception Score (higher is better)', fontsize=11, color='black')
    
    ax3.set_title('Training Metrics Comparison', fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_secondary.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Add explanatory text
    fig.text(0.02, 0.02, 
             "Key Insights:\n"
             "• Mode collapse occurs when generator captures only a subset of data distribution\n"
             "• Results in low diversity despite potentially good sample quality\n"
             "• Metrics like FID and Inception Score can detect mode collapse\n"
             "• Solutions: WGAN-GP, minibatch discrimination, spectral normalization",
             fontsize=11, style='italic',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/gan_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Generating mode collapse visualization...")
    
    # Generate the main mode collapse figure
    fig1 = generate_mode_collapse_demo()
    print("✓ Generated 'images/mode_collapse.png'")
    
    # Generate additional visualizations
    create_gan_training_visualization()
    print("✓ Generated 'images/gan_training_dynamics.png'")
    
    create_comparison_figure()
    print("✓ Generated 'images/gan_comparison.png'")
    
    print("\nAll visualizations have been generated successfully!")
    print("These images illustrate various aspects of mode collapse in GANs.")
    
 # pip install numpy matplotlib torch scikit-learn