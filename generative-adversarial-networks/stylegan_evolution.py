import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon, Ellipse, RegularPolygon
import numpy as np
import os
from matplotlib.collections import PatchCollection

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def create_stylegan_evolution_diagram():
    """
    Create a comprehensive visualization of StyleGAN evolution from StyleGAN to StyleGAN3
    """
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('StyleGAN Evolution: From StyleGAN to StyleGAN3', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Define colors for different versions
    colors = {
        'stylegan': '#FF6B6B',      # Red
        'stylegan2': '#4ECDC4',     # Teal
        'stylegan3': '#45B7D1',     # Blue
        'style': '#FFD166',         # Yellow
        'noise': '#06D6A0',         # Green
        'latent': '#118AB2',        # Dark blue
        'mapping': '#EF476F',       # Pink
        'synthesis': '#073B4C',     # Dark
    }
    
    # ========== STYLEGAN (2019) ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('StyleGAN (2019)', fontsize=14, fontweight='bold', 
                  color=colors['stylegan'], pad=10)
    ax1.set_xlim(-1, 11)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Add background
    ax1.add_patch(Rectangle((-0.5, -0.5), 11.5, 10.5, 
                           facecolor=colors['stylegan'], alpha=0.05, 
                           edgecolor=colors['stylegan'], linewidth=2))
    
    # Mapping Network
    ax1.text(0.5, 9.2, 'Mapping Network', fontsize=10, fontweight='bold', 
             ha='left', va='center')
    mapping_rect = FancyBboxPatch((0.5, 7.5), 3, 1.5, 
                                 boxstyle="round,pad=0.1",
                                 facecolor=colors['mapping'], alpha=0.6,
                                 edgecolor='black', linewidth=1)
    ax1.add_patch(mapping_rect)
    ax1.text(2, 8.25, 'Z → W', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Latent vector input
    ax1.text(2, 9.7, 'Latent z ∈ ℝ⁵¹²', ha='center', va='center', fontsize=8)
    ax1.arrow(2, 9.5, 0, -0.7, head_width=0.2, head_length=0.2, 
              fc=colors['latent'], ec=colors['latent'], linewidth=2)
    
    # Style vector
    ax1.arrow(4, 8.25, 1, 0, head_width=0.2, head_length=0.2, 
              fc=colors['style'], ec=colors['style'], linewidth=2)
    ax1.text(5.5, 8.25, 'Style w ∈ ℝ⁵¹²', ha='left', va='center', fontsize=8)
    
    # Synthesis Network with AdaIN blocks
    ax1.text(7, 9.2, 'Synthesis Network', fontsize=10, fontweight='bold', 
             ha='left', va='center')
    
    # Draw synthesis network blocks
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    n_blocks = min(6, len(resolutions))
    
    for i in range(n_blocks):
        y_pos = 8 - i * 1.2
        
        # AdaIN block
        block_width = 3
        block = FancyBboxPatch((6.5, y_pos - 0.5), block_width, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['synthesis'], alpha=0.5 - i*0.05,
                              edgecolor='black', linewidth=1)
        ax1.add_patch(block)
        
        # Resolution label
        ax1.text(6.5 + block_width/2, y_pos, f'{resolutions[i]}×{resolutions[i]}', 
                ha='center', va='center', fontsize=8)
        
        # Style input to block
        style_dot = Circle((6.3, y_pos), 0.15, facecolor=colors['style'], 
                          edgecolor='black', linewidth=1)
        ax1.add_patch(style_dot)
        
        # Noise input (only for some blocks)
        if i % 2 == 0:
            noise_dot = Circle((6.5 + block_width + 0.3, y_pos), 0.15, 
                              facecolor=colors['noise'], edgecolor='black', linewidth=1)
            ax1.add_patch(noise_dot)
            ax1.text(6.5 + block_width + 0.5, y_pos, 'Noise', 
                    ha='left', va='center', fontsize=7)
        
        # Arrows between blocks
        if i < n_blocks - 1:
            ax1.arrow(6.5 + block_width/2, y_pos - 0.6, 0, -0.9,
                     head_width=0.2, head_length=0.15, fc='black', ec='black',
                     alpha=0.5, linewidth=1)
    
    # Output
    ax1.text(8, 0.5, 'Generated Image\n1024×1024', ha='center', va='center',
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor=colors['stylegan'], alpha=0.3))
    
    # Key innovations
    innovations1 = [
        "• Style-based generator",
        "• AdaIN (Adaptive Instance Norm)",
        "• Stochastic variation via noise",
        "• Progressive growing",
        "• Truncation trick for quality"
    ]
    
    for i, text in enumerate(innovations1):
        ax1.text(0.5, 2 - i*0.6, text, fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    # ========== STYLEGAN2 (2020) ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('StyleGAN2 (2020)', fontsize=14, fontweight='bold', 
                  color=colors['stylegan2'], pad=10)
    ax2.set_xlim(-1, 11)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Add background
    ax2.add_patch(Rectangle((-0.5, -0.5), 11.5, 10.5, 
                           facecolor=colors['stylegan2'], alpha=0.05, 
                           edgecolor=colors['stylegan2'], linewidth=2))
    
    # Mapping Network (improved)
    ax2.text(0.5, 9.2, 'Mapping Network', fontsize=10, fontweight='bold', 
             ha='left', va='center')
    mapping2_rect = FancyBboxPatch((0.5, 7.5), 3, 1.5, 
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['mapping'], alpha=0.6,
                                  edgecolor='black', linewidth=1)
    ax2.add_patch(mapping2_rect)
    ax2.text(2, 8.25, 'Z → W+', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(2, 7.8, '(Extended style space)', ha='center', va='center', fontsize=7)
    
    # Synthesis Network - No progressive growing
    ax2.text(7, 9.2, 'Synthesis Network (Fixed)', fontsize=10, fontweight='bold', 
             ha='left', va='center')
    
    # Draw improved synthesis blocks
    for i in range(6):
        y_pos = 8 - i * 1.2
        block_width = 3
        
        # Improved AdaIN block (weight demodulation)
        block = FancyBboxPatch((6.5, y_pos - 0.5), block_width, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['synthesis'], alpha=0.5 - i*0.05,
                              edgecolor='black', linewidth=1)
        ax2.add_patch(block)
        
        # Add "Demod" label for weight demodulation
        if i > 0:
            demod_text = Rectangle((6.5 + block_width - 0.8, y_pos + 0.2), 0.7, 0.3,
                                  facecolor='red', alpha=0.5, edgecolor='red')
            ax2.add_patch(demod_text)
            ax2.text(6.5 + block_width - 0.45, y_pos + 0.35, 'Demod', 
                    ha='center', va='center', fontsize=6, fontweight='bold', color='white')
        
        # Style input
        style_dot = Circle((6.3, y_pos), 0.15, facecolor=colors['style'], 
                          edgecolor='black', linewidth=1)
        ax2.add_patch(style_dot)
        
        # Skip connections
        if i > 0:
            # Draw skip connection from previous block
            ax2.plot([6.5 + block_width, 6.5 + block_width + 0.5, 
                      6.5 + block_width + 0.5, 6.5 - 0.3],
                    [y_pos + 0.5, y_pos + 0.5, y_pos - 1.2 + 0.5, y_pos - 1.2 + 0.5],
                    'b-', alpha=0.3, linewidth=1, linestyle='--')
        
        # Arrows between blocks
        if i < 5:
            ax2.arrow(6.5 + block_width/2, y_pos - 0.6, 0, -0.9,
                     head_width=0.2, head_length=0.15, fc='black', ec='black',
                     alpha=0.5, linewidth=1)
    
    # Output with path length regularization indicator
    output_box = FancyBboxPatch((7.5, 0.2), 2, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['stylegan2'], alpha=0.5,
                               edgecolor='black', linewidth=2)
    ax2.add_patch(output_box)
    ax2.text(8.5, 0.5, 'Output', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Path length regularization indicator
    path_indicator = Circle((10, 0.5), 0.3, facecolor='green', alpha=0.5,
                           edgecolor='black', linewidth=1)
    ax2.add_patch(path_indicator)
    ax2.text(10, 0.5, 'PPL', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Key improvements
    improvements2 = [
        "• Fixed artifacts (water droplets)",
        "• Weight demodulation",
        "• No progressive growing needed",
        "• Path length regularization",
        "• Lazy regularization",
        "• Improved style mixing"
    ]
    
    for i, text in enumerate(improvements2):
        ax2.text(0.5, 2.5 - i*0.5, text, fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    # ========== STYLEGAN3 (2021) ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('StyleGAN3 (2021)', fontsize=14, fontweight='bold', 
                  color=colors['stylegan3'], pad=10)
    ax3.set_xlim(-1, 11)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # Add background
    ax3.add_patch(Rectangle((-0.5, -0.5), 11.5, 10.5, 
                           facecolor=colors['stylegan3'], alpha=0.05, 
                           edgecolor=colors['stylegan3'], linewidth=2))
    
    # Equivariant mapping network
    ax3.text(0.5, 9.2, 'Equivariant Mapping', fontsize=10, fontweight='bold', 
             ha='left', va='center')
    mapping3_rect = FancyBboxPatch((0.5, 7.5), 3, 1.5, 
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['mapping'], alpha=0.6,
                                  edgecolor='black', linewidth=1)
    ax3.add_patch(mapping3_rect)
    ax3.text(2, 8.25, 'Z → W+', ha='center', va='center', fontsize=9, fontweight='bold')
    ax3.text(2, 7.8, '(Translation equivariant)', ha='center', va='center', fontsize=7)
    
    # Aliasing-free synthesis network
    ax3.text(7, 9.2, 'Aliasing-Free Synthesis', fontsize=10, fontweight='bold', 
             ha='left', va='center')
    
    # Draw aliasing-free blocks
    for i in range(6):
        y_pos = 8 - i * 1.2
        block_width = 3
        
        # Simplified aliasing-free block
        block = FancyBboxPatch((6.5, y_pos - 0.5), block_width, 1,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['synthesis'], alpha=0.5 - i*0.05,
                              edgecolor='black', linewidth=1)
        ax3.add_patch(block)
        
        # Add "AF" for aliasing-free
        af_text = Rectangle((6.5, y_pos + 0.2), 0.7, 0.3,
                           facecolor='purple', alpha=0.5, edgecolor='purple')
        ax3.add_patch(af_text)
        ax3.text(6.5 + 0.35, y_pos + 0.35, 'AF', 
                ha='center', va='center', fontsize=6, fontweight='bold', color='white')
        
        # Style input
        style_dot = Circle((6.3, y_pos), 0.15, facecolor=colors['style'], 
                          edgecolor='black', linewidth=1)
        ax3.add_patch(style_dot)
        
        # Show translation equivariance
        if i == 2:
            # Add translation arrows
            ax3.arrow(6.5 + block_width + 0.5, y_pos, 1, 0, 
                     head_width=0.15, head_length=0.15, fc='blue', ec='blue', alpha=0.5)
            ax3.arrow(6.5 + block_width + 0.5, y_pos, 0, 0.8, 
                     head_width=0.15, head_length=0.15, fc='blue', ec='blue', alpha=0.5)
            ax3.text(6.5 + block_width + 1.2, y_pos + 0.4, 'Equivariant', 
                    ha='left', va='center', fontsize=7, rotation=45)
        
        # Arrows between blocks
        if i < 5:
            ax3.arrow(6.5 + block_width/2, y_pos - 0.6, 0, -0.9,
                     head_width=0.2, head_length=0.15, fc='black', ec='black',
                     alpha=0.5, linewidth=1)
    
    # Output with improved interpolation
    output_box3 = FancyBboxPatch((7.5, 0.2), 2, 0.6,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['stylegan3'], alpha=0.5,
                                edgecolor='black', linewidth=2)
    ax3.add_patch(output_box3)
    ax3.text(8.5, 0.5, 'Smooth Output', ha='center', va='center', 
            fontsize=9, fontweight='bold')
    
    # Key innovations
    innovations3 = [
        "• Aliasing-free architecture",
        "• Translation/rotation equivariance",
        "• Improved interpolation",
        "• No positional references",
        "• Simplified up/downsampling",
        "• Better motion in videos"
    ]
    
    for i, text in enumerate(innovations3):
        ax3.text(0.5, 2.5 - i*0.5, text, fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    # ========== QUALITY COMPARISON ==========
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('Quality Metrics Evolution', fontsize=14, fontweight='bold', pad=10)
    
    # Simulate quality metrics over versions
    versions = ['StyleGAN', 'StyleGAN2', 'StyleGAN3']
    metrics = {
        'FID (lower better)': [8.5, 4.5, 3.2],
        'Precision': [0.65, 0.78, 0.82],
        'Recall': [0.42, 0.52, 0.61],
        'PPL (lower better)': [350, 125, 85],
    }
    
    x = np.arange(len(versions))
    width = 0.2
    multiplier = 0
    
    colors_metrics = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD166']
    
    for i, (metric, values) in enumerate(metrics.items()):
        offset = width * multiplier
        bars = ax4.bar(x + offset, values, width, label=metric, 
                      color=colors_metrics[i], alpha=0.7, edgecolor='black')
        ax4.bar_label(bars, padding=3, fmt='%.2f' if metric != 'PPL (lower better)' else '%.0f')
        multiplier += 1
    
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_xticks(x + width * 1.5)
    ax4.set_xticklabels(versions, fontsize=11, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add annotation about improvements
    ax4.text(0.5, ax4.get_ylim()[1] * 0.9, '↑ Quality\n↑ Diversity\n↑ Training Stability',
             fontsize=9, ha='center', va='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # ========== SAMPLE EVOLUTION ==========
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title('Sample Quality Evolution', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    
    # Draw sample progression
    sample_stages = [
        {'version': 'StyleGAN', 'y': 8, 'issues': ['Water droplets', 'Texture sticking']},
        {'version': 'StyleGAN2', 'y': 5, 'issues': ['Fixed artifacts', 'Better details']},
        {'version': 'StyleGAN3', 'y': 2, 'issues': ['Smooth interpolation', 'No aliasing']},
    ]
    
    for stage in sample_stages:
        # Sample "face" representation
        face_radius = 1.0 if stage['version'] == 'StyleGAN' else 1.2 if stage['version'] == 'StyleGAN2' else 1.4
        face = Circle((5, stage['y']), face_radius, 
                     facecolor='lightgray', edgecolor='black', linewidth=2,
                     alpha=0.7)
        ax5.add_patch(face)
        
        # Add features (eyes, mouth)
        # Eyes
        left_eye = Circle((4.5, stage['y'] + 0.3), 0.2, facecolor='black')
        right_eye = Circle((5.5, stage['y'] + 0.3), 0.2, facecolor='black')
        ax5.add_patch(left_eye)
        ax5.add_patch(right_eye)
        
        # Mouth
        mouth = patches.Arc((5, stage['y'] - 0.3), 1, 0.5, 
                           theta1=200, theta2=340, linewidth=2, color='black')
        ax5.add_patch(mouth)
        
        # Version label
        ax5.text(5, stage['y'] - 1.8, stage['version'], 
                ha='center', va='center', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.2", 
                         facecolor=colors[stage['version'].lower().replace(' ', '')], 
                         alpha=0.3))
        
        # Issues/fixes
        for i, issue in enumerate(stage['issues']):
            ax5.text(7, stage['y'] - i*0.5, f'• {issue}', 
                    ha='left', va='center', fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    # ========== TRAINING IMPROVEMENTS ==========
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title('Training Stability Improvements', fontsize=12, fontweight='bold', pad=10)
    
    # Simulate training curves
    epochs = np.linspace(0, 100, 100)
    
    # StyleGAN training (unstable)
    stylegan_loss = 2.5 * np.exp(-epochs/50) + 0.5 * np.sin(epochs/5) + 0.8 * np.exp(-epochs/80)
    stylegan_loss += 0.3 * (np.sin(epochs/3) > 0.7)  # Add spikes for instability
    
    # StyleGAN2 training (more stable)
    stylegan2_loss = 2.0 * np.exp(-epochs/40) + 0.3 * np.sin(epochs/10) + 0.5 * np.exp(-epochs/70)
    
    # StyleGAN3 training (most stable)
    stylegan3_loss = 1.8 * np.exp(-epochs/30) + 0.1 * np.sin(epochs/15) + 0.3 * np.exp(-epochs/60)
    
    ax6.plot(epochs, stylegan_loss, color=colors['stylegan'], linewidth=2, 
             label='StyleGAN', linestyle='-')
    ax6.plot(epochs, stylegan2_loss, color=colors['stylegan2'], linewidth=2, 
             label='StyleGAN2', linestyle='--')
    ax6.plot(epochs, stylegan3_loss, color=colors['stylegan3'], linewidth=2, 
             label='StyleGAN3', linestyle='-.')
    
    ax6.set_xlabel('Training Epochs', fontsize=10)
    ax6.set_ylabel('Generator Loss', fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    
    # Add stability annotations
    ax6.text(20, 2.2, 'Unstable\nOscillations', fontsize=8, color=colors['stylegan'],
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax6.text(60, 1.0, 'More Stable', fontsize=8, color=colors['stylegan2'],
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    ax6.text(80, 0.4, 'Most Stable', fontsize=8, color=colors['stylegan3'],
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # ========== APPLICATIONS EVOLUTION ==========
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_title('Applications & Capabilities', fontsize=12, fontweight='bold', pad=10)
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 10)
    ax7.axis('off')
    
    applications = [
        {'version': 'StyleGAN', 'y': 8, 'apps': ['High-res faces', 'Art generation', 'Limited editing']},
        {'version': 'StyleGAN2', 'y': 5.5, 'apps': ['Better editing', 'Video synthesis', '3D-aware']},
        {'version': 'StyleGAN3', 'y': 3, 'apps': ['Video editing', 'Animation', 'Real-time']},
    ]
    
    for app in applications:
        # Version box
        version_box = FancyBboxPatch((1, app['y'] - 0.5), 2.5, 1,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors[app['version'].lower().replace(' ', '')], 
                                    alpha=0.5, edgecolor='black', linewidth=1)
        ax7.add_patch(version_box)
        ax7.text(2.25, app['y'], app['version'], 
                ha='center', va='center', fontsize=9, fontweight='bold')
        
        # Applications
        for i, application in enumerate(app['apps']):
            app_box = FancyBboxPatch((4, app['y'] - 0.3 + i*0.6), 5, 0.5,
                                    boxstyle="round,pad=0.05",
                                    facecolor='lightgray', alpha=0.7,
                                    edgecolor='black', linewidth=0.5)
            ax7.add_patch(app_box)
            ax7.text(6.5, app['y'] - 0.05 + i*0.6, application,
                    ha='center', va='center', fontsize=8)
            
            # Arrow from version to application
            ax7.arrow(3.5, app['y'], 0.3, i*0.6 - 0.3, 
                     head_width=0.1, head_length=0.1, fc='black', ec='black',
                     alpha=0.5, linewidth=1)
    
    # Timeline arrow
    ax7.arrow(2.25, 8.5, 0, -5.5, head_width=0.2, head_length=0.2,
             fc='black', ec='black', linewidth=2, alpha=0.7)
    ax7.text(2.6, 5.5, 'Evolution →', fontsize=9, rotation=90,
            ha='center', va='center', fontweight='bold')
    
    # Add summary text
    summary_text = """
    Key Evolutionary Steps:
    
    1. StyleGAN (2019): Introduced style-based generation and AdaIN
    2. StyleGAN2 (2020): Fixed artifacts, added weight demodulation
    3. StyleGAN3 (2021): Aliasing-free, equivariant to transformations
    
    Overall Trend: Better quality, more stable training, wider applications
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/stylegan_evolution.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def create_stylegan_comparison_grid():
    """
    Create a grid showing comparison of generated samples across StyleGAN versions
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('StyleGAN Family: Generated Samples Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Define sample "images" as grids with different patterns
    np.random.seed(42)
    
    # Column titles
    titles = ['StyleGAN (2019)', 'StyleGAN2 (2020)', 'StyleGAN3 (2021)', 'Improvement']
    
    for col, title in enumerate(titles):
        axes[0, col].set_title(title, fontsize=12, fontweight='bold', 
                              color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#000000'][col], 
                              pad=10)
    
    # Row titles
    row_titles = ['Face Generation', 'Art Synthesis', 'Interpolation Quality']
    
    for row, title in enumerate(row_titles):
        axes[row, 0].text(-0.5, 5, title, fontsize=12, fontweight='bold',
                         rotation=90, ha='center', va='center')
    
    # Create sample visualizations
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis('off')
            
            # Create different patterns for each version
            if row == 0:  # Face generation
                if col == 0:  # StyleGAN
                    # Simulate face with some artifacts
                    face = Circle((5, 5), 3, facecolor='#FFCCCC', edgecolor='black', linewidth=2)
                    ax.add_patch(face)
                    
                    # Add water droplet artifacts
                    for _ in range(5):
                        x = np.random.uniform(3, 7)
                        y = np.random.uniform(3, 7)
                        droplet = Circle((x, y), 0.3, facecolor='blue', alpha=0.3)
                        ax.add_patch(droplet)
                    
                    # Features with some sticking
                    left_eye = Circle((3.5, 5.5), 0.8, facecolor='black', alpha=0.8)
                    right_eye = Circle((6.5, 5.5), 0.8, facecolor='black', alpha=0.8)
                    mouth = patches.Arc((5, 3.5), 3, 1.5, theta1=220, theta2=320, 
                                       linewidth=3, color='black')
                    ax.add_patch(left_eye)
                    ax.add_patch(right_eye)
                    ax.add_patch(mouth)
                    
                    ax.text(5, 9, 'Some artifacts', fontsize=9, ha='center', 
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                    
                elif col == 1:  # StyleGAN2
                    # Improved face without droplets
                    face = Circle((5, 5), 3, facecolor='#CCFFCC', edgecolor='black', linewidth=2)
                    ax.add_patch(face)
                    
                    # Better features using Ellipse
                    left_eye = Ellipse((3.5, 5.5), 1, 0.8, angle=10, 
                                      facecolor='black', alpha=0.9)
                    right_eye = Ellipse((6.5, 5.5), 1, 0.8, angle=-10, 
                                       facecolor='black', alpha=0.9)
                    mouth = patches.Arc((5, 3.5), 3, 1.2, theta1=230, theta2=310, 
                                       linewidth=3, color='black')
                    ax.add_patch(left_eye)
                    ax.add_patch(right_eye)
                    ax.add_patch(mouth)
                    
                    ax.text(5, 9, 'Cleaner details', fontsize=9, ha='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
                    
                else:  # StyleGAN3
                    # Most realistic face
                    face = Circle((5, 5), 3, facecolor='#CCCCFF', edgecolor='black', linewidth=2)
                    ax.add_patch(face)
                    
                    # Natural-looking features using Ellipse
                    left_eye = Ellipse((3.5, 5.5), 1.2, 0.7, angle=15, 
                                      facecolor='#333333', alpha=0.9)
                    right_eye = Ellipse((6.5, 5.5), 1.2, 0.7, angle=-15, 
                                       facecolor='#333333', alpha=0.9)
                    mouth = patches.Arc((5, 3.8), 2.5, 1, theta1=240, theta2=300, 
                                       linewidth=2.5, color='#222222')
                    ax.add_patch(left_eye)
                    ax.add_patch(right_eye)
                    ax.add_patch(mouth)
                    
                    # Add subtle details
                    nose = Polygon([[5, 5], [4.7, 4.5], [5.3, 4.5]], 
                                  facecolor='#444444', alpha=0.6)
                    ax.add_patch(nose)
                    
                    ax.text(5, 9, 'Most realistic', fontsize=9, ha='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
            
            elif row == 1:  # Art synthesis
                # Create abstract art patterns
                if col == 0:
                    # StyleGAN - less coherent
                    for i in range(20):
                        x = np.random.uniform(2, 8)
                        y = np.random.uniform(2, 8)
                        size = np.random.uniform(0.5, 2)
                        color = np.random.choice(['#FF6B6B', '#FFD166', '#06D6A0'])
                        circle = Circle((x, y), size, facecolor=color, alpha=0.6)
                        ax.add_patch(circle)
                    ax.text(5, 9, 'Abstract pattern', fontsize=9, ha='center')
                    
                elif col == 1:
                    # StyleGAN2 - more structured
                    for i in range(5):
                        for j in range(5):
                            x = 2 + i * 1.5
                            y = 2 + j * 1.5
                            color_idx = (i+j)%3
                            colors = ['#4ECDC4', '#FFD166', '#EF476F']
                            circle = Circle((x, y), 0.6, facecolor=colors[color_idx], alpha=0.7)
                            ax.add_patch(circle)
                    ax.text(5, 9, 'Structured art', fontsize=9, ha='center')
                    
                else:
                    # StyleGAN3 - most artistic
                    # Create a radial pattern
                    for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                        x = 5 + 3 * np.cos(angle)
                        y = 5 + 3 * np.sin(angle)
                        color = plt.cm.hsv(angle/(2*np.pi))
                        # Use regular polygon instead of star
                        polygon = RegularPolygon((x, y), 6, radius=0.8, 
                                                orientation=angle,
                                                facecolor=color, alpha=0.8)
                        ax.add_patch(polygon)
                    ax.text(5, 9, 'Artistic pattern', fontsize=9, ha='center')
            
            else:  # Interpolation quality
                # Show interpolation between two points
                if col == 0:
                    # StyleGAN - jerky interpolation
                    for i in range(5):
                        x = 2 + i * 1.5
                        y = 5 + np.sin(i) * 1.5  # Non-linear
                        circle = Circle((x, y), 0.5, facecolor='#FF6B6B', alpha=0.7)
                        ax.add_patch(circle)
                    ax.plot([2, 9], [4.5, 4.5], 'k--', alpha=0.3)
                    ax.text(5, 9, 'Non-linear interp', fontsize=9, ha='center')
                    
                elif col == 1:
                    # StyleGAN2 - smoother
                    for i in range(5):
                        x = 2 + i * 1.75
                        y = 5
                        circle = Circle((x, y), 0.5, facecolor='#4ECDC4', alpha=0.7)
                        ax.add_patch(circle)
                    ax.plot([2, 9], [5, 5], 'k-', alpha=0.5)
                    ax.text(5, 9, 'Smoother', fontsize=9, ha='center')
                    
                else:
                    # StyleGAN3 - perfectly smooth
                    for i in range(7):
                        x = 2 + i * 1.2
                        y = 5
                        circle = Circle((x, y), 0.4, facecolor='#45B7D1', alpha=0.8)
                        ax.add_patch(circle)
                    ax.plot([2, 9], [5, 5], 'b-', alpha=0.7, linewidth=2)
                    ax.text(5, 9, 'Perfectly smooth', fontsize=9, ha='center')
        
        # Improvement column
        ax_improve = axes[row, 3]
        ax_improve.set_xlim(0, 10)
        ax_improve.set_ylim(0, 10)
        ax_improve.axis('off')
        
        improvements = [
            ['• Fixed water droplets', '• Reduced texture sticking', '• Better details'],
            ['• More coherent patterns', '• Better structure', '• Artistic control'],
            ['• Linear interpolation', '• No aliasing', '• Smooth transitions']
        ]
        
        for i, point in enumerate(improvements[row]):
            y_pos = 8 - i * 1.5
            ax_improve.text(1, y_pos, point, fontsize=9, ha='left', va='center',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.5))
        
        # Arrow showing improvement
        ax_improve.arrow(7, 8, -4, 0, head_width=0.3, head_length=0.4,
                        fc='green', ec='green', linewidth=2, alpha=0.7)
        ax_improve.text(7.5, 8.5, 'Improvement →', fontsize=9, fontweight='bold', color='green')
    
    plt.tight_layout()
    plt.savefig('images/stylegan_comparison_grid.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_stylegan_technical_details():
    """
    Create a detailed technical comparison of StyleGAN innovations
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('StyleGAN Technical Innovations Comparison', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Technical details for each version
    technical_data = {
        'StyleGAN': {
            'color': '#FF6B6B',
            'innovations': [
                ('Style-based Generator', 'Separate mapping & synthesis networks'),
                ('AdaIN', 'Adaptive Instance Normalization for style control'),
                ('Noise Inputs', 'Stochastic variation at multiple resolutions'),
                ('Progressive Growing', 'Train from low to high resolution'),
                ('Mixing Regularization', 'Random style mixing during training')
            ],
            'limitations': [
                'Texture sticking artifacts',
                'Water droplet-like artifacts',
                'Phase artifacts in progressive growing',
                'Poor interpolation in latent space'
            ]
        },
        'StyleGAN2': {
            'color': '#4ECDC4',
            'innovations': [
                ('Weight Demodulation', 'Replace AdaIN with weight modulation/demod'),
                ('No Progressive Growing', 'Fixed generator architecture'),
                ('Path Length Regularization', 'Smoother latent space interpolation'),
                ('Lazy Regularization', 'Apply regularization less frequently'),
                ('Style Mixing Improvements', 'Better disentanglement')
            ],
            'limitations': [
                'Still some positional references',
                'Not fully translation equivariant',
                'Aliasing in generated images'
            ]
        },
        'StyleGAN3': {
            'color': '#45B7D1',
            'innovations': [
                ('Aliasing-Free Design', 'Careful signal processing in all operations'),
                ('Translation Equivariance', 'Consistent output under translation'),
                ('Simplified Architecture', 'Removed unnecessary complexity'),
                ('Improved Upsampling', 'Proper upsampling filters'),
                ('Continuous Signals', 'Treat all signals as continuous')
            ],
            'limitations': [
                'Higher computational cost',
                'More complex implementation',
                'Still evolving architecture'
            ]
        }
    }
    
    # Plot technical details
    for idx, (version, data) in enumerate(technical_data.items()):
        ax = axes[idx // 2, idx % 2]
        ax.set_title(version, fontsize=14, fontweight='bold', 
                    color=data['color'], pad=10)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        # Add background
        ax.add_patch(Rectangle((0, 0), 10, 10, 
                              facecolor=data['color'], alpha=0.05, 
                              edgecolor=data['color'], linewidth=2))
        
        # Innovations
        ax.text(1, 9.5, 'Key Innovations:', fontsize=11, fontweight='bold', 
                ha='left', va='center')
        
        for i, (innovation, description) in enumerate(data['innovations']):
            y_pos = 8.5 - i * 0.8
            # Innovation box
            innov_box = FancyBboxPatch((1, y_pos - 0.3), 8, 0.6,
                                      boxstyle="round,pad=0.1",
                                      facecolor='white', alpha=0.8,
                                      edgecolor='black', linewidth=1)
            ax.add_patch(innov_box)
            
            # Innovation name
            ax.text(1.2, y_pos, innovation, fontsize=9, fontweight='bold',
                   ha='left', va='center')
            
            # Description
            ax.text(5, y_pos, description, fontsize=8, ha='left', va='center',
                   style='italic')
        
        # Limitations
        ax.text(1, 4, 'Limitations Fixed in Next Version:', 
                fontsize=11, fontweight='bold', ha='left', va='center')
        
        for i, limitation in enumerate(data['limitations']):
            y_pos = 3.5 - i * 0.6
            limit_box = FancyBboxPatch((1, y_pos - 0.2), 8, 0.4,
                                      boxstyle="round,pad=0.05",
                                      facecolor='#FFCCCC', alpha=0.6,
                                      edgecolor='red', linewidth=0.5)
            ax.add_patch(limit_box)
            ax.text(1.2, y_pos, f'• {limitation}', fontsize=8, ha='left', va='center')
    
    # Evolution timeline
    ax_timeline = axes[1, 1]
    ax_timeline.clear()
    ax_timeline.set_title('Evolution Timeline & Impact', fontsize=14, fontweight='bold', pad=10)
    ax_timeline.set_xlim(2018, 2023)
    ax_timeline.set_ylim(0, 100)
    ax_timeline.set_xlabel('Year', fontsize=11)
    ax_timeline.set_ylabel('Impact Score (Relative)', fontsize=11)
    ax_timeline.grid(True, alpha=0.3)
    
    # Timeline data
    years = [2018.5, 2019.5, 2020.5, 2021.5, 2022.5]
    impact = [30, 85, 92, 95, 97]  # Relative impact scores
    versions = ['ProGAN', 'StyleGAN', 'StyleGAN2', 'StyleGAN3', 'StyleGAN3-T']
    
    # Plot impact curve
    ax_timeline.plot(years, impact, 'b-o', linewidth=3, markersize=8, 
                    markerfacecolor='white', markeredgewidth=2)
    
    # Add version labels
    for year, imp, version in zip(years, impact, versions):
        ax_timeline.text(year, imp + 3, version, fontsize=9, ha='center', 
                        fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Add annotations
    ax_timeline.annotate('Style-based\ngeneration', xy=(2019.5, 85), xytext=(2018.8, 70),
                        arrowprops=dict(arrowstyle='->', color='red', linewidth=2),
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    ax_timeline.annotate('Aliasing-free\ndesign', xy=(2021.5, 95), xytext=(2022, 80),
                        arrowprops=dict(arrowstyle='->', color='green', linewidth=2),
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('images/stylegan_technical_details.png', dpi=300, 
                bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Generating StyleGAN evolution visualizations...")
    
    # Generate main evolution diagram
    fig1 = create_stylegan_evolution_diagram()
    print("✓ Generated 'images/stylegan_evolution.png'")
    
    # Generate comparison grid
    create_stylegan_comparison_grid()
    print("✓ Generated 'images/stylegan_comparison_grid.png'")
    
    # Generate technical details
    create_stylegan_technical_details()
    print("✓ Generated 'images/stylegan_technical_details.png'")
    
    print("\nAll StyleGAN evolution visualizations have been generated successfully!")
    print("The visualizations include:")
    print("1. Architecture evolution from StyleGAN to StyleGAN3")
    print("2. Quality metrics and training stability improvements")
    print("3. Sample quality comparison across versions")
    print("4. Technical innovations and limitations")