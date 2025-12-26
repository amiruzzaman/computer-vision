import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patches as patches
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def create_biggan_samples_visualization():
    """
    Create a visualization of BigGAN samples showing high-quality image generation
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('BigGAN: Large Scale GAN Training for High-Fidelity Image Generation', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create a grid for layout
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== BIGGAN OVERVIEW ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('BigGAN Key Innovations', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Background
    ax1.add_patch(Rectangle((0, 0), 10, 10, facecolor='#2E86AB', alpha=0.05, 
                           edgecolor='#2E86AB', linewidth=2))
    
    # Key innovations
    innovations = [
        ('Scale Up', 'Larger models (up to 158M params)\nLarger batches (up to 2048)'),
        ('Orthogonal Regularization', 'Stabilize training\nPrevent mode collapse'),
        ('Truncation Trick', 'Trade diversity for quality\nControl sample fidelity'),
        ('Shared Embedding', 'Conditioning via class embeddings\nBetter class-conditional generation'),
        ('Hinge Loss', 'Improved training stability\nBetter gradient flow'),
    ]
    
    for i, (title, desc) in enumerate(innovations):
        y_pos = 9 - i * 1.8
        
        # Title box
        title_box = FancyBboxPatch((0.5, y_pos - 0.3), 4, 0.6,
                                  boxstyle="round,pad=0.1",
                                  facecolor='#2E86AB', alpha=0.3,
                                  edgecolor='#2E86AB', linewidth=2)
        ax1.add_patch(title_box)
        ax1.text(2.5, y_pos, title, fontsize=11, fontweight='bold',
                ha='center', va='center', color='#2E86AB')
        
        # Description
        desc_box = FancyBboxPatch((5, y_pos - 0.5), 4.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightgray', alpha=0.7,
                                 edgecolor='gray', linewidth=1)
        ax1.add_patch(desc_box)
        ax1.text(7.25, y_pos, desc, fontsize=9, ha='center', va='center')
    
    # ========== BIGGAN ARCHITECTURE ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('BigGAN Architecture Overview', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Architecture visualization
    # Generator components
    components = [
        {'name': 'z ~ N(0, I)', 'type': 'input', 'y': 9, 'color': '#A23B72'},
        {'name': 'Conditioning\nc (class label)', 'type': 'input', 'y': 8, 'color': '#F18F01'},
        {'name': 'Shared Embedding', 'type': 'layer', 'y': 7, 'color': '#2E86AB'},
        {'name': 'Residual Blocks\nwith Self-Attention', 'type': 'layer', 'y': 5.5, 'color': '#73AB84'},
        {'name': 'Upsampling\nLayers', 'type': 'layer', 'y': 4, 'color': '#C73E1D'},
        {'name': 'Output\n256×256×3', 'type': 'output', 'y': 2.5, 'color': '#3D348B'},
    ]
    
    for i, comp in enumerate(components):
        # Draw component
        if comp['type'] == 'input':
            box = FancyBboxPatch((3, comp['y'] - 0.4), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=comp['color'], alpha=0.7,
                                edgecolor='black', linewidth=1)
            ax2.add_patch(box)
            ax2.text(5, comp['y'], comp['name'], fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white')
        elif comp['type'] == 'layer':
            box = FancyBboxPatch((2.5, comp['y'] - 0.6), 5, 1.2,
                                boxstyle="round,pad=0.1",
                                facecolor=comp['color'], alpha=0.6,
                                edgecolor='black', linewidth=1)
            ax2.add_patch(box)
            ax2.text(5, comp['y'], comp['name'], fontsize=9, fontweight='bold',
                    ha='center', va='center', color='white')
        else:  # output
            box = FancyBboxPatch((3, comp['y'] - 0.4), 4, 0.8,
                                boxstyle="round,pad=0.1",
                                facecolor=comp['color'], alpha=0.8,
                                edgecolor='black', linewidth=2)
            ax2.add_patch(box)
            ax2.text(5, comp['y'], comp['name'], fontsize=10, fontweight='bold',
                    ha='center', va='center', color='white')
        
        # Draw arrows between components
        if i < len(components) - 1:
            y_start = comp['y'] - (0.6 if comp['type'] == 'layer' else 0.4)
            y_end = components[i + 1]['y'] + (0.6 if components[i + 1]['type'] == 'layer' else 0.4)
            ax2.arrow(5, y_start, 0, y_end - y_start - 0.2,
                     head_width=0.3, head_length=0.2, fc='black', ec='black',
                     alpha=0.5, linewidth=1)
    
    # Add orthogonal regularization indicator
    ortho_indicator = patches.RegularPolygon((8, 7), 6, radius=0.5,
                                           facecolor='red', alpha=0.6,
                                           edgecolor='black', linewidth=1)
    ax2.add_patch(ortho_indicator)
    ax2.text(8, 7, 'Ortho\nReg', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Add hinge loss indicator
    hinge_indicator = patches.RegularPolygon((8, 5.5), 4, radius=0.4,
                                           facecolor='green', alpha=0.6,
                                           edgecolor='black', linewidth=1)
    ax2.add_patch(hinge_indicator)
    ax2.text(8, 5.5, 'Hinge\nLoss', fontsize=7, fontweight='bold',
            ha='center', va='center', color='white')
    
    # ========== TRUNCATION TRICK VISUALIZATION ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Truncation Trick: Quality vs Diversity Trade-off', 
                  fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # Truncation levels
    trunc_levels = [
        {'psi': 0.5, 'y': 8, 'quality': 'Low', 'diversity': 'High', 'color': '#FF6B6B'},
        {'psi': 1.0, 'y': 5, 'quality': 'Medium', 'diversity': 'Medium', 'color': '#4ECDC4'},
        {'psi': 2.0, 'y': 2, 'quality': 'High', 'diversity': 'Low', 'color': '#45B7D1'},
    ]
    
    for level in trunc_levels:
        # Create sample grid representation
        n_cells = 4
        cell_size = 0.6
        
        # Create grid of "pixels" with varying quality
        for i in range(n_cells):
            for j in range(n_cells):
                x = 1 + i * cell_size
                y = level['y'] + j * cell_size - n_cells*cell_size/2
                
                # Vary cell appearance based on truncation
                if level['psi'] == 0.5:
                    # Low quality: noisy
                    intensity = np.random.uniform(0.3, 0.7)
                    alpha = np.random.uniform(0.4, 0.8)
                elif level['psi'] == 1.0:
                    # Medium quality: somewhat structured
                    intensity = 0.5 + 0.3 * np.sin(i + j) + np.random.uniform(-0.2, 0.2)
                    alpha = 0.7
                else:
                    # High quality: clear structure
                    intensity = 0.6 + 0.3 * np.sin(0.5*i) * np.cos(0.5*j)
                    alpha = 0.9
                
                cell = Rectangle((x, y), cell_size*0.9, cell_size*0.9,
                                facecolor=(intensity, intensity*0.9, intensity*0.8),
                                alpha=alpha, edgecolor='gray', linewidth=0.5)
                ax3.add_patch(cell)
        
        # Truncation level label
        ax3.text(4, level['y'], f'ψ = {level["psi"]}', fontsize=11, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=level['color'], alpha=0.3))
        
        # Quality vs diversity info
        info_text = f'Quality: {level["quality"]}\nDiversity: {level["diversity"]}'
        ax3.text(7, level['y'], info_text, fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    # Add arrow showing trade-off
    ax3.arrow(5.5, 8.5, 0, -6, head_width=0.3, head_length=0.3,
             fc='black', ec='black', linewidth=2, alpha=0.7)
    ax3.text(5.8, 5.5, 'Increasing ψ\n↑ Quality\n↓ Diversity', fontsize=9,
            ha='left', va='center', rotation=90,
            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightyellow', alpha=0.8))
    
    # ========== SAMPLE GRID VISUALIZATION ==========
    ax4 = fig.add_subplot(gs[1, :])
    ax4.set_title('BigGAN Generated Samples (256×256)', fontsize=16, fontweight='bold', pad=10)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Create a 4x4 grid of sample "images"
    n_rows, n_cols = 4, 4
    cell_size = 2.0
    margin = 0.2
    
    # Categories of samples
    categories = [
        'Animals', 'Objects', 'Scenes', 'Food',
        'Vehicles', 'People', 'Art', 'Nature',
        'Buildings', 'Toys', 'Instruments', 'Sports',
        'Electronics', 'Fashion', 'Furniture', 'Abstract'
    ]
    
    for row in range(n_rows):
        for col in range(n_cols):
            x = col * (cell_size + margin) + margin
            y = 10 - (row + 1) * (cell_size + margin) - margin
            
            # Create a "sample image" with patterns
            idx = row * n_cols + col
            
            # Different patterns for different categories
            pattern_type = idx % 4
            
            # Base rectangle
            sample = Rectangle((x, y), cell_size, cell_size,
                              facecolor='white', alpha=0.9,
                              edgecolor='black', linewidth=2)
            ax4.add_patch(sample)
            
            # Add pattern based on category
            if pattern_type == 0:  # Animal-like pattern
                # Fur-like texture
                for i in range(20):
                    px = x + np.random.uniform(0.2, cell_size-0.2)
                    py = y + np.random.uniform(0.2, cell_size-0.2)
                    size = np.random.uniform(0.1, 0.3)
                    intensity = np.random.uniform(0.4, 0.8)
                    fur = patches.Ellipse((px, py), size, size*0.5,
                                         angle=np.random.uniform(0, 180),
                                         facecolor=(intensity, intensity*0.8, intensity*0.6),
                                         alpha=0.6, edgecolor='none')
                    ax4.add_patch(fur)
                
                # Eyes
                eye1 = patches.Circle((x + cell_size*0.3, y + cell_size*0.7), 
                                     cell_size*0.1, facecolor='black')
                eye2 = patches.Circle((x + cell_size*0.7, y + cell_size*0.7), 
                                     cell_size*0.1, facecolor='black')
                ax4.add_patch(eye1)
                ax4.add_patch(eye2)
                
            elif pattern_type == 1:  # Object-like pattern
                # Geometric shapes
                for i in range(8):
                    shape_type = np.random.choice(['circle', 'square', 'triangle'])
                    px = x + np.random.uniform(0.3, cell_size-0.3)
                    py = y + np.random.uniform(0.3, cell_size-0.3)
                    size = np.random.uniform(0.15, 0.25)
                    
                    if shape_type == 'circle':
                        shape = patches.Circle((px, py), size, 
                                              facecolor=np.random.rand(3),
                                              alpha=0.7, edgecolor='black', linewidth=0.5)
                    elif shape_type == 'square':
                        shape = Rectangle((px-size/2, py-size/2), size, size,
                                         facecolor=np.random.rand(3),
                                         alpha=0.7, edgecolor='black', linewidth=0.5)
                    else:  # triangle
                        shape = patches.RegularPolygon((px, py), 3, radius=size,
                                                      facecolor=np.random.rand(3),
                                                      alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax4.add_patch(shape)
                    
            elif pattern_type == 2:  # Scene-like pattern
                # Landscape-like gradient
                for i in range(10):
                    h = cell_size / 10
                    segment = Rectangle((x, y + i*h), cell_size, h,
                                       facecolor=(0.2, 0.3 + i*0.07, 0.5 - i*0.05),
                                       alpha=0.8, edgecolor='none')
                    ax4.add_patch(segment)
                
                # Sun/moon
                celestial = patches.Circle((x + cell_size*0.8, y + cell_size*0.8), 
                                          cell_size*0.15, 
                                          facecolor=(1.0, 0.9, 0.3) if idx % 2 == 0 else (0.9, 0.9, 0.9))
                ax4.add_patch(celestial)
                
            else:  # Food-like pattern
                # Organic shapes
                for i in range(12):
                    px = x + np.random.uniform(0.2, cell_size-0.2)
                    py = y + np.random.uniform(0.2, cell_size-0.2)
                    size = np.random.uniform(0.08, 0.18)
                    
                    # Vary colors like food items
                    colors = [
                        (0.9, 0.7, 0.5),  # bread
                        (0.8, 0.2, 0.2),  # tomato
                        (0.3, 0.6, 0.3),  # vegetable
                        (1.0, 0.8, 0.2),  # cheese
                    ]
                    color = colors[i % len(colors)]
                    
                    food_item = patches.Ellipse((px, py), size, size*0.8,
                                               angle=np.random.uniform(0, 360),
                                               facecolor=color, alpha=0.8,
                                               edgecolor='none')
                    ax4.add_patch(food_item)
            
            # Category label
            ax4.text(x + cell_size/2, y + cell_size + 0.1, categories[idx],
                    fontsize=8, fontweight='bold', ha='center', va='bottom')
    
    # Add note about sample quality
    ax4.text(5, 0.5, 'Note: BigGAN produces photorealistic 256×256 images across diverse categories',
            fontsize=10, ha='center', va='center', style='italic',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # ========== SCALING LAWS VISUALIZATION ==========
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title('Scaling Laws: Model Size vs Quality', fontsize=14, fontweight='bold', pad=10)
    
    # Simulate scaling law data
    model_sizes = ['Small\n(50M)', 'Medium\n(100M)', 'Large\n(158M)']
    x_pos = np.arange(len(model_sizes))
    
    # FID scores (lower is better)
    fid_scores = [25.0, 12.5, 8.5]
    
    # Inception scores (higher is better)
    inception_scores = [45.0, 65.0, 85.0]
    
    # Create bars
    width = 0.35
    bars1 = ax5.bar(x_pos - width/2, fid_scores, width, 
                   label='FID (lower better)', color='#FF6B6B', alpha=0.7)
    bars2 = ax5.bar(x_pos + width/2, inception_scores, width, 
                   label='Inception Score (higher better)', color='#4ECDC4', alpha=0.7)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.0f}', ha='center', va='bottom', fontsize=9)
    
    ax5.set_xlabel('Model Size', fontsize=11)
    ax5.set_ylabel('Score', fontsize=11)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(model_sizes, fontsize=10)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Add scaling law annotation
    ax5.text(1, ax5.get_ylim()[1] * 0.8, 'Bigger models →\nBetter quality',
            fontsize=10, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # ========== BATCH SIZE EFFECT ==========
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title('Batch Size Impact on Training', fontsize=14, fontweight='bold', pad=10)
    
    # Simulate training curves for different batch sizes
    epochs = np.linspace(0, 100, 100)
    
    # Different batch sizes
    batch_sizes = [256, 512, 1024, 2048]
    colors = ['#FF9999', '#FF6666', '#FF3333', '#FF0000']
    
    for i, (bs, color) in enumerate(zip(batch_sizes, colors)):
        # Simulate loss curve - larger batches converge faster but may plateau
        if bs == 256:
            loss = 2.5 * np.exp(-epochs/50) + 0.5 + 0.3 * np.sin(epochs/10)
        elif bs == 512:
            loss = 2.2 * np.exp(-epochs/40) + 0.4 + 0.2 * np.sin(epochs/12)
        elif bs == 1024:
            loss = 1.8 * np.exp(-epochs/30) + 0.3 + 0.1 * np.sin(epochs/15)
        else:  # 2048
            loss = 1.5 * np.exp(-epochs/20) + 0.25 + 0.05 * np.sin(epochs/20)
        
        ax6.plot(epochs, loss, color=color, linewidth=2, 
                label=f'Batch Size: {bs}', linestyle=['-', '--', '-.', ':'][i])
    
    ax6.set_xlabel('Training Epochs', fontsize=11)
    ax6.set_ylabel('Generator Loss', fontsize=11)
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Add annotations
    ax6.text(30, 2.0, 'Small batches:\nSlow convergence', fontsize=9, color='#FF9999',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    ax6.text(70, 0.8, 'Large batches:\nFast convergence', fontsize=9, color='#FF0000',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # ========== COMPARISON WITH OTHER MODELS ==========
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_title('BigGAN vs Other Models (2018)', fontsize=14, fontweight='bold', pad=10)
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 10)
    ax7.axis('off')
    
    # Model comparison
    models = [
        {'name': 'BigGAN', 'fid': 8.5, 'is': 85, 'size': 158, 'color': '#2E86AB'},
        {'name': 'StyleGAN', 'fid': 12.3, 'is': 72, 'size': 26, 'color': '#FF6B6B'},
        {'name': 'ProGAN', 'fid': 18.6, 'is': 58, 'size': 25, 'color': '#4ECDC4'},
        {'name': 'SN-GAN', 'fid': 21.7, 'is': 52, 'size': 45, 'color': '#F18F01'},
    ]
    
    for i, model in enumerate(models):
        y_pos = 9 - i * 2.2
        
        # Model card
        card = FancyBboxPatch((1, y_pos - 0.8), 8, 1.6,
                             boxstyle="round,pad=0.1",
                             facecolor=model['color'], alpha=0.2,
                             edgecolor=model['color'], linewidth=2)
        ax7.add_patch(card)
        
        # Model name
        ax7.text(1.5, y_pos, model['name'], fontsize=12, fontweight='bold',
                ha='left', va='center', color=model['color'])
        
        # Metrics
        metrics_text = f'FID: {model["fid"]:.1f}  |  IS: {model["is"]}  |  Size: {model["size"]}M params'
        ax7.text(5, y_pos, metrics_text, fontsize=10, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Performance bar
        perf_score = (100 - model['fid']) * model['is'] / 100  # Combined score
        bar_length = perf_score / 15  # Normalize
        
        bar = Rectangle((1.5, y_pos - 0.5), bar_length, 0.3,
                       facecolor=model['color'], alpha=0.6,
                       edgecolor=model['color'], linewidth=1)
        ax7.add_patch(bar)
        
        ax7.text(1.5 + bar_length + 0.2, y_pos - 0.35, f'Score: {perf_score:.0f}', 
                fontsize=9, ha='left', va='center')
    
    # Add comparison note
    comparison_note = """
    Key Advantages:
    • State-of-the-art FID and Inception Score (2018)
    • Class-conditional generation with high fidelity
    • Demonstrated importance of scaling (model & batch size)
    • Orthogonal regularization for stability
    """
    
    fig.text(0.02, 0.02, comparison_note, fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    # Add BigGAN equation for orthogonal regularization
    equation_text = r"$\mathcal{R}_\beta(W) = \beta \|W^\top W - I\|_F^2$"
    fig.text(0.85, 0.05, equation_text, fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('images/biggan_samples.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def create_biggan_class_conditional():
    """
    Create a visualization of class-conditional generation with BigGAN
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('BigGAN: Class-Conditional Generation and Interpolation', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Class interpolation visualization
    ax1 = axes[0, 0]
    ax1.set_title('Class Interpolation', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Two classes
    classes = ['Dog', 'Cat']
    
    # Start class (dog-like pattern)
    ax1.text(2, 8.5, classes[0], fontsize=11, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FF9999", alpha=0.5))
    
    # Dog pattern
    for i in range(15):
        x = 1 + np.random.uniform(0, 3)
        y = 6 + np.random.uniform(0, 2)
        size = np.random.uniform(0.2, 0.4)
        # Dog-like features (pointy ears, snout)
        if i < 5:
            # Ears
            ear = patches.RegularPolygon((x, y), 3, radius=size*0.8,
                                       orientation=np.random.uniform(0, 360),
                                       facecolor='brown', alpha=0.7)
            ax1.add_patch(ear)
        else:
            # Body/fur
            ear = patches.Ellipse((x, y), size, size*0.7,
                                 angle=np.random.uniform(0, 180),
                                 facecolor='brown', alpha=0.5)
            ax1.add_patch(ear)
    
    # End class (cat-like pattern)
    ax1.text(8, 8.5, classes[1], fontsize=11, fontweight='bold',
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#9999FF", alpha=0.5))
    
    # Cat pattern
    for i in range(15):
        x = 6 + np.random.uniform(0, 3)
        y = 6 + np.random.uniform(0, 2)
        size = np.random.uniform(0.15, 0.35)
        # Cat-like features (pointy ears, slender)
        if i < 5:
            # Ears
            ear = patches.RegularPolygon((x, y), 3, radius=size,
                                       orientation=np.random.uniform(0, 360),
                                       facecolor='gray', alpha=0.7)
            ax1.add_patch(ear)
        else:
            # Body
            ear = patches.Ellipse((x, y), size, size*0.6,
                                 angle=np.random.uniform(0, 180),
                                 facecolor='gray', alpha=0.5)
            ax1.add_patch(ear)
    
    # Interpolation samples
    interpolation_labels = ['α=0.2', 'α=0.5', 'α=0.8']
    for i, alpha in enumerate([0.2, 0.5, 0.8]):
        x_pos = 3.5 + i * 1.5
        y_pos = 3
        
        # Mixed pattern (blend of dog and cat)
        for j in range(10):
            x = x_pos - 0.5 + np.random.uniform(0, 1)
            y = y_pos - 0.5 + np.random.uniform(0, 1)
            size = np.random.uniform(0.1, 0.25)
            
            # Blend colors and shapes
            if np.random.random() < alpha:
                # More cat-like
                color = (0.7, 0.7, 0.9)  # Bluish gray
                shape = patches.Ellipse((x, y), size, size*0.6,
                                       angle=np.random.uniform(0, 180),
                                       facecolor=color, alpha=0.6)
            else:
                # More dog-like
                color = (0.9, 0.7, 0.5)  # Brownish
                shape = patches.Ellipse((x, y), size, size*0.8,
                                       angle=np.random.uniform(0, 180),
                                       facecolor=color, alpha=0.6)
            ax1.add_patch(shape)
        
        ax1.text(x_pos, y_pos + 1, interpolation_labels[i], fontsize=9,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Interpolation arrow
    ax1.arrow(2, 6.5, 6, 0, head_width=0.3, head_length=0.3,
             fc='green', ec='green', linewidth=2, alpha=0.7)
    ax1.text(5, 7.2, 'Interpolation →', fontsize=10, fontweight='bold',
            ha='center', va='center', color='green')
    
    # Shared embedding visualization
    ax2 = axes[0, 1]
    ax2.set_title('Shared Class Embedding', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Class labels
    class_labels = ['Airplane', 'Car', 'Bird', 'Cat', 'Dog', 'Ship']
    
    # Embedding space representation
    for i, label in enumerate(class_labels):
        angle = 2 * np.pi * i / len(class_labels)
        x = 5 + 3 * np.cos(angle)
        y = 5 + 3 * np.sin(angle)
        
        # Class point
        point = patches.Circle((x, y), 0.3, facecolor=plt.cm.tab10(i/len(class_labels)), 
                              alpha=0.8, edgecolor='black', linewidth=1)
        ax2.add_patch(point)
        
        # Label
        ax2.text(x + 0.4*np.cos(angle), y + 0.4*np.sin(angle), label,
                fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
    
    # Center - shared embedding
    center = patches.Circle((5, 5), 0.8, facecolor='purple', alpha=0.6,
                           edgecolor='black', linewidth=2)
    ax2.add_patch(center)
    ax2.text(5, 5, 'Shared\nEmbedding', fontsize=9, fontweight='bold',
            ha='center', va='center', color='white')
    
    # Lines connecting to center
    for i in range(len(class_labels)):
        angle = 2 * np.pi * i / len(class_labels)
        x_end = 5 + 3 * np.cos(angle)
        y_end = 5 + 3 * np.sin(angle)
        ax2.plot([5, x_end], [5, y_end], 'k-', alpha=0.3, linewidth=1)
    
    # Conditional generation examples
    ax3 = axes[0, 2]
    ax3.set_title('Conditional Generation Examples', fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    conditions = [
        {'class': 'Bird', 'features': ['Wings', 'Beak', 'Feathers'], 'y': 8},
        {'class': 'Car', 'features': ['Wheels', 'Windows', 'Lights'], 'y': 5},
        {'class': 'Flower', 'features': ['Petals', 'Stem', 'Leaves'], 'y': 2},
    ]
    
    for cond in conditions:
        # Class label
        ax3.text(1, cond['y'], cond['class'], fontsize=10, fontweight='bold',
                ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#4ECDC4", alpha=0.3))
        
        # Generated sample representation
        sample_box = Rectangle((3, cond['y'] - 1), 4, 1.5,
                              facecolor='white', alpha=0.9,
                              edgecolor='black', linewidth=2)
        ax3.add_patch(sample_box)
        
        # Add class-specific features
        if cond['class'] == 'Bird':
            # Bird-like features
            body = patches.Ellipse((5, cond['y'] - 0.3), 1.5, 0.8,
                                  facecolor='blue', alpha=0.6)
            wing = patches.Ellipse((4, cond['y'] - 0.5), 1, 0.5,
                                  angle=45, facecolor='lightblue', alpha=0.6)
            beak = patches.RegularPolygon((6, cond['y'] - 0.3), 3, radius=0.3,
                                         orientation=0, facecolor='orange', alpha=0.8)
            ax3.add_patch(body)
            ax3.add_patch(wing)
            ax3.add_patch(beak)
            
        elif cond['class'] == 'Car':
            # Car-like features
            body = Rectangle((4, cond['y'] - 0.5), 2, 0.8,
                            facecolor='red', alpha=0.7)
            wheel1 = patches.Circle((4.5, cond['y'] - 0.7), 0.2, facecolor='black')
            wheel2 = patches.Circle((5.5, cond['y'] - 0.7), 0.2, facecolor='black')
            window = Rectangle((4.5, cond['y'] - 0.2), 1, 0.4,
                              facecolor='lightblue', alpha=0.5)
            ax3.add_patch(body)
            ax3.add_patch(wheel1)
            ax3.add_patch(wheel2)
            ax3.add_patch(window)
            
        else:  # Flower
            # Flower-like features
            stem = Rectangle((4.9, cond['y'] - 0.8), 0.2, 1, facecolor='green', alpha=0.8)
            petals = []
            for angle in np.linspace(0, 2*np.pi, 6, endpoint=False):
                x_pos = 5 + 0.6 * np.cos(angle)
                y_pos = cond['y'] - 0.3 + 0.6 * np.sin(angle)
                petal = patches.Ellipse((x_pos, y_pos), 0.5, 0.3,
                                       angle=angle*180/np.pi,
                                       facecolor='pink', alpha=0.7)
                petals.append(petal)
            center_circle = patches.Circle((5, cond['y'] - 0.3), 0.3, facecolor='yellow', alpha=0.8)
            
            ax3.add_patch(stem)
            for petal in petals:
                ax3.add_patch(petal)
            ax3.add_patch(center_circle)
        
        # Features list
        features_text = '\n'.join([f'• {f}' for f in cond['features']])
        ax3.text(8, cond['y'], features_text, fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", alpha=0.7))
    
    # Performance metrics for conditional generation
    ax4 = axes[1, 0]
    ax4.set_title('Conditional vs Unconditional Performance', fontsize=12, fontweight='bold', pad=10)
    
    models = ['Unconditional', 'Conditional\n(BigGAN)']
    metrics = {
        'FID': [15.2, 8.5],
        'Precision': [0.68, 0.82],
        'Recall': [0.51, 0.63],
    }
    
    x = np.arange(len(models))
    width = 0.25
    multiplier = 0
    
    for metric, values in metrics.items():
        offset = width * multiplier
        bars = ax4.bar(x + offset, values, width, label=metric)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        multiplier += 1
    
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_xticks(x + width)
    ax4.set_xticklabels(models, fontsize=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add improvement annotation
    ax4.text(0.5, ax4.get_ylim()[1] * 0.8, 'Conditional\nbetter than\nunconditional',
            fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.7))
    
    # Training with class labels
    ax5 = axes[1, 1]
    ax5.set_title('Training with Class Information', fontsize=12, fontweight='bold', pad=10)
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    
    # Training process visualization
    training_steps = [
        {'step': '1. Input', 'desc': 'Noise z + Class c', 'y': 9},
        {'step': '2. Embed', 'desc': 'Shared embedding layer', 'y': 7},
        {'step': '3. Generate', 'desc': 'Conditional synthesis', 'y': 5},
        {'step': '4. Discriminate', 'desc': 'Class-aware discrimination', 'y': 3},
        {'step': '5. Update', 'desc': 'Gradient with class info', 'y': 1},
    ]
    
    for i, step in enumerate(training_steps):
        # Step box
        step_box = FancyBboxPatch((1, step['y'] - 0.4), 3, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor='#2E86AB', alpha=0.3 + i*0.1,
                                 edgecolor='#2E86AB', linewidth=1)
        ax5.add_patch(step_box)
        ax5.text(2.5, step['y'], step['step'], fontsize=9, fontweight='bold',
                ha='center', va='center')
        
        # Description
        ax5.text(6, step['y'], step['desc'], fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
        
        # Arrow to next step
        if i < len(training_steps) - 1:
            ax5.arrow(2.5, step['y'] - 0.5, 0, -1,
                     head_width=0.2, head_length=0.15, fc='black', ec='black',
                     alpha=0.5, linewidth=1)
    
    # Applications
    ax6 = axes[1, 2]
    ax6.set_title('Applications of Class-Conditional GANs', fontsize=12, fontweight='bold', pad=10)
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    
    applications = [
        {'app': 'Data Augmentation', 'desc': 'Generate rare class samples', 'y': 8.5},
        {'app': 'Controlled Generation', 'desc': 'Generate specific categories', 'y': 6.5},
        {'app': 'Style Transfer', 'desc': 'Transfer style between classes', 'y': 4.5},
        {'app': 'Few-shot Learning', 'desc': 'Learn from few examples per class', 'y': 2.5},
    ]
    
    for i, app in enumerate(applications):
        # Application icon
        if i == 0:
            icon = patches.Circle((2, app['y']), 0.4,
                                 facecolor=plt.cm.tab10(i/len(applications)),
                                 alpha=0.7, edgecolor='black', linewidth=1)
        elif i == 1:
            icon = Rectangle((1.6, app['y'] - 0.4), 0.8, 0.8,
                            facecolor=plt.cm.tab10(i/len(applications)),
                            alpha=0.7, edgecolor='black', linewidth=1)
        elif i == 2:
            icon = patches.RegularPolygon((2, app['y']), 6, radius=0.4,
                                         facecolor=plt.cm.tab10(i/len(applications)),
                                         alpha=0.7, edgecolor='black', linewidth=1)
        else:
            icon = patches.Ellipse((2, app['y']), 0.6, 0.4,
                                  angle=45,
                                  facecolor=plt.cm.tab10(i/len(applications)),
                                  alpha=0.7, edgecolor='black', linewidth=1)
        ax6.add_patch(icon)
        
        # Application name
        ax6.text(3, app['y'], app['app'], fontsize=10, fontweight='bold',
                ha='left', va='center')
        
        # Description
        ax6.text(6, app['y'], app['desc'], fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('images/biggan_class_conditional.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Generating BigGAN visualizations...")
    
    # Generate main BigGAN samples visualization
    fig1 = create_biggan_samples_visualization()
    print("✓ Generated 'images/biggan_samples.png'")
    
    # Generate class-conditional visualization
    create_biggan_class_conditional()
    print("✓ Generated 'images/biggan_class_conditional.png'")
    
    print("\nAll BigGAN visualizations have been generated successfully!")
    print("The visualizations include:")
    print("1. BigGAN architecture, innovations, and sample quality")
    print("2. Class-conditional generation and interpolation")