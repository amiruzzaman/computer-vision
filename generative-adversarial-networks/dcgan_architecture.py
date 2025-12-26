import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, ConnectionPatch
import numpy as np
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def create_dcgan_architecture_diagram():
    """
    Create a detailed DCGAN architecture diagram showing both Generator and Discriminator
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 10))
    fig.suptitle('DCGAN Architecture: Deep Convolutional Generative Adversarial Networks', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Colors
    gen_color = '#4CAF50'  # Green for generator
    disc_color = '#2196F3'  # Blue for discriminator
    layer_color = '#FF9800'  # Orange for layers
    
    # ========== GENERATOR ==========
    ax_gen = axes[0]
    ax_gen.set_title('Generator Architecture', fontsize=14, fontweight='bold', pad=15)
    ax_gen.set_xlim(-1, 12)
    ax_gen.set_ylim(0, 16)
    ax_gen.axis('off')
    
    # Add Generator title background
    ax_gen.add_patch(Rectangle((-0.5, 15.2), 12.5, 0.8, 
                              facecolor=gen_color, alpha=0.3, edgecolor=gen_color, linewidth=2))
    
    # Noise input
    noise_rect = FancyBboxPatch((0.5, 13.5), 3, 1, 
                                boxstyle="round,pad=0.1", 
                                facecolor=gen_color, alpha=0.5, edgecolor='black', linewidth=2)
    ax_gen.add_patch(noise_rect)
    ax_gen.text(2, 14, 'Noise Vector\n(100-dim)', ha='center', va='center', 
                fontsize=10, fontweight='bold')
    
    # Project and reshape
    proj_rect = FancyBboxPatch((5, 13.5), 2, 1, 
                               boxstyle="round,pad=0.1", 
                               facecolor=layer_color, alpha=0.7, edgecolor='black', linewidth=1)
    ax_gen.add_patch(proj_rect)
    ax_gen.text(6, 14, 'Dense Layer\nProject & Reshape', ha='center', va='center', 
                fontsize=9, fontweight='bold')
    
    # ConvTranspose layers
    conv_layers = [
        {'name': 'ConvT 4x4\ns=1, p=0\n1024 ch', 'channels': 1024, 'size': 4, 'y': 11},
        {'name': 'ConvT 4x4\ns=2, p=1\n512 ch', 'channels': 512, 'size': 8, 'y': 9},
        {'name': 'ConvT 4x4\ns=2, p=1\n256 ch', 'channels': 256, 'size': 16, 'y': 7},
        {'name': 'ConvT 4x4\ns=2, p=1\n128 ch', 'channels': 128, 'size': 32, 'y': 5},
        {'name': 'ConvT 4x4\ns=2, p=1\n3 ch', 'channels': 3, 'size': 64, 'y': 3},
    ]
    
    # Draw convolutional layers as feature maps
    for i, layer in enumerate(conv_layers):
        # Feature map representation
        size = layer['size']
        channels = layer['channels']
        y_pos = layer['y']
        
        # Draw multiple feature maps (simplified)
        for j in range(min(3, channels // 100)):  # Show up to 3 feature maps
            x_offset = 1 + j * 0.5
            feature_map = FancyBboxPatch((x_offset, y_pos), size/10, size/10,
                                        boxstyle="round,pad=0.02",
                                        facecolor=gen_color, alpha=0.3 + j*0.2,
                                        edgecolor='black', linewidth=0.5)
            ax_gen.add_patch(feature_map)
        
        # Layer description
        layer_rect = FancyBboxPatch((5, y_pos - 0.5), 2, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=layer_color, alpha=0.7, edgecolor='black', linewidth=1)
        ax_gen.add_patch(layer_rect)
        ax_gen.text(6, y_pos, layer['name'], ha='center', va='center',
                   fontsize=8, fontweight='bold')
    
    # Output image
    output_rect = FancyBboxPatch((1, 1), 8, 1,
                                boxstyle="round,pad=0.1",
                                facecolor=gen_color, alpha=0.5, edgecolor='black', linewidth=2)
    ax_gen.add_patch(output_rect)
    ax_gen.text(5, 1.5, 'Generated Image\n64×64×3', ha='center', va='center',
               fontsize=10, fontweight='bold')
    
    # Arrows
    arrow_y_positions = [13, 11, 9, 7, 5, 3]
    for i in range(len(arrow_y_positions) - 1):
        y_start = arrow_y_positions[i] - 0.5
        y_end = arrow_y_positions[i + 1] + 0.5
        ax_gen.arrow(6, y_start, 0, y_end - y_start - 0.8,
                    head_width=0.3, head_length=0.2, fc='black', ec='black',
                    length_includes_head=True, alpha=0.7)
    
    # Add activation functions and normalization
    act_positions = [10.5, 8.5, 6.5, 4.5]
    for i, y_pos in enumerate(act_positions):
        # BatchNorm
        bn_circle = plt.Circle((3.5, y_pos), 0.3, facecolor='yellow', edgecolor='black', linewidth=1)
        ax_gen.add_patch(bn_circle)
        ax_gen.text(3.5, y_pos, 'BN', ha='center', va='center', fontsize=7, fontweight='bold')
        
        # ReLU
        relu_rect = FancyBboxPatch((4, y_pos - 0.3), 0.6, 0.6,
                                  boxstyle="round,pad=0.05",
                                  facecolor='red', alpha=0.3, edgecolor='black', linewidth=1)
        ax_gen.add_patch(relu_rect)
        ax_gen.text(4.3, y_pos, 'ReLU', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Tanh for output
    tanh_rect = FancyBboxPatch((4, 2.2), 0.6, 0.6,
                              boxstyle="round,pad=0.05",
                              facecolor='purple', alpha=0.3, edgecolor='black', linewidth=1)
    ax_gen.add_patch(tanh_rect)
    ax_gen.text(4.3, 2.5, 'Tanh', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # ========== DISCRIMINATOR ==========
    ax_disc = axes[1]
    ax_disc.set_title('Discriminator Architecture', fontsize=14, fontweight='bold', pad=15)
    ax_disc.set_xlim(-1, 12)
    ax_disc.set_ylim(0, 16)
    ax_disc.axis('off')
    
    # Add Discriminator title background
    ax_disc.add_patch(Rectangle((-0.5, 15.2), 12.5, 0.8, 
                               facecolor=disc_color, alpha=0.3, edgecolor=disc_color, linewidth=2))
    
    # Input image
    input_rect = FancyBboxPatch((0.5, 13.5), 3, 1,
                               boxstyle="round,pad=0.1",
                               facecolor=disc_color, alpha=0.5, edgecolor='black', linewidth=2)
    ax_disc.add_patch(input_rect)
    ax_disc.text(2, 14, 'Input Image\n64×64×3', ha='center', va='center',
                fontsize=10, fontweight='bold')
    
    # Conv layers
    conv_layers_disc = [
        {'name': 'Conv 4x4\ns=2, p=1\n128 ch', 'channels': 128, 'size': 32, 'y': 11},
        {'name': 'Conv 4x4\ns=2, p=1\n256 ch', 'channels': 256, 'size': 16, 'y': 9},
        {'name': 'Conv 4x4\ns=2, p=1\n512 ch', 'channels': 512, 'size': 8, 'y': 7},
        {'name': 'Conv 4x4\ns=2, p=1\n1024 ch', 'channels': 1024, 'size': 4, 'y': 5},
    ]
    
    # Draw convolutional layers
    for i, layer in enumerate(conv_layers_disc):
        size = layer['size']
        channels = layer['channels']
        y_pos = layer['y']
        
        # Feature map representation
        for j in range(min(3, channels // 100)):
            x_offset = 1 + j * 0.5
            feature_map = FancyBboxPatch((x_offset, y_pos), size/10, size/10,
                                        boxstyle="round,pad=0.02",
                                        facecolor=disc_color, alpha=0.3 + j*0.2,
                                        edgecolor='black', linewidth=0.5)
            ax_disc.add_patch(feature_map)
        
        # Layer description
        layer_rect = FancyBboxPatch((5, y_pos - 0.5), 2, 1,
                                   boxstyle="round,pad=0.1",
                                   facecolor=layer_color, alpha=0.7, edgecolor='black', linewidth=1)
        ax_disc.add_patch(layer_rect)
        ax_disc.text(6, y_pos, layer['name'], ha='center', va='center',
                    fontsize=8, fontweight='bold')
    
    # Flatten and dense
    flatten_rect = FancyBboxPatch((5, 3.5), 2, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor=layer_color, alpha=0.7, edgecolor='black', linewidth=1)
    ax_disc.add_patch(flatten_rect)
    ax_disc.text(6, 4, 'Flatten', ha='center', va='center',
                fontsize=9, fontweight='bold')
    
    # Output layer
    output_disc_rect = FancyBboxPatch((5, 1.5), 2, 1,
                                     boxstyle="round,pad=0.1",
                                     facecolor=disc_color, alpha=0.5, edgecolor='black', linewidth=2)
    ax_disc.add_patch(output_disc_rect)
    ax_disc.text(6, 2, 'Dense Layer\nReal/Fake Probability', ha='center', va='center',
                fontsize=9, fontweight='bold')
    
    # Arrows for discriminator
    arrow_y_positions_disc = [13, 11, 9, 7, 5, 3]
    for i in range(len(arrow_y_positions_disc) - 1):
        y_start = arrow_y_positions_disc[i] - 0.5
        y_end = arrow_y_positions_disc[i + 1] + 0.5
        ax_disc.arrow(6, y_start, 0, y_end - y_start - 0.8,
                     head_width=0.3, head_length=0.2, fc='black', ec='black',
                     length_includes_head=True, alpha=0.7)
    
    # Add activation functions and normalization for discriminator
    act_positions_disc = [10.5, 8.5, 6.5, 4.5]
    for i, y_pos in enumerate(act_positions_disc):
        # BatchNorm (not in first layer)
        if i > 0:
            bn_circle = plt.Circle((3.5, y_pos), 0.3, facecolor='yellow', edgecolor='black', linewidth=1)
            ax_disc.add_patch(bn_circle)
            ax_disc.text(3.5, y_pos, 'BN', ha='center', va='center', fontsize=7, fontweight='bold')
        
        # LeakyReLU
        lrelu_rect = FancyBboxPatch((4, y_pos - 0.3), 0.6, 0.6,
                                   boxstyle="round,pad=0.05",
                                   facecolor='orange', alpha=0.5, edgecolor='black', linewidth=1)
        ax_disc.add_patch(lrelu_rect)
        ax_disc.text(4.3, y_pos, 'LReLU', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Sigmoid for output
    sigmoid_rect = FancyBboxPatch((4, 2.2), 0.6, 0.6,
                                 boxstyle="round,pad=0.05",
                                 facecolor='green', alpha=0.3, edgecolor='black', linewidth=1)
    ax_disc.add_patch(sigmoid_rect)
    ax_disc.text(4.3, 2.5, 'Sigmoid', ha='center', va='center', fontsize=7, fontweight='bold')
    
    # Add DCGAN guidelines
    guidelines_text = """
    DCGAN Guidelines (Radford et al., 2015):
    
    1. Replace pooling with strided convolutions
    2. Use BatchNorm in both generator and discriminator
    3. Remove fully connected hidden layers
    4. Use ReLU in generator, LeakyReLU in discriminator
    5. Use Tanh in generator output, Sigmoid in discriminator output
    """
    
    fig.text(0.02, 0.02, guidelines_text, fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/dcgan_architecture.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def create_dcgan_progressive_growing():
    """
    Create a visualization of progressive growing in DCGAN
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle('DCGAN: Progressive Growing for High-Resolution Generation', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Left: Progressive growing process
    ax1 = axes[0]
    ax1.set_title('Progressive Growing Strategy', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Progressive growing stages
    stages = [
        {'size': 4, 'res': '4×4', 'y': 8, 'color': '#FF6B6B'},
        {'size': 8, 'res': '8×8', 'y': 6, 'color': '#4ECDC4'},
        {'size': 16, 'res': '16×16', 'y': 4, 'color': '#45B7D1'},
        {'size': 32, 'res': '32×32', 'y': 2, 'color': '#96CEB4'},
        {'size': 64, 'res': '64×64', 'y': 0.5, 'color': '#FECA57'},
    ]
    
    for i, stage in enumerate(stages):
        # Draw feature map grid
        size = stage['size']
        for row in range(min(3, size)):  # Show simplified grid
            for col in range(min(3, size)):
                x = 2 + col * 0.3
                y = stage['y'] + row * 0.3
                cell = Rectangle((x, y), 0.25, 0.25,
                                facecolor=stage['color'], alpha=0.6,
                                edgecolor='black', linewidth=0.5)
                ax1.add_patch(cell)
        
        # Label
        ax1.text(5, stage['y'] + 0.15, f'Stage {i+1}: {stage["res"]}', 
                fontsize=10, fontweight='bold', ha='left', va='center')
        
        # Add layer description
        if i < len(stages) - 1:
            ax1.text(7, stage['y'] + 0.15, 
                    f'ConvT/Conv Layer\n+ Upsample/Downsample',
                    fontsize=8, ha='left', va='center',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # Arrows between stages
    for i in range(len(stages) - 1):
        y_start = stages[i]['y'] + 0.8
        y_end = stages[i + 1]['y'] + 0.8
        ax1.arrow(4, y_start, 0, y_end - y_start - 0.3,
                 head_width=0.2, head_length=0.15, fc='black', ec='black',
                 length_includes_head=True, alpha=0.7)
    
    # Right: Training stability techniques
    ax2 = axes[1]
    ax2.set_title('Training Stabilization Techniques', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Techniques visualization
    techniques = [
        {'name': 'Batch Normalization', 'desc': 'Normalize activations\nReduce internal covariate shift', 
         'y': 8.5, 'symbol': '⚖️'},
        {'name': 'Strided Convolutions', 'desc': 'Replace pooling layers\nLearn up/down sampling', 
         'y': 6.5, 'symbol': '↕️'},
        {'name': 'Leaky ReLU', 'desc': 'Prevent dying neurons\nBetter gradient flow in D', 
         'y': 4.5, 'symbol': '⚡'},
        {'name': 'No Fully Connected', 'desc': 'Remove FC layers\nAll convolutional network', 
         'y': 2.5, 'symbol': '🔄'},
        {'name': 'Tanh/Sigmoid Output', 'desc': 'Tanh for G output\nSigmoid for D output', 
         'y': 0.5, 'symbol': '📊'},
    ]
    
    for i, tech in enumerate(techniques):
        # Symbol
        ax2.text(1, tech['y'], tech['symbol'], fontsize=20, ha='center', va='center')
        
        # Name
        ax2.text(3, tech['y'], tech['name'], fontsize=11, fontweight='bold',
                ha='left', va='center')
        
        # Description
        ax2.text(7, tech['y'], tech['desc'], fontsize=9,
                ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F4F8", alpha=0.7))
    
    # Connection line
    for i in range(len(techniques) - 1):
        ax2.plot([1, 1], [techniques[i]['y'] - 0.5, techniques[i + 1]['y'] + 0.5],
                'k-', alpha=0.3, linewidth=2)
    
    # Add note
    note_text = """
    Key Innovations of DCGAN:
    • First stable CNN-based GAN architecture
    • Learned meaningful latent representations
    • Enabled vector arithmetic in latent space
    • Foundation for all modern GAN architectures
    """
    
    fig.text(0.02, 0.02, note_text, fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/dcgan_progressive.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def create_dcgan_implementation_example():
    """
    Create a code and implementation example for DCGAN
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DCGAN Implementation and Results', fontsize=18, fontweight='bold', y=0.98)
    
    # Sample architecture visualization
    ax1 = axes[0, 0]
    ax1.set_title('Generator Architecture Details', fontsize=12, fontweight='bold')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Generator layers in detail
    gen_layers = [
        {'name': 'Input: z ~ N(0,1)', 'shape': '(100,)', 'y': 9},
        {'name': 'Dense → Reshape', 'shape': '(4, 4, 1024)', 'y': 8},
        {'name': 'Conv2DTranspose', 'shape': '(8, 8, 512)', 'y': 7},
        {'name': 'BatchNorm + ReLU', 'shape': '(8, 8, 512)', 'y': 6},
        {'name': 'Conv2DTranspose', 'shape': '(16, 16, 256)', 'y': 5},
        {'name': 'BatchNorm + ReLU', 'shape': '(16, 16, 256)', 'y': 4},
        {'name': 'Conv2DTranspose', 'shape': '(32, 32, 128)', 'y': 3},
        {'name': 'BatchNorm + ReLU', 'shape': '(32, 32, 128)', 'y': 2},
        {'name': 'Conv2DTranspose', 'shape': '(64, 64, 3)', 'y': 1},
        {'name': 'Tanh Activation', 'shape': '(64, 64, 3)', 'y': 0},
    ]
    
    for i, layer in enumerate(gen_layers):
        # Layer box
        rect = Rectangle((1, layer['y']), 5, 0.8,
                        facecolor='#4CAF50', alpha=0.3 + i*0.05,
                        edgecolor='black', linewidth=1)
        ax1.add_patch(rect)
        
        # Layer info
        ax1.text(1.1, layer['y'] + 0.4, layer['name'], 
                fontsize=9, ha='left', va='center', fontweight='bold')
        ax1.text(5.5, layer['y'] + 0.4, layer['shape'], 
                fontsize=8, ha='right', va='center', style='italic')
        
        # Arrow
        if i < len(gen_layers) - 1:
            ax1.arrow(3.5, layer['y'] - 0.1, 0, -0.6,
                     head_width=0.2, head_length=0.1, fc='black', ec='black', alpha=0.5)
    
    # Discriminator architecture
    ax2 = axes[0, 1]
    ax2.set_title('Discriminator Architecture Details', fontsize=12, fontweight='bold')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    disc_layers = [
        {'name': 'Input: Image', 'shape': '(64, 64, 3)', 'y': 9},
        {'name': 'Conv2D', 'shape': '(32, 32, 128)', 'y': 8},
        {'name': 'LeakyReLU', 'shape': '(32, 32, 128)', 'y': 7},
        {'name': 'Conv2D', 'shape': '(16, 16, 256)', 'y': 6},
        {'name': 'BatchNorm + LeakyReLU', 'shape': '(16, 16, 256)', 'y': 5},
        {'name': 'Conv2D', 'shape': '(8, 8, 512)', 'y': 4},
        {'name': 'BatchNorm + LeakyReLU', 'shape': '(8, 8, 512)', 'y': 3},
        {'name': 'Conv2D', 'shape': '(4, 4, 1024)', 'y': 2},
        {'name': 'BatchNorm + LeakyReLU', 'shape': '(4, 4, 1024)', 'y': 1},
        {'name': 'Flatten → Dense → Sigmoid', 'shape': '(1,)', 'y': 0},
    ]
    
    for i, layer in enumerate(disc_layers):
        rect = Rectangle((1, layer['y']), 5, 0.8,
                        facecolor='#2196F3', alpha=0.3 + i*0.05,
                        edgecolor='black', linewidth=1)
        ax2.add_patch(rect)
        
        ax2.text(1.1, layer['y'] + 0.4, layer['name'], 
                fontsize=9, ha='left', va='center', fontweight='bold')
        ax2.text(5.5, layer['y'] + 0.4, layer['shape'], 
                fontsize=8, ha='right', va='center', style='italic')
        
        if i < len(disc_layers) - 1:
            ax2.arrow(3.5, layer['y'] - 0.1, 0, -0.6,
                     head_width=0.2, head_length=0.1, fc='black', ec='black', alpha=0.5)
    
    # Training process
    ax3 = axes[1, 0]
    ax3.set_title('Training Process and Loss Curves', fontsize=12, fontweight='bold')
    
    # Simulate training curves
    epochs = np.linspace(0, 100, 100)
    
    # Generator loss
    g_loss = 2.0 * np.exp(-epochs/30) + 0.1 * np.sin(epochs/5) + 0.5 * np.exp(-epochs/50)
    # Discriminator loss
    d_loss_real = 0.1 * np.exp(-epochs/20) + 0.02 * np.sin(epochs/3) + 0.15
    d_loss_fake = 0.15 * np.exp(-epochs/25) + 0.03 * np.sin(epochs/4 + 1) + 0.2
    
    ax3.plot(epochs, g_loss, 'g-', linewidth=2, label='Generator Loss')
    ax3.plot(epochs, d_loss_real, 'b-', linewidth=2, label='Discriminator Loss (Real)')
    ax3.plot(epochs, d_loss_fake, 'r-', linewidth=2, label='Discriminator Loss (Fake)')
    
    ax3.set_xlabel('Epochs', fontsize=10)
    ax3.set_ylabel('Loss', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_ylim(0, 2.5)
    
    # Add equilibrium line
    ax3.axhline(y=0.693, color='k', linestyle='--', alpha=0.5, label='Theoretical Equilibrium')
    ax3.text(70, 0.75, 'log(2) ≈ 0.693', fontsize=8, style='italic')
    
    # Generated samples visualization
    ax4 = axes[1, 1]
    ax4.set_title('Generated Samples at Different Epochs', fontsize=12, fontweight='bold')
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Simulate sample progression
    epochs_samples = [10, 30, 50, 80]
    colors = ['#FFCCCC', '#FF9999', '#FF6666', '#FF3333']
    
    for i, (epoch, color) in enumerate(zip(epochs_samples, colors)):
        y_pos = 8 - i * 2
        
        # Sample "image" representation
        img_grid = np.zeros((4, 4, 3))
        for row in range(4):
            for col in range(4):
                # Simulate improving quality with epochs
                intensity = min(1.0, 0.3 + epoch/100 + np.random.rand()*0.3)
                img_grid[row, col] = [intensity, intensity*0.9, intensity*0.8]
        
        ax4.imshow(img_grid, extent=[1, 3, y_pos, y_pos + 1.5], aspect='auto')
        
        # Epoch label
        ax4.text(3.5, y_pos + 0.75, f'Epoch {epoch}', fontsize=10, fontweight='bold')
        
        # Quality description
        quality = ['Noisy', 'Blurry Shapes', 'Recognizable', 'High Quality'][i]
        ax4.text(5, y_pos + 0.75, f'→ {quality}', fontsize=9, style='italic')
    
    # Add implementation note
    code_snippet = """
    # Key PyTorch Implementation Snippets:
    
    # Generator
    self.main = nn.Sequential(
        nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
        nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
        nn.BatchNorm2d(256),
        nn.ReLU(True),
        ...  # More layers
        nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    
    # Discriminator  
    self.main = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(64, 128, 4, 2, 1, bias=False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, inplace=True),
        ...  # More layers
        nn.Conv2d(512, 1, 4, 1, 0, bias=False),
        nn.Sigmoid()
    )
    """
    
    fig.text(0.02, 0.02, code_snippet, fontsize=8, fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f5f5f5", alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('images/dcgan_implementation.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Generating DCGAN architecture diagrams...")
    
    # Generate main architecture diagram
    fig1 = create_dcgan_architecture_diagram()
    print("✓ Generated 'images/dcgan_architecture.png'")
    
    # Generate progressive growing diagram
    create_dcgan_progressive_growing()
    print("✓ Generated 'images/dcgan_progressive.png'")
    
    # Generate implementation example
    create_dcgan_implementation_example()
    print("✓ Generated 'images/dcgan_implementation.png'")
    
    print("\nAll DCGAN diagrams have been generated successfully!")
    print("The visualizations include:")
    print("1. Detailed architecture of Generator and Discriminator")
    print("2. Progressive growing strategy for high-resolution generation")
    print("3. Implementation details and training results")
#`images/dcgan_architecture.png'