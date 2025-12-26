import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon, Ellipse
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def create_text_to_image_visualization():
    """
    Create a comprehensive visualization of text-to-image synthesis models
    """
    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Text-to-Image Synthesis: From Text Descriptions to Photorealistic Images', 
                 fontsize=22, fontweight='bold', y=0.98)
    
    # Create a grid for layout
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # ========== OVERVIEW OF MODELS ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('Evolution of Text-to-Image Models', fontsize=14, fontweight='bold', pad=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Timeline of models
    models = [
        {'year': '2015-2016', 'name': 'Early GANs', 'desc': 'Simple text conditioning\nLimited quality & diversity', 'color': '#FF9999'},
        {'year': '2018-2019', 'name': 'StackGAN, AttnGAN', 'desc': 'Multi-stage generation\nAttention mechanisms', 'color': '#FFCC99'},
        {'year': '2020-2021', 'name': 'DALL-E, CogView', 'desc': 'Transformer-based\nDiscrete VAE + GPT', 'color': '#CCFFCC'},
        {'year': '2022', 'name': 'DALL-E 2, Imagen', 'desc': 'Diffusion models\nCLIP guidance', 'color': '#99CCFF'},
        {'year': '2022-2023', 'name': 'Stable Diffusion', 'desc': 'Latent diffusion\nOpen source', 'color': '#CC99FF'},
    ]
    
    for i, model in enumerate(models):
        y_pos = 9 - i * 1.8
        
        # Year
        ax1.text(1, y_pos, model['year'], fontsize=10, fontweight='bold',
                ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgray', alpha=0.8))
        
        # Model name
        name_box = FancyBboxPatch((2.5, y_pos - 0.3), 2.5, 0.6,
                                 boxstyle="round,pad=0.1",
                                 facecolor=model['color'], alpha=0.7,
                                 edgecolor='black', linewidth=1)
        ax1.add_patch(name_box)
        ax1.text(3.75, y_pos, model['name'], fontsize=10, fontweight='bold',
                ha='center', va='center')
        
        # Description
        desc_box = FancyBboxPatch((5.5, y_pos - 0.5), 4, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightgray', alpha=0.6,
                                 edgecolor='gray', linewidth=0.5)
        ax1.add_patch(desc_box)
        ax1.text(7.5, y_pos, model['desc'], fontsize=8, ha='center', va='center')
        
        # Timeline connector
        if i < len(models) - 1:
            ax1.plot([3.75, 3.75], [y_pos - 0.5, y_pos - 1.8 + 0.5], 
                    'k-', alpha=0.3, linewidth=2)
    
    # ========== ARCHITECTURE COMPARISON ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Architecture Comparison', fontsize=14, fontweight='bold', pad=10)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    architectures = [
        {'name': 'GAN-based', 'y': 8.5, 'components': ['Text Encoder', 'Conditional GAN', 'Upsampling'], 'color': '#FF6B6B'},
        {'name': 'Transformer-based', 'y': 6, 'components': ['Text Tokenizer', 'VQ-VAE', 'GPT-like Decoder'], 'color': '#4ECDC4'},
        {'name': 'Diffusion-based', 'y': 3.5, 'components': ['Text Encoder', 'Diffusion Model', 'Decoder'], 'color': '#45B7D1'},
    ]
    
    for arch in architectures:
        # Architecture name
        ax2.text(1, arch['y'], arch['name'], fontsize=11, fontweight='bold',
                ha='left', va='center', color=arch['color'])
        
        # Components
        for i, component in enumerate(arch['components']):
            x_pos = 3 + i * 2
            component_box = FancyBboxPatch((x_pos - 0.8, arch['y'] - 0.4), 1.6, 0.8,
                                          boxstyle="round,pad=0.1",
                                          facecolor=arch['color'], alpha=0.3,
                                          edgecolor=arch['color'], linewidth=1)
            ax2.add_patch(component_box)
            ax2.text(x_pos, arch['y'], component, fontsize=8, ha='center', va='center')
            
            # Arrows between components
            if i < len(arch['components']) - 1:
                ax2.arrow(x_pos + 0.8, arch['y'], 0.4, 0, 
                         head_width=0.1, head_length=0.1, fc='black', ec='black', alpha=0.5)
    
    # ========== EXAMPLE PROMPT VISUALIZATION ==========
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title('Example: "An astronaut riding a horse in photorealistic style"', 
                  fontsize=14, fontweight='bold', pad=10)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    ax3.axis('off')
    
    # Text prompt
    prompt_text = '"An astronaut riding a horse\nin photorealistic style"'
    prompt_box = FancyBboxPatch((2, 8.5), 6, 1.2,
                               boxstyle="round,pad=0.1",
                               facecolor='lightblue', alpha=0.3,
                               edgecolor='blue', linewidth=2)
    ax3.add_patch(prompt_box)
    ax3.text(5, 9, prompt_text, fontsize=12, fontweight='bold',
            ha='center', va='center', style='italic')
    
    # Generated image representation
    image_box = FancyBboxPatch((2, 4), 6, 4,
                              facecolor='white', alpha=0.9,
                              edgecolor='black', linewidth=3)
    ax3.add_patch(image_box)
    
    # Draw the scene
    # Background - space
    for i in range(20):
        x = 2 + np.random.uniform(0, 6)
        y = 4 + np.random.uniform(0, 4)
        size = np.random.uniform(0.02, 0.05)
        star = Circle((x, y), size, facecolor='white', alpha=np.random.uniform(0.5, 1))
        ax3.add_patch(star)
    
    # Planet
    planet = Circle((7, 5.5), 1.2, facecolor='#4A90E2', alpha=0.8)
    ax3.add_patch(planet)
    
    # Horse body
    horse_body = Ellipse((4.5, 6), 2, 1, angle=0, facecolor='#8B4513', alpha=0.9)
    ax3.add_patch(horse_body)
    
    # Horse legs
    for i in range(4):
        leg_x = 3.5 + i * 0.5
        leg = Rectangle((leg_x, 5), 0.2, 1, facecolor='#8B4513', alpha=0.9)
        ax3.add_patch(leg)
    
    # Horse head
    horse_head = Ellipse((5.8, 6.8), 0.8, 0.6, angle=30, facecolor='#8B4513', alpha=0.9)
    ax3.add_patch(horse_head)
    
    # Astronaut
    astronaut_body = Ellipse((4.5, 7.2), 0.8, 1.2, angle=0, facecolor='white', alpha=0.9)
    ax3.add_patch(astronaut_body)
    
    # Astronaut helmet
    helmet = Circle((4.5, 7.8), 0.5, facecolor='lightgray', alpha=0.8, edgecolor='gray', linewidth=1)
    ax3.add_patch(helmet)
    
    # Visor
    visor = Ellipse((4.5, 7.8), 0.8, 0.4, angle=0, facecolor='#1E90FF', alpha=0.5)
    ax3.add_patch(visor)
    
    # Add labels for components
    components = [
        ('Space background', 8, 5),
        ('Planet', 7, 4.2),
        ('Horse', 3.5, 5.5),
        ('Astronaut', 4.5, 8.5),
    ]
    
    for text, x, y in components:
        ax3.text(x, y, text, fontsize=8, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
    
    # ========== DIFFUSION PROCESS VISUALIZATION ==========
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.set_title('Diffusion Process: From Noise to Image', fontsize=14, fontweight='bold', pad=10)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Diffusion steps
    steps = [
        {'name': 'Pure Noise', 'noise': 1.0, 'y': 8, 'color': '#666666'},
        {'name': 'Step 1/50', 'noise': 0.8, 'y': 6.5, 'color': '#888888'},
        {'name': 'Step 25/50', 'noise': 0.4, 'y': 5, 'color': '#AAAAAA'},
        {'name': 'Step 49/50', 'noise': 0.1, 'y': 3.5, 'color': '#CCCCCC'},
        {'name': 'Final Image', 'noise': 0.0, 'y': 2, 'color': '#FFFFFF'},
    ]
    
    for step in steps:
        # Step label
        ax4.text(1, step['y'], step['name'], fontsize=9, fontweight='bold',
                ha='left', va='center')
        
        # Image representation
        img_box = Rectangle((3, step['y'] - 0.8), 6, 1.6,
                           facecolor='white', alpha=0.9,
                           edgecolor='black', linewidth=1)
        ax4.add_patch(img_box)
        
        # Add noise/pattern
        noise_level = step['noise']
        n_points = int(100 * (1 - noise_level) + 20)  # More points for less noise
        
        if noise_level > 0.5:
            # Mostly random noise
            for i in range(n_points):
                x = 3 + np.random.uniform(0, 6)
                y = step['y'] - 0.8 + np.random.uniform(0, 1.6)
                size = np.random.uniform(0.02, 0.08)
                intensity = np.random.uniform(0.3, 0.7)
                point = Circle((x, y), size, facecolor=(intensity, intensity, intensity))
                ax4.add_patch(point)
        elif noise_level > 0.1:
            # Some structure emerging
            # Create some simple shapes
            for i in range(min(n_points, 30)):
                x = 3 + np.random.uniform(0, 6)
                y = step['y'] - 0.8 + np.random.uniform(0, 1.6)
                size = np.random.uniform(0.05, 0.15)
                
                if i % 3 == 0:
                    shape = Circle((x, y), size, facecolor=np.random.rand(3), alpha=0.6)
                elif i % 3 == 1:
                    shape = Rectangle((x-size/2, y-size/2), size, size,
                                     facecolor=np.random.rand(3), alpha=0.6)
                else:
                    shape = Ellipse((x, y), size, size*0.7,
                                   angle=np.random.uniform(0, 180),
                                   facecolor=np.random.rand(3), alpha=0.6)
                ax4.add_patch(shape)
        else:
            # Clear image - draw simple scene
            # Sky
            sky = Rectangle((3, step['y'] - 0.8), 6, 1.6,
                           facecolor=(0.2, 0.4, 0.8), alpha=0.7)
            ax4.add_patch(sky)
            
            # Sun
            sun = Circle((8, step['y'] + 0.5), 0.3, facecolor='yellow')
            ax4.add_patch(sun)
            
            # Simple landscape
            ground = Rectangle((3, step['y'] - 0.8), 6, 0.6,
                              facecolor=(0.3, 0.6, 0.3), alpha=0.8)
            ax4.add_patch(ground)
            
            # Tree
            tree_trunk = Rectangle((4.5, step['y'] - 0.2), 0.2, 0.6,
                                  facecolor='#8B4513')
            tree_top = Circle((4.6, step['y'] + 0.3), 0.4, facecolor='#228B22')
            ax4.add_patch(tree_trunk)
            ax4.add_patch(tree_top)
        
        # Arrow to next step
        if step['y'] > 2:
            ax4.arrow(6, step['y'] - 1, 0, -1,
                     head_width=0.2, head_length=0.15, fc='black', ec='black',
                     alpha=0.5, linewidth=1)
    
    # ========== MODEL COMPARISON TABLE ==========
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.set_title('Model Comparison (2022)', fontsize=14, fontweight='bold', pad=10)
    ax5.set_xlim(0, 10)
    ax5.set_ylim(0, 10)
    ax5.axis('off')
    
    # Models to compare
    comparison_data = [
        {
            'name': 'DALL-E 2',
            'company': 'OpenAI',
            'params': '3.5B',
            'fid': '10.4',
            'strengths': ['Best semantic alignment', 'Good composition'],
            'limitations': ['Not open source', 'Limited access'],
            'color': '#4ECDC4'
        },
        {
            'name': 'Imagen',
            'company': 'Google',
            'params': '2.3B',
            'fid': '7.3',
            'strengths': ['Best FID score', 'High photorealism'],
            'limitations': ['Not released', 'Compute intensive'],
            'color': '#45B7D1'
        },
        {
            'name': 'Stable Diffusion',
            'company': 'Stability AI',
            'params': '890M',
            'fid': '12.6',
            'strengths': ['Open source', 'Efficient (latent)'],
            'limitations': ['Lower FID', 'Smaller model'],
            'color': '#FF6B6B'
        },
        {
            'name': 'Midjourney',
            'company': 'Midjourney',
            'params': 'N/A',
            'fid': 'N/A',
            'strengths': ['Artistic style', 'User friendly'],
            'limitations': ['Closed model', 'Subscription'],
            'color': '#FFD166'
        }
    ]
    
    for i, model in enumerate(comparison_data):
        y_pos = 9 - i * 2.2
        
        # Model header
        header = FancyBboxPatch((0.5, y_pos - 0.3), 9, 0.6,
                               boxstyle="round,pad=0.1",
                               facecolor=model['color'], alpha=0.3,
                               edgecolor=model['color'], linewidth=2)
        ax5.add_patch(header)
        
        # Model name and basic info
        basic_info = f"{model['name']} ({model['company']}) | Params: {model['params']} | FID: {model['fid']}"
        ax5.text(5, y_pos, basic_info, fontsize=10, fontweight='bold',
                ha='center', va='center')
        
        # Strengths
        strengths_text = 'Strengths:\n' + '\n'.join([f'• {s}' for s in model['strengths']])
        ax5.text(1.5, y_pos - 1, strengths_text, fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="lightgreen", alpha=0.3))
        
        # Limitations
        limits_text = 'Limitations:\n' + '\n'.join([f'• {l}' for l in model['limitations']])
        ax5.text(6.5, y_pos - 1, limits_text, fontsize=8, ha='left', va='top',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFCCCC", alpha=0.3))
    
    # ========== LATENT DIFFUSION EXPLANATION ==========
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_title('Latent Diffusion (Stable Diffusion)', fontsize=14, fontweight='bold', pad=10)
    ax6.set_xlim(0, 10)
    ax6.set_ylim(0, 10)
    ax6.axis('off')
    
    # Process flow
    steps = [
        {'name': 'Text\nEncoder', 'x': 1, 'y': 8, 'color': '#FF9999'},
        {'name': 'Image\nEncoder', 'x': 4, 'y': 8, 'color': '#99FF99'},
        {'name': 'Latent\nSpace', 'x': 7, 'y': 6, 'color': '#9999FF'},
        {'name': 'Diffusion\nModel', 'x': 4, 'y': 4, 'color': '#FFFF99'},
        {'name': 'Image\nDecoder', 'x': 7, 'y': 2, 'color': '#FF99FF'},
    ]
    
    connections = [
        (0, 2),  # Text encoder → Latent space
        (1, 2),  # Image encoder → Latent space
        (2, 3),  # Latent space → Diffusion
        (3, 4),  # Diffusion → Decoder
    ]
    
    # Draw steps
    for i, step in enumerate(steps):
        step_box = FancyBboxPatch((step['x'] - 0.8, step['y'] - 0.4), 1.6, 0.8,
                                 boxstyle="round,pad=0.1",
                                 facecolor=step['color'], alpha=0.6,
                                 edgecolor='black', linewidth=1)
        ax6.add_patch(step_box)
        ax6.text(step['x'], step['y'], step['name'], fontsize=9, fontweight='bold',
                ha='center', va='center')
    
    # Draw connections
    for start_idx, end_idx in connections:
        start = steps[start_idx]
        end = steps[end_idx]
        
        # Draw arrow
        ax6.arrow(start['x'], start['y'] - 0.4, 
                 end['x'] - start['x'], end['y'] + 0.4 - (start['y'] - 0.4),
                 head_width=0.15, head_length=0.15, fc='black', ec='black',
                 alpha=0.5, linewidth=1)
    
    # Add explanation
    explanation = """
    Key Innovations:
    1. Works in compressed latent space
    2. Much faster than pixel-space diffusion
    3. Text conditioning via cross-attention
    4. Open source and efficient
    """
    
    ax6.text(5, 1, explanation, fontsize=9, ha='center', va='top',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # ========== PROMPT ENGINEERING EXAMPLES ==========
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.set_title('Prompt Engineering: Adding Details Improves Results', 
                  fontsize=14, fontweight='bold', pad=10)
    ax7.set_xlim(0, 10)
    ax7.set_ylim(0, 10)
    ax7.axis('off')
    
    prompt_examples = [
        {
            'prompt': 'A cat',
            'quality': 'Basic',
            'y': 9,
            'image_elements': ['Simple cat shape', 'Plain background']
        },
        {
            'prompt': 'A cute cat sitting on a windowsill',
            'quality': 'Better',
            'y': 6.5,
            'image_elements': ['Detailed cat', 'Windowsill', 'Indoor scene']
        },
        {
            'prompt': 'A photorealistic Persian cat with blue eyes,\nsitting on a sunny windowsill with curtains,\ncinematic lighting, 8k',
            'quality': 'Best',
            'y': 3,
            'image_elements': ['Specific breed', 'Eye color', 'Lighting details', 'High resolution']
        }
    ]
    
    for example in prompt_examples:
        # Prompt text
        prompt_box = FancyBboxPatch((1, example['y'] - 0.6), 4, 1.2,
                                   boxstyle="round,pad=0.1",
                                   facecolor='lightblue', alpha=0.3,
                                   edgecolor='blue', linewidth=1)
        ax7.add_patch(prompt_box)
        ax7.text(3, example['y'], example['prompt'], fontsize=8, ha='center', va='center')
        
        # Quality rating
        ax7.text(0.5, example['y'], example['quality'], fontsize=9, fontweight='bold',
                ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Image representation
        img_box = Rectangle((5.5, example['y'] - 0.8), 4, 1.6,
                           facecolor='white', alpha=0.9,
                           edgecolor='black', linewidth=2)
        ax7.add_patch(img_box)
        
        # Draw cat based on prompt complexity
        if 'Basic' in example['quality']:
            # Simple cat
            cat_body = Ellipse((7.5, example['y']), 1, 0.6, facecolor='gray', alpha=0.7)
            cat_head = Circle((8.2, example['y'] + 0.2), 0.4, facecolor='gray', alpha=0.7)
            ax7.add_patch(cat_body)
            ax7.add_patch(cat_head)
        elif 'Better' in example['quality']:
            # More detailed cat
            cat_body = Ellipse((7.5, example['y']), 1.2, 0.8, facecolor='orange', alpha=0.8)
            cat_head = Circle((8.3, example['y'] + 0.2), 0.5, facecolor='orange', alpha=0.8)
            # Windowsill
            sill = Rectangle((6, example['y'] - 0.4), 3, 0.2, facecolor='#8B4513', alpha=0.7)
            ax7.add_patch(cat_body)
            ax7.add_patch(cat_head)
            ax7.add_patch(sill)
        else:
            # Most detailed cat
            cat_body = Ellipse((7.5, example['y']), 1.5, 1, facecolor=(0.8, 0.6, 0.4), alpha=0.9)
            cat_head = Circle((8.5, example['y'] + 0.3), 0.6, facecolor=(0.8, 0.6, 0.4), alpha=0.9)
            # Blue eyes
            eye1 = Circle((8.3, example['y'] + 0.4), 0.1, facecolor='blue')
            eye2 = Circle((8.7, example['y'] + 0.4), 0.1, facecolor='blue')
            # Windowsill with curtains
            sill = Rectangle((6, example['y'] - 0.4), 3, 0.2, facecolor='#654321', alpha=0.8)  # Changed from 'darkbrown' to hex
            curtain_left = Rectangle((6, example['y'] - 0.2), 0.5, 1.2, facecolor='red', alpha=0.6)
            curtain_right = Rectangle((8.5, example['y'] - 0.2), 0.5, 1.2, facecolor='red', alpha=0.6)
            # Sunlight effect
            for i in range(5):
                x = 7 + np.random.uniform(-0.5, 0.5)
                y = example['y'] + np.random.uniform(-0.3, 0.3)
                sunbeam = Ellipse((x, y), 0.3, 0.1, angle=45,
                                 facecolor='yellow', alpha=0.3)
                ax7.add_patch(sunbeam)
            
            ax7.add_patch(cat_body)
            ax7.add_patch(cat_head)
            ax7.add_patch(eye1)
            ax7.add_patch(eye2)
            ax7.add_patch(sill)
            ax7.add_patch(curtain_left)
            ax7.add_patch(curtain_right)
        
        # Elements list
        elements_text = '\n'.join([f'• {e}' for e in example['image_elements']])
        ax7.text(9.5, example['y'], elements_text, fontsize=7, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # ========== APPLICATIONS AND IMPACT ==========
    ax8 = fig.add_subplot(gs[2, 1])
    ax8.set_title('Applications and Societal Impact', fontsize=14, fontweight='bold', pad=10)
    ax8.set_xlim(0, 10)
    ax8.set_ylim(0, 10)
    ax8.axis('off')
    
    applications = [
        {'app': 'Creative Arts', 'desc': 'Digital art, illustrations,\nconcept art generation', 'icon': '🎨', 'y': 9},
        {'app': 'Design & Advertising', 'desc': 'Product visualization,\nad creatives, mockups', 'icon': '📱', 'y': 7},
        {'app': 'Education', 'desc': 'Visual aids, textbook\nillustrations, simulations', 'icon': '📚', 'y': 5},
        {'app': 'Entertainment', 'desc': 'Game assets, movie\nconcepts, storyboarding', 'icon': '🎬', 'y': 3},
        {'app': 'Research', 'desc': 'Data visualization,\nscientific illustrations', 'icon': '🔬', 'y': 1},
    ]
    
    for app in applications:
        # Icon
        ax8.text(1, app['y'], app['icon'], fontsize=20, ha='center', va='center')
        
        # Application name
        ax8.text(2.5, app['y'], app['app'], fontsize=10, fontweight='bold',
                ha='left', va='center')
        
        # Description
        desc_box = FancyBboxPatch((5, app['y'] - 0.5), 4.5, 1,
                                 boxstyle="round,pad=0.1",
                                 facecolor='lightgray', alpha=0.6,
                                 edgecolor='gray', linewidth=0.5)
        ax8.add_patch(desc_box)
        ax8.text(7.25, app['y'], app['desc'], fontsize=8, ha='center', va='center')
    
    # ========== ETHICAL CONSIDERATIONS ==========
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.set_title('Ethical Considerations and Challenges', fontsize=14, fontweight='bold', pad=10)
    ax9.set_xlim(0, 10)
    ax9.set_ylim(0, 10)
    ax9.axis('off')
    
    ethical_issues = [
        {
            'issue': 'Bias & Fairness',
            'desc': 'Models reflect biases in\ntraining data',
            'examples': ['Underrepresentation', 'Stereotypes'],
            'color': '#FF6B6B',
            'y': 9
        },
        {
            'issue': 'Misinformation',
            'desc': 'Realistic fake images\nfor disinformation',
            'examples': ['Deepfakes', 'Fake news'],
            'color': '#FFD166',
            'y': 7
        },
        {
            'issue': 'Copyright',
            'desc': 'Training on copyrighted\nworks without permission',
            'examples': ['Artist styles', 'Stock photos'],
            'color': '#06D6A0',
            'y': 5
        },
        {
            'issue': 'Job Displacement',
            'desc': 'Impact on creative\nprofessions',
            'examples': ['Illustrators', 'Designers'],
            'color': '#118AB2',
            'y': 3
        },
        {
            'issue': 'Watermarking',
            'desc': 'Identifying AI-generated\ncontent',
            'examples': ['Detection tools', 'Metadata'],
            'color': '#EF476F',
            'y': 1
        }
    ]
    
    for issue in ethical_issues:
        # Issue box
        issue_box = FancyBboxPatch((0.5, issue['y'] - 0.4), 3, 0.8,
                                  boxstyle="round,pad=0.1",
                                  facecolor=issue['color'], alpha=0.3,
                                  edgecolor=issue['color'], linewidth=1)
        ax9.add_patch(issue_box)
        ax9.text(2, issue['y'], issue['issue'], fontsize=9, fontweight='bold',
                ha='center', va='center')
        
        # Description
        ax9.text(4, issue['y'], issue['desc'], fontsize=8, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white", alpha=0.7))
        
        # Examples
        examples_text = 'Examples:\n' + '\n'.join([f'• {e}' for e in issue['examples']])
        ax9.text(7.5, issue['y'], examples_text, fontsize=7, ha='left', va='center',
                bbox=dict(boxstyle="round,pad=0.1", facecolor="lightyellow", alpha=0.5))
    
    # Add summary text
    summary_text = """
    Text-to-Image Synthesis has revolutionized creative AI:
    • From simple GANs to sophisticated diffusion models
    • Enables anyone to create visual content from text
    • Raises important ethical questions about AI creativity
    • Continues to improve in quality and accessibility
    """
    
    fig.text(0.02, 0.02, summary_text, fontsize=10, style='italic',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('images/text_to_image.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

def create_diffusion_process_detail():
    """
    Create a detailed visualization of the diffusion process
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Diffusion Models: The Core of Modern Text-to-Image Synthesis', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Forward diffusion process
    ax1 = axes[0, 0]
    ax1.set_title('Forward Diffusion: Adding Noise', fontsize=12, fontweight='bold', pad=10)
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')
    
    # Create a simple image and show noise addition
    # Original image (simple smiley face)
    ax1.text(5, 9, 'Original Image → Noise', fontsize=10, fontweight='bold',
            ha='center', va='center')
    
    # Draw original image
    face_original = Circle((3, 6), 2, facecolor='yellow', edgecolor='black', linewidth=2)
    eye1_original = Circle((2.3, 6.5), 0.3, facecolor='black')
    eye2_original = Circle((3.7, 6.5), 0.3, facecolor='black')
    mouth_original = patches.Arc((3, 5.5), 2, 1, theta1=200, theta2=340, linewidth=3, color='black')
    
    ax1.add_patch(face_original)
    ax1.add_patch(eye1_original)
    ax1.add_patch(eye2_original)
    ax1.add_patch(mouth_original)
    
    # Add noise gradually
    for i in range(3):
        x_offset = 3.5 + i * 2
        noise_level = 0.3 * (i + 1)
        
        # Draw noisy version
        face_noisy = Circle((x_offset, 6), 2, facecolor='yellow', alpha=1-noise_level*0.5,
                           edgecolor='black', linewidth=2)
        ax1.add_patch(face_noisy)
        
        # Add random noise dots
        for j in range(int(50 * noise_level)):
            noise_x = x_offset - 2 + np.random.uniform(0, 4)
            noise_y = 4 + np.random.uniform(0, 4)
            noise_size = np.random.uniform(0.05, 0.15)
            noise_color = np.random.rand(3)
            noise = Circle((noise_x, noise_y), noise_size, facecolor=noise_color, alpha=0.5)
            ax1.add_patch(noise)
        
        # Label noise level
        ax1.text(x_offset, 3, f'Noise: {noise_level:.1f}', fontsize=9,
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Arrow to next
        if i < 2:
            ax1.arrow(x_offset + 1, 6, 0.8, 0, head_width=0.2, head_length=0.15,
                     fc='black', ec='black', alpha=0.5)
    
    ax1.text(8.5, 6, 'Pure Noise', fontsize=10, fontweight='bold',
            ha='center', va='center')
    
    # Reverse diffusion process
    ax2 = axes[0, 1]
    ax2.set_title('Reverse Diffusion: Learning to Denoise', fontsize=12, fontweight='bold', pad=10)
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')
    
    # Show denoising process
    ax2.text(5, 9, 'Noise → Learned Denoising → Image', fontsize=10, fontweight='bold',
            ha='center', va='center')
    
    # Start with noise
    for i in range(3):
        x_offset = 1 + i * 3
        clarity = 0.7 - i * 0.2
        
        if i == 0:
            # Pure noise
            for j in range(100):
                noise_x = x_offset + np.random.uniform(-1, 1)
                noise_y = 6 + np.random.uniform(-1, 1)
                noise_size = np.random.uniform(0.02, 0.08)
                noise_color = np.random.rand(3)
                noise = Circle((noise_x, noise_y), noise_size, facecolor=noise_color)
                ax2.add_patch(noise)
            ax2.text(x_offset, 3, 'Pure Noise', fontsize=9,
                    ha='center', va='center')
        elif i == 1:
            # Partially denoised
            # Some structure emerging
            base_color = (0.8, 0.8, 0.3)
            face_partial = Circle((x_offset, 6), 1, facecolor=base_color, alpha=0.5,
                                 edgecolor='black', linewidth=1)
            ax2.add_patch(face_partial)
            
            # Add some noise on top
            for j in range(30):
                noise_x = x_offset + np.random.uniform(-1, 1)
                noise_y = 6 + np.random.uniform(-1, 1)
                noise_size = np.random.uniform(0.02, 0.05)
                noise_color = np.random.rand(3)
                noise = Circle((noise_x, noise_y), noise_size, facecolor=noise_color, alpha=0.3)
                ax2.add_patch(noise)
            
            ax2.text(x_offset, 3, 'Partially\nDenoised', fontsize=9,
                    ha='center', va='center')
        else:
            # Clear image
            face_clear = Circle((x_offset, 6), 1, facecolor='yellow', edgecolor='black', linewidth=2)
            eye1_clear = Circle((x_offset - 0.3, 6.2), 0.15, facecolor='black')
            eye2_clear = Circle((x_offset + 0.3, 6.2), 0.15, facecolor='black')
            mouth_clear = patches.Arc((x_offset, 5.8), 1.2, 0.6, theta1=220, theta2=320, 
                                     linewidth=2, color='black')
            
            ax2.add_patch(face_clear)
            ax2.add_patch(eye1_clear)
            ax2.add_patch(eye2_clear)
            ax2.add_patch(mouth_clear)
            
            ax2.text(x_offset, 3, 'Clean Image', fontsize=9,
                    ha='center', va='center')
        
        # Arrow to next
        if i < 2:
            ax2.arrow(x_offset + 1.2, 6, 0.6, 0, head_width=0.2, head_length=0.15,
                     fc='green', ec='green', alpha=0.7)
    
    # Training process
    ax3 = axes[1, 0]
    ax3.set_title('Training: Learning the Denoising Process', fontsize=12, fontweight='bold', pad=10)
    
    # Simulate training loss curves
    epochs = np.linspace(0, 100, 100)
    
    # Different components of loss
    reconstruction_loss = 2.0 * np.exp(-epochs/20) + 0.1 * np.sin(epochs/5) + 0.2
    kl_loss = 0.5 * np.exp(-epochs/30) + 0.05 * np.sin(epochs/8) + 0.1
    total_loss = reconstruction_loss + kl_loss
    
    ax3.plot(epochs, reconstruction_loss, 'b-', linewidth=2, label='Reconstruction Loss')
    ax3.plot(epochs, kl_loss, 'r-', linewidth=2, label='KL Divergence')
    ax3.plot(epochs, total_loss, 'g-', linewidth=3, label='Total Loss')
    
    ax3.set_xlabel('Training Steps', fontsize=10)
    ax3.set_ylabel('Loss', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    # Add training annotations
    ax3.text(20, 1.8, 'Rapid initial\nimprovement', fontsize=8,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    ax3.text(70, 0.6, 'Convergence', fontsize=8,
            ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))
    
    # Sampling process
    ax4 = axes[1, 1]
    ax4.set_title('Sampling: Generating New Images', fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlim(0, 10)
    ax4.set_ylim(0, 10)
    ax4.axis('off')
    
    # Sampling steps visualization
    ax4.text(5, 9, 'Sampling Process (T=50 steps)', fontsize=10, fontweight='bold',
            ha='center', va='center')
    
    # Show progressive sampling
    sampling_steps = [0, 10, 25, 40, 49]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sampling_steps)))
    
    for i, step in enumerate(sampling_steps):
        x_pos = 2 + i * 1.5
        progress = step / 49
        
        # Create representation of image at this step
        if progress < 0.3:
            # Mostly noise
            for j in range(20):
                noise_x = x_pos + np.random.uniform(-0.4, 0.4)
                noise_y = 6 + np.random.uniform(-0.4, 0.4)
                noise_size = np.random.uniform(0.02, 0.05)
                noise = Circle((noise_x, noise_y), noise_size, facecolor=colors[i], alpha=0.7)
                ax4.add_patch(noise)
        elif progress < 0.7:
            # Some structure
            # Base shape
            shape = Ellipse((x_pos, 6), 0.8, 0.6, angle=0,
                           facecolor=colors[i], alpha=0.5)
            ax4.add_patch(shape)
            
            # Some noise
            for j in range(10):
                noise_x = x_pos + np.random.uniform(-0.4, 0.4)
                noise_y = 6 + np.random.uniform(-0.4, 0.4)
                noise_size = np.random.uniform(0.01, 0.03)
                noise = Circle((noise_x, noise_y), noise_size, facecolor='black', alpha=0.3)
                ax4.add_patch(noise)
        else:
            # Clear structure
            # Draw a simple object
            if i % 2 == 0:
                # Circle-based object
                main = Circle((x_pos, 6), 0.5, facecolor=colors[i], alpha=0.8, edgecolor='black')
                detail1 = Circle((x_pos - 0.2, 6.1), 0.1, facecolor='white')
                detail2 = Circle((x_pos + 0.2, 6.1), 0.1, facecolor='white')
                ax4.add_patch(main)
                ax4.add_patch(detail1)
                ax4.add_patch(detail2)
            else:
                # Square-based object
                main = Rectangle((x_pos - 0.4, 5.6), 0.8, 0.8,
                                facecolor=colors[i], alpha=0.8, edgecolor='black')
                detail = Rectangle((x_pos - 0.2, 5.8), 0.4, 0.2, facecolor='white')
                ax4.add_patch(main)
                ax4.add_patch(detail)
        
        # Step label
        ax4.text(x_pos, 4.5, f'Step {step}', fontsize=8, fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))
        
        # Progress indicator
        progress_bar = Rectangle((x_pos - 0.5, 4), 1, 0.2,
                                facecolor='lightgray', edgecolor='black', linewidth=1)
        progress_fill = Rectangle((x_pos - 0.5, 4), progress, 0.2,
                                 facecolor=colors[i], alpha=0.7)
        ax4.add_patch(progress_bar)
        ax4.add_patch(progress_fill)
        
        # Arrow to next
        if i < len(sampling_steps) - 1:
            ax4.arrow(x_pos + 0.6, 6, 0.3, 0, head_width=0.15, head_length=0.1,
                     fc='black', ec='black', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('images/diffusion_process.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

if __name__ == "__main__":
    print("Generating text-to-image synthesis visualizations...")
    
    # Generate main text-to-image visualization
    fig1 = create_text_to_image_visualization()
    print("✓ Generated 'images/text_to_image.png'")
    
    # Generate diffusion process detail
    create_diffusion_process_detail()
    print("✓ Generated 'images/diffusion_process.png'")
    
    print("\nAll text-to-image visualizations have been generated successfully!")
    print("The visualizations include:")
    print("1. Comprehensive overview of text-to-image models and architectures")
    print("2. Example generation with the astronaut prompt")
    print("3. Model comparisons and ethical considerations")
    print("4. Detailed diffusion process explanation")