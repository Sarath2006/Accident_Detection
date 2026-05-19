"""
Example script to generate 3D visualization output
This demonstrates how to use the Visualization3D module
"""
import os
from forensics.visualization_3d import Visualization3D


def generate_3d_visualization():
    """
    Generate a sample 3D accident reconstruction visualization
    Output: collision_3d.html (interactive 3D visualization)
    """
    
    print("\n" + "="*70)
    print("3D VISUALIZATION GENERATION")
    print("="*70)
    
    # Initialize the 3D visualization engine
    viz_3d = Visualization3D()
    
    # Create output directory if it doesn't exist
    output_dir = "forensics_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate the 3D visualization
    output_path = os.path.join(output_dir, "collision_3d.html")
    
    print(f"\n[*] Generating 3D visualization...")
    print(f"[*] Output path: {output_path}")
    
    # Render collision (None = use default scene)
    viz_3d.render_collision(
        accident_scene=None,  # Uses default visualization
        output_path=output_path
    )
    
    print(f"\n[✓] 3D Visualization created successfully!")
    print(f"\n[+] How to view the output:")
    print(f"    1. Open this file in a web browser:")
    print(f"       {os.path.abspath(output_path)}")
    print(f"    2. You'll see an interactive 3D scene with:")
    print(f"       - Two vehicles (Red and Blue)")
    print(f"       - Green ground plane")
    print(f"       - Lighting and shadows")
    print(f"\n[+] Interactive Controls:")
    print(f"    - Drag mouse to rotate view")
    print(f"    - Scroll wheel to zoom in/out")
    print(f"    - Vehicles rotate automatically")
    
    return output_path


if __name__ == "__main__":
    # Generate and return the output path
    html_file = generate_3d_visualization()
    
    print(f"\n{'-'*70}")
    print(f"Output file: {html_file}")
    print(f"{'-'*70}\n")
    
    # Optional: Auto-open in browser (Windows)
    import sys
    if sys.platform == 'win32':
        import webbrowser
        full_path = os.path.abspath(html_file)
        webbrowser.open(f'file:///{full_path}')
        print(f"[✓] Opening {os.path.basename(html_file)} in default browser...")
