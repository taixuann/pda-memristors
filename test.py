import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_ph_effect():
    # 1. Setup the canvas with two side-by-side plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.4) # Space between plots

    # --- Common Definitions ---
    da_color = '#D4A017'  # Gold-ish for Dopamine core
    h_color = '#FF5733'   # Red for Protons (H+)
    bg_acid = '#ffebee'   # Light red background
    bg_base = '#e0f7fa'   # Light blue background

    def draw_dopamine_core(ax, x, y):
        # A simplified representation of the dopamine core molecule
        core = patches.Ellipse((x, y), width=3, height=2, angle=30, 
                               facecolor=da_color, edgecolor='black', zorder=2)
        ax.add_patch(core)
        ax.text(x, y, "Dopamine\nCore", ha='center', va='center', 
                fontweight='bold', zorder=3)
        
        # Functional group "ports" where H+ attaches
        # (Approximate positions for OH, OH, NH2)
        ports = [(x-1.2, y+1.2), (x+0.2, y+1.5), (x+1.8, y-0.5)]
        for px, py in ports:
            ax.plot([x, px], [y, py], color='black', lw=2, zorder=1)
            circle = patches.Circle((px, py), 0.3, facecolor='white', edgecolor='black', zorder=1)
            ax.add_patch(circle)
        return ports

    # ==============================
    # PANEL 1: LOW pH (ACID)
    # ==============================
    ax1.set_facecolor(bg_acid)
    ax1.set_title("A. LOW pH (Acidic Solution)\nHigh external [H+] concentration", fontsize=14, fontweight='bold', color='#b71c1c')
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Draw Core
    ports1 = draw_dopamine_core(ax1, 5, 5)

    # 1. Draw Protons attached tightly
    for px, py in ports1:
        # H+ stuck right on the port
        ax1.text(px, py, "H+", color='white', ha='center', va='center', 
                 fontweight='bold', bbox=dict(facecolor=h_color, edgecolor=h_color, boxstyle='circle'))

    # 2. Draw Environmental Protons (The Crowd) pushing in
    import random
    rng = random.Random(42) # Fixed seed for reproducibility
    for _ in range(20):
        # Random positions around the center
        ex = rng.uniform(1, 9)
        ey = rng.uniform(1, 9)
        # Don't draw too close to center
        if (ex-5)**2 + (ey-5)**2 > 4:
            ax1.text(ex, ey, "H+", color=h_color, ha='center', va='center', alpha=0.6, fontsize=9)
            # Arrow pushing inward
            ax1.arrow(ex, ey, (5-ex)*0.1, (5-ey)*0.1, head_width=0.15, head_length=0.2, fc=h_color, ec=h_color, alpha=0.5)

    # 3. Status Label
    ax1.text(5, 1, "STATUS: BLOCKED / STABLE\nExternal 'pressure' keeps protons attached.\nCannot oxidize.", 
             ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))


    # ==============================
    # PANEL 2: HIGH pH (BASIC)
    # ==============================
    ax2.set_facecolor(bg_base)
    ax2.set_title("B. HIGH pH (Basic Solution)\nLow external [H+] concentration", fontsize=14, fontweight='bold', color='#006064')
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 10)
    ax2.axis('off')

    # Draw Core
    ports2 = draw_dopamine_core(ax2, 5, 5)

    # 1. Draw Protons leaving
    for i, (px, py) in enumerate(ports2):
        # H+ moving away from the port
        dx, dy = (px-5)*0.5, (py-5)*0.5
        ax2.text(px+dx, py+dy, "H+", color='white', ha='center', va='center', 
                 fontweight='bold', bbox=dict(facecolor=h_color, edgecolor=h_color, boxstyle='circle'))
        
        # Arrow pointing outward
        ax2.arrow(px, py, dx*0.8, dy*0.8, head_width=0.2, head_length=0.3, fc='black', ec='black', width=0.05)
        
        # Label the open port
        if i == 2: # Amine group
            ax2.text(px, py, "NH2", ha='center', va='center', fontsize=9)
        else: # Hydroxyl groups
             ax2.text(px, py, "O-", ha='center', va='center', fontsize=9)


    # 2. Status Label
    ax2.text(5, 1, "STATUS: REACTIVE\nProtons are free to leave.\nElectrons can now flow (Oxidation).", 
             ha='center', va='center', fontsize=11, bbox=dict(facecolor='white', alpha=0.8))

    # Final Layout adjustments
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_ph_effect()