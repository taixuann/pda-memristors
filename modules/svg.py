import svgwrite
from svgwrite import cm, mm
import utils

class DiagramStyle:
    """Configuration constants for styling the SVG diagrams."""
    FONT_FAMILY = "Arial"
    
    # Conductivity Diagram Colors
    COND_LOW_CONC_BG = "#f4f4f4"
    COND_HIGH_CONC_BG = "#e8f4fc"
    COND_PRIMARY_BLUE = "#0275d8"
    COND_PRIMARY_RED = "#d9534f"
    K_ION_COLOR = "#9b59b6"
    PO4_ION_COLOR = "#2ecc71"
    DA_COLOR = "#333"
    ELECTRODE_FILL = "#ffd700"
    ELECTRODE_STROKE = "#d4af37"
    
    # pH Diagram Colors
    PH_LOW_BG = "#fff0f0"
    PH_HIGH_BG = "#f0f8ff"
    PH_ACIDIC_COLOR = "#c0392b"
    PH_BASIC_COLOR = "#2980b9"
    PH_PROTON_COLOR = "#e74c3c"
    PH_BUFFER_COLOR = "#27ae60"
    PH_BUFFER_FILL = "#2ecc71"

class SVGDrawer:
    """A helper class to create and configure SVG diagrams."""
    def __init__(self, filename, size):
        # Ensure the target directory exists
        utils.SVG_DIR.mkdir(parents=True, exist_ok=True)
        # Construct the full path for the SVG file
        full_path = utils.SVG_DIR / filename
        self.dwg = svgwrite.Drawing(str(full_path), size=size)
        self.style = DiagramStyle()

    def add_arrow_marker(self, marker_id, color):
        marker = self.dwg.marker(id=marker_id, insert=(0, 3), size=(10, 10), orient='auto')
        marker.add(self.dwg.path(d="M0,0 L0,6 L9,3 z", fill=color))
        self.dwg.defs.add(marker)
        return marker

    def add_header(self, text, subtitle, x_pos, color):
        self.dwg.add(self.dwg.text(text, insert=(x_pos, 40), text_anchor="middle", font_size=20, font_weight="bold", fill="#333", font_family=self.style.FONT_FAMILY))
        self.dwg.add(self.dwg.text(subtitle, insert=(x_pos, 70), text_anchor="middle", font_size=14, fill=color, font_family=self.style.FONT_FAMILY))

    def add_electrode(self, x_pos, width, y_pos=350, height=20, label="Working Electrode (Au)"):
        self.dwg.add(self.dwg.rect(insert=(x_pos, y_pos), size=(width, height), fill=self.style.ELECTRODE_FILL, stroke=self.style.ELECTRODE_STROKE, stroke_width=2))
        self.dwg.add(self.dwg.text(label, insert=(x_pos + width / 2, y_pos + height + 15), text_anchor="middle", font_size=14, font_family=self.style.FONT_FAMILY))

    def add_ion(self, center, radius, color, label="", label_color="white", font_size=10, opacity=1.0):
        self.dwg.add(self.dwg.circle(center=center, r=radius, fill=color, opacity=opacity))
        if label:
            self.dwg.add(self.dwg.text(label, insert=(center[0], center[1] + 4), text_anchor="middle", font_size=font_size, fill=label_color, font_family=self.style.FONT_FAMILY))

    def save(self):
        self.dwg.save(pretty=True)
        print(f"✔ SVG saved to: {self.dwg.filename}")

def create_conductivity_diagram():
    drawer = SVGDrawer('conductivity_regime.svg', size=(800, 400))
    dwg, style = drawer.dwg, drawer.style
    marker = drawer.add_arrow_marker('arrowBlue', style.COND_PRIMARY_BLUE)

    # Backgrounds and Divider
    dwg.add(dwg.rect(insert=(0, 0), size=(400, 400), fill=style.COND_LOW_CONC_BG))
    dwg.add(dwg.rect(insert=(400, 0), size=(400, 400), fill=style.COND_HIGH_CONC_BG))
    dwg.add(dwg.line(start=(400, 0), end=(400, 400), stroke="#333", stroke_width=2, stroke_dasharray="5,5"))

    # Headers
    drawer.add_header("0.01 M (Low Concentration)", "High Resistance (iR Drop)", 200, style.COND_PRIMARY_RED)
    drawer.add_header("0.1 M (High Concentration)", "Low Resistance (Fast Balance)", 600, style.COND_PRIMARY_BLUE)

    # Electrodes
    drawer.add_electrode(x_pos=50, width=300)
    drawer.add_electrode(x_pos=450, width=300)

    # Left Side (Low Conc)
    drawer.add_ion(center=(100, 150), radius=8, color=style.K_ION_COLOR, label="K+", opacity=0.6)
    drawer.add_ion(center=(300, 200), radius=8, color=style.K_ION_COLOR, label="K+", opacity=0.6)
    drawer.add_ion(center=(200, 120), radius=10, color=style.PO4_ION_COLOR, label="PO4", font_size=8, opacity=0.6)
    drawer.add_ion(center=(200, 320), radius=15, color=style.DA_COLOR, label="DA")
    dwg.add(dwg.path(d="M 200 305 L 200 250", stroke=style.COND_PRIMARY_RED, stroke_width=2, stroke_dasharray="4,4"))
    dwg.add(dwg.text("Slow Charge Balance", insert=(210, 280), font_size=12, fill=style.COND_PRIMARY_RED, font_family=style.FONT_FAMILY))

    # Right Side (High Conc)
    k_positions = [(480, 320), (520, 310), (700, 320), (650, 300), (550, 150), (680, 100), (620, 180)]
    for x, y in k_positions:
        drawer.add_ion(center=(x, y), radius=8, color=style.K_ION_COLOR, label="K+" if x == 480 else "")

    p_positions = [(600, 250), (500, 200), (720, 220)]
    for x, y in p_positions:
        drawer.add_ion(center=(x, y), radius=10, color=style.PO4_ION_COLOR)

    drawer.add_ion(center=(600, 320), radius=15, color=style.DA_COLOR, label="DA")

    line1 = dwg.line(start=(530, 315), end=(580, 320), stroke=style.COND_PRIMARY_BLUE, stroke_width=2)
    line1.set_markers((None, None, marker))
    dwg.add(line1)
    
    line2 = dwg.line(start=(690, 315), end=(620, 320), stroke=style.COND_PRIMARY_BLUE, stroke_width=2)
    line2.set_markers((None, None, marker))
    dwg.add(line2)

    dwg.add(dwg.text("Instant Shielding", insert=(600, 290), font_size=12, fill=style.COND_PRIMARY_BLUE, font_weight="bold", text_anchor="middle", font_family=style.FONT_FAMILY))

    drawer.save()

def create_ph_mechanism_diagram():
    drawer = SVGDrawer('ph_mechanism.svg', size=(800, 500))
    dwg, style = drawer.dwg, drawer.style

    marker_red = drawer.add_arrow_marker('arrowRed', style.PH_ACIDIC_COLOR)
    marker_blue = drawer.add_arrow_marker('arrowBlue', style.PH_BASIC_COLOR)

    # Backgrounds and Divider
    dwg.add(dwg.rect(insert=(0, 0), size=(400, 500), fill=style.PH_LOW_BG))
    dwg.add(dwg.rect(insert=(400, 0), size=(400, 500), fill=style.PH_HIGH_BG))
    dwg.add(dwg.line(start=(400, 0), end=(400, 500), stroke="#333", stroke_width=2))

    # Headers
    dwg.add(dwg.text("Low pH (6.4)", insert=(200, 40), text_anchor="middle", font_size=22, font_weight="bold", fill=style.PH_ACIDIC_COLOR, font_family=style.FONT_FAMILY))
    dwg.add(dwg.text("Protonated (Stable)", insert=(200, 70), text_anchor="middle", font_size=16, fill="#555", font_family=style.FONT_FAMILY))
    dwg.add(dwg.text("High pH (8.4)", insert=(600, 40), text_anchor="middle", font_size=22, font_weight="bold", fill=style.PH_BASIC_COLOR, font_family=style.FONT_FAMILY))
    dwg.add(dwg.text("Deprotonated (Reactive)", insert=(600, 70), text_anchor="middle", font_size=16, fill="#555", font_family=style.FONT_FAMILY))

    # Electrode
    dwg.add(dwg.rect(insert=(50, 450), size=(700, 30), fill=style.ELECTRODE_FILL, stroke=style.ELECTRODE_STROKE, stroke_width=3))
    dwg.add(dwg.text("GOLD ELECTRODE SURFACE", insert=(400, 475), text_anchor="middle", font_weight="bold", font_family=style.FONT_FAMILY))

    # Left Side (Low pH)
    protons = [(50, 150), (150, 100), (300, 180), (350, 300)]
    for x, y in protons:
        dwg.add(dwg.circle(center=(x, y), r=5, fill=style.PH_PROTON_COLOR))
        if x == 50:
            dwg.add(dwg.text("H+", insert=(x+15, y+5), font_size=12, fill=style.PH_PROTON_COLOR, font_family=style.FONT_FAMILY))

    da_group = dwg.g(transform="translate(150, 250)")
    da_group.add(dwg.polygon(points=[(30,0), (60,17), (60,52), (30,70), (0,52), (0,17)], fill="none", stroke="#333", stroke_width=3))
    da_group.add(dwg.text("DA", insert=(30, 45), text_anchor="middle", font_size=10, font_family=style.FONT_FAMILY))
    da_group.add(dwg.line(start=(30, 0), end=(30, -30), stroke="#333", stroke_width=3))
    da_group.add(dwg.text("NH3+", insert=(30, -35), text_anchor="middle", font_weight="bold", font_size="18", fill=style.PH_ACIDIC_COLOR, font_family=style.FONT_FAMILY))
    dwg.add(da_group)

    r_line = dwg.line(start=(180, 330), end=(180, 430), stroke=style.PH_ACIDIC_COLOR, stroke_width=2, stroke_dasharray="5,5")
    r_line.set_markers((None, None, marker_red))
    dwg.add(r_line)
    dwg.add(dwg.text("Hard to Oxidize", insert=(200, 380), font_size=14, fill=style.PH_ACIDIC_COLOR, font_family=style.FONT_FAMILY))

    # Right Side (High pH)
    buf_group = dwg.g(transform="translate(550, 150)")
    buf_group.add(dwg.circle(center=(0, 0), r=30, fill=style.PH_BUFFER_FILL, opacity=0.3))
    buf_group.add(dwg.text("HPO4--", insert=(0, 5), text_anchor="middle", font_weight="bold", fill=style.PH_BUFFER_COLOR, font_family=style.FONT_FAMILY))
    buf_group.add(dwg.text("(Base)", insert=(0, 20), text_anchor="middle", font_size=12, fill=style.PH_BUFFER_COLOR, font_family=style.FONT_FAMILY))
    dwg.add(buf_group)

    da_fast_group = dwg.g(transform="translate(550, 250)")
    da_fast_group.add(dwg.polygon(points=[(30,0), (60,17), (60,52), (30,70), (0,52), (0,17)], fill="none", stroke="#333", stroke_width=3))
    da_fast_group.add(dwg.text("DA", insert=(30, 45), text_anchor="middle", font_size=10, font_family=style.FONT_FAMILY))
    da_fast_group.add(dwg.line(start=(30, 0), end=(30, -30), stroke="#333", stroke_width=3))
    da_fast_group.add(dwg.text("NH2", insert=(30, -35), text_anchor="middle", font_weight="bold", font_size="18", fill=style.PH_BASIC_COLOR, font_family=style.FONT_FAMILY))
    dwg.add(da_fast_group)

    path_proton = dwg.path(d="M 580 210 Q 600 190 570 185", fill="none", stroke=style.PH_PROTON_COLOR, stroke_width=2)
    path_proton.set_markers((None, None, marker_red))
    dwg.add(path_proton)
    dwg.add(dwg.text("H+ removed by Buffer", insert=(650, 200), font_size=12, fill=style.PH_PROTON_COLOR, font_family=style.FONT_FAMILY))

    fast_line = dwg.line(start=(580, 330), end=(580, 440), stroke=style.PH_BASIC_COLOR, stroke_width=6)
    fast_line.set_markers((None, None, marker_blue))
    dwg.add(fast_line)
    
    dwg.add(dwg.text("FAST OXIDATION", insert=(610, 380), font_size=14, font_weight="bold", fill=style.PH_BASIC_COLOR, font_family=style.FONT_FAMILY))
    dwg.add(dwg.text("Rough Film Growth", insert=(610, 400), font_size=12, fill="#333", font_family=style.FONT_FAMILY))

    drawer.save()

if __name__ == "__main__":
    create_conductivity_diagram()
    create_ph_mechanism_diagram()