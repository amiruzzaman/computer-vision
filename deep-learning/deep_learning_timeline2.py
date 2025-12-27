import json
import matplotlib.pyplot as plt

# Load data
with open("timeline.json", "r", encoding="utf-8") as f:
    data = json.load(f)

data.sort(key=lambda x: x["year"])
years = [item["year"] for item in data]
labels = [f'{item["year"]}: {item["model"]}' for item in data]

# Create plot
fig, ax = plt.subplots(figsize=(26, 12))

# Main timeline line and markers
ax.hlines(0, min(years) - 2, max(years) + 2, color='black', linewidth=2, zorder=1)
ax.plot(years, [0] * len(years), "o", color='red', markersize=12, zorder=2)

# Multi-level offsets to prevent horizontal overlap
# [Level 1 Top, Level 1 Bottom, Level 2 Top, Level 2 Bottom]
offsets = [0.8, -0.8, 2.2, -2.2]
FONT_SIZE = 16

for i, (year, label) in enumerate(zip(years, labels)):
    offset = offsets[i % 4]
    va = "bottom" if offset > 0 else "top"
    
    # Leader lines for labels further from the axis
    if abs(offset) > 1.0:
        ax.vlines(year, 0, offset, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    
    ax.text(
        year, offset, label,
        rotation=90, ha="center", va=va,
        fontsize=FONT_SIZE, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1)
    )

# Final Styling
ax.set_ylim(-4.5, 4.5)
ax.set_yticks([])
ax.set_xlabel("Year", fontsize=18, labelpad=20)
ax.set_xlim(min(years) - 3, max(years) + 3)
ax.grid(axis='x', linestyle='--', alpha=0.3)

for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("deep_learning_timeline_improved.png", dpi=300, bbox_inches="tight")
plt.show()