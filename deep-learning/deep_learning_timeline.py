import json
import matplotlib.pyplot as plt

# -----------------------------
# Load JSON from file
# -----------------------------
json_file = "timeline.json"

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

data.sort(key=lambda x: x["year"])
years = [item["year"] for item in data]

# Build labels: Year + Model + Inventor (if available)
labels = []
for item in data:
    label = f'{item["year"]}\n{item["model"]}'
    #if item.get("inventor"):
    #    label += "\n" + ", ".join(item["inventor"])
    labels.append(label)

# -----------------------------
# Plot timeline
# -----------------------------
fig, ax = plt.subplots(figsize=(20, 7))

ax.hlines(0, min(years) - 2, max(years) + 2)
ax.plot(years, [0] * len(years), "o")

# Choose rotation: 45 (angled) or 90 (perpendicular)
LABEL_ROTATION = 90   # change to 45 if you prefer

for i, (year, label) in enumerate(zip(years, labels)):
    if i % 2 == 0:
        ax.text(
            year,
            0.75,
            label,
            rotation=LABEL_ROTATION,
            ha="left",
            va="bottom",
            fontsize=9
        )
    else:
        ax.text(
            year,
            -0.75,
            label,
            rotation=LABEL_ROTATION,
            ha="left",
            va="top",
            fontsize=9
        )

# -----------------------------
# Styling
# -----------------------------
ax.set_ylim(-2.0, 2.0)
ax.set_yticks([])
ax.set_xlabel("Year")
#ax.set_title("Timeline of Neural Network and Deep Learning Models")

ax.set_xlim(min(years) - 3, max(years) + 3)

for spine in ["left", "right", "top"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.savefig("deep_learning_timeline.png", dpi=300, bbox_inches="tight")
plt.show()
