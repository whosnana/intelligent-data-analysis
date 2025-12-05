import networkx as nx
import matplotlib.pyplot as plt
import os

os.makedirs("img", exist_ok=True)

brands = [
    "Fenty", "L'Oreal", "Maybelline", "Rare Beauty", "Sephora",
    "Huda Beauty", "NYX", "Dior", "Chanel"
]

segments = {
    "Fenty": "premium",
    "Rare Beauty": "premium",
    "Huda Beauty": "premium",
    "Sephora": "premium",
    "Dior": "luxury",
    "Chanel": "luxury",
    "L'Oreal": "mass",
    "Maybelline": "mass",
    "NYX": "mass"
}

channels = {
    "Fenty": "online",
    "Rare Beauty": "online",
    "Huda Beauty": "online",
    "Sephora": "omnichannel",
    "Dior": "offline",
    "Chanel": "offline",
    "L'Oreal": "omnichannel",
    "Maybelline": "omnichannel",
    "NYX": "online"
}

regions = {
    "Fenty": "North America",
    "Rare Beauty": "North America",
    "Huda Beauty": "Middle East",
    "Sephora": "Europe",
    "Dior": "Europe",
    "Chanel": "Europe",
    "L'Oreal": "Europe",
    "Maybelline": "North America",
    "NYX": "North America"
}

G = nx.Graph()

for b in brands:
    G.add_node(b, segment=segments[b], channel=channels[b], region=regions[b])

for i in range(len(brands)):
    for j in range(i + 1, len(brands)):
        b1 = brands[i]
        b2 = brands[j]
        sim = 0
        if segments[b1] == segments[b2]:
            sim += 2
        if channels[b1] == channels[b2]:
            sim += 1
        if regions[b1] == regions[b2]:
            sim += 1
        if sim >= 2:
            G.add_edge(b1, b2, weight=sim)

print("--- Інформація про граф косметичних брендів ---")
print(f"Кількість вузлів: {G.number_of_nodes()}")
print(f"Кількість ребер: {G.number_of_edges()}")

color_map = []
for node in G:
    if segments[node] == "mass":
        color_map.append("#1f78b4")
    elif segments[node] == "premium":
        color_map.append("#33a02c")
    else:
        color_map.append("#ff7f00")

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    node_color=color_map,
    with_labels=True,
    node_size=800,
    font_weight="bold",
    edge_color="#555555"
)

legend_handles = [
    plt.Line2D([0], [0], marker="o", color="w", label="mass", markerfacecolor="#1f78b4", markersize=10),
    plt.Line2D([0], [0], marker="o", color="w", label="premium", markerfacecolor="#33a02c", markersize=10),
    plt.Line2D([0], [0], marker="o", color="w", label="luxury", markerfacecolor="#ff7f00", markersize=10),
]

plt.legend(handles=legend_handles, title="Сегмент бренду", loc="upper left")
plt.title("Мережа косметичних брендів")

plt.savefig("img/1.png", dpi=300)
plt.close()

print("Графік збережено у img/1.png")
