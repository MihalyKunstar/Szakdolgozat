import networkx as nx
import matplotlib.pyplot as plt

# Paraméterek
N = 20
p = 0.2

# Háló generálása (Erdős–Rényi prototípus)
G = nx.erdos_renyi_graph(N, p, seed=42)

# Kirajzolás
plt.figure()
nx.draw(G, with_labels=True)
plt.title("Prototípus kontaktusháló")
plt.savefig("network.png")
plt.show()

print("Háló létrehozva és elmentve.")
