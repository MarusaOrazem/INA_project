import networkx as nx
import os
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import floor

from create_directed_atlas import create_directed_atlas

p = Path(os.getcwd())
nets = p.parent.parent

G0 = nx.DiGraph(nx.read_pajek(str(nets) + '\\nets\\3749552\\3749552_Arsenal (1)_period_0.net'))
G1 = nx.DiGraph(nx.read_pajek(str(nets) + '\\nets\\3749552\\3749552_Arsenal (1)_period_1.net'))


""""
Cutoff
"""


# nx.draw(G0, with_labels = True)
# plt.show()

# cutoff_denominator = 3
# cutoff = floor(min(max([weight['weight'] for (a, b, weight) in G0.edges.data()]), max([weight['weight'] for (a, b, weight) in G1.edges.data()]))/cutoff_denominator)
# print(cutoff)
# G0_cutoff = nx.DiGraph()
# G0_cutoff.add_nodes_from(G0.nodes())
# G0_cutoff.add_edges_from([(a, b) for (a, b, c) in G0.edges.data() if c['weight'] > cutoff])
# nx.draw(G0_cutoff, with_labels = True)
# plt.subplots_adjust(left=0.2)
# plt.show()
#
# G1_cutoff = nx.DiGraph()
# G1_cutoff.add_nodes_from(G1.nodes())
# G1_cutoff.add_edges_from([(a, b) for (a, b, c) in G1.edges.data() if c['weight'] > cutoff])
# nx.draw(G1_cutoff, with_labels = True)
# plt.subplots_adjust(left=0.2)
# plt.title('test')
# plt.show()
# G0_randoms = []
# G1_randoms = []

"""
Creating random graph with same degree distribution
"""

din0 = list(d for n, d in G0.in_degree())
dout0 = list(d for n, d in G0.out_degree())

din1 = list(d for n, d in G1.in_degree())
dout1 = list(d for n, d in G1.out_degree())

G0_rand = nx.directed_configuration_model(din0, dout0)
G1_rand = nx.directed_configuration_model(din1, dout1)
# nx.draw(G1_rand, with_labels = True)
# plt.show()


example_motifs = create_directed_atlas()
motifs = {}
for i in range(1, 4):
    motifs[i] = []
for motif in example_motifs:
    motifs[motif.number_of_nodes()].append(motif)

unique_motif_count = len(example_motifs)

graph = G0
subsets = dict()
ss = [[edge[0], edge[1]] for edge in graph.edges()]
subsets[2] = ss
unique_subsets = dict()
for i in [3]:
    for subset in tqdm(ss):
        for node in subset:
            for neb in graph.neighbors(node):
                new_subset = subset + [neb]
                if len(set(new_subset)) == i:
                    new_subset.sort()
                    unique_subsets[tuple(new_subset)] = 1
    ss = [list(k) for k, v in unique_subsets.items()]
    subsets[i] = ss
    unique_subsets = dict()

motif_counts = {i: {j: 0 for j in range(len(motifs[i]))} for i in motifs.keys()}
my_graph = G0
for size, node_lists in subsets.items():
    this_motifs = motifs[size]
    for nodes in tqdm(node_lists):
        sub_gr = my_graph.subgraph(list(nodes))
        for index, graph in enumerate(this_motifs):
            if nx.is_isomorphic(sub_gr, graph):
                motif_counts[size][index] += 1


"""

def setup_features(self):
    print("\nCounting orbital roles.\n")
    self.features = {node: {i:0 for i in range(self.unique_motif_count)}for node in self.graph.nodes()}
    for size, node_lists in self.edge_subsets.items():
        graphs = self.interesting_graphs[size]
        for nodes in tqdm(node_lists):
            sub_gr = self.graph.subgraph(nodes)
            for index, graph in enumerate(graphs):
                if nx.is_isomorphic(sub_gr, graph):
                    for node in sub_gr.nodes():
                        self.features[node][self.categories[size][index][(sub_gr.out_degree(node), sub_gr.in_degree(node))]] += 1
                    break
"""
