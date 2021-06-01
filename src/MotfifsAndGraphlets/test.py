import networkx as nx
import os
from pathlib import Path


p=Path(os.getcwd())

nets = p.parent.parent

#G2 = nx.read_pajek(str(nets) + '\\nets\\7430\\7430_period_1.net')

graph = nx.DiGraph(nx.read_pajek(str(nets) + '\\nets\\18242\\18242_Barcelona (217)_period_0.net'))
a = [weight['weight'] for (a, b, weight) in graph.edges.data()]
graph = nx.DiGraph(nx.read_pajek(str(nets) + '\\nets\\18242\\18242_Barcelona (217)_period_1.net'))
b = [weight['weight'] for (a, b, weight) in graph.edges.data()]
print(sum(a))
print(sum(b))
print(sum(a) +sum(b))