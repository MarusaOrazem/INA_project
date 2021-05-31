import networkx as nx

"""
Modelled after example at 
https://www.researchgate.net/figure/All-possible-3-node-directed-graphs_fig1_252312340
"""

"""
1-graphs
"""

"""
2-graphs
"""
nodes = [1,2]
e0 = [1,2]


"""
3-graphs
"""
nodes = [1, 2, 3]
e1 = [(2, 1), (2, 3)]
e2 = [(2, 1), (3, 2)]
e3 = [(2, 1), (2, 3), (3, 2)]
e4 = [(3, 1), (2, 1)]
e5 = [(3,1), (2, 1), (2, 3)]
e6 = [(3, 1), (2, 1), (3, 2), (2, 3)]
e7 = [(3,2), (2,1), (1,2)]
e8 = [(1,2), (2,1), (2,3),(3,2)]
e9 = [(1,2), (2,3), (3,1)]
e10 = [(3,1), (1,2), (2,1), (2,3)]
e11 = [(3,1), (1,2), (2,1), (3,2)]
e12 = [(3,1), (1,2), (2,1), (2,3), (3,1)]
e13 = [(1,3), (3,1), (1,2), (2,1), (2,3),(3,2)]

edges = [e1, e2, e3, e4, e5, e6, e7, e8, e9, e10, e11, e12, e13]

def create_directed_atlas(size = 3):
    atlas = []
    for edge in edges:
        G3 = nx.DiGraph()
        G3.add_nodes_from(nodes)
        G3.add_edges_from(edge)
        atlas.append(G3)

    G2 = nx.DiGraph()
    G2.add_nodes_from([1,2])
    G2.add_edges_from([(1,2)])
    atlas.append(G2)

    G1 = nx.DiGraph()
    G1.add_nodes_from([1])
    atlas.append(G1)
    return atlas