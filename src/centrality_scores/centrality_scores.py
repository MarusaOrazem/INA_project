import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.legend import Legend
import matplotlib.patches as mpatches

def tops(G, centralities):
    nodes = list(G.nodes())
    nodes.sort(key = lambda node:centralities[node], reverse = True)
    print(f" top nodes: { nodes } ")

def calculate_centralies(G1, G2):
    all_closeness = []
    all_betweeness = []
    for G in [G1,G2]:
        n = G.number_of_nodes()
        names = list(G.nodes())
        values_closeness = []
        values_betweeness = []

        # Closeness centrality
        closeness = nx.closeness_centrality(G)
        for node in names:
            values_closeness.append(closeness[node])

        #tops(G, closeness)

        # Betweeness centrality
        betweeness = nx.betweenness_centrality(G)
        for node in names:
            values_betweeness.append(betweeness[node])
        #tops(G, betweeness)

        all_closeness.append(values_closeness)
        all_betweeness.append(values_betweeness)

    types = ['first' for _ in range(n)]
    types+= ['second' for _ in range(n)]
    df = pd.DataFrame(data = {'players': names+names, 'type': types, 'closeness': np.array(all_closeness).flatten()})
    fig, axs = plt.subplots(2)
    fig.suptitle('Centrality scores')
    #axs[0].bar(names, all_closeness[0])
    #axs[1].bar(names, all_closeness[1])
    sns.barplot(x = 'players', y='closeness',
               hue = 'type',data=df, ax=axs[0])
    #axs[0].xlabel([str(i) for i in range(11)])
    axs[0].legend_.remove()

    df1 = pd.DataFrame(data={'players': names + names, 'type': types, 'betweeness': np.array(all_betweeness).flatten()})

    sns.barplot(x='players', y='betweeness',
                         hue='type', data=df1, ax=axs[1])

    fig.subplots_adjust(hspace=0)
    for ax in axs:
        ax.label_outer()
    plt.show()


if __name__ == "__main__":
    matches_ids = [303524, 303610, 303470, 18237, 7584, 7545, 7567, 303715, 22912, 2302764, 7565, 7542]
    for match_id in matches_ids[:1]:
        path = f'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets/{match_id}'
        nets = [f for f in os.listdir(path) if f.endswith('.net')]
        print(nets)
        net1='303524_Atlético Madrid (212)_period_0.net'
        net2='303524_Atlético Madrid (212)_period_1.net'
        G1 = nx.read_pajek(path + '/' + net1)
        G1 = nx.DiGraph(G1)
        G2 = nx.read_pajek(path + '/' + net2)
        G2 = nx.DiGraph(G2)
        calculate_centralies(G1,G2)
        '''
        for net in nets:
            G = nx.read_pajek(path + '/' + net)
            G = nx.DiGraph(G)
            calculate_centralies(G)'''