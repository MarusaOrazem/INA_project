import networkx as nx
import os
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import floor
import statistics
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from create_directed_atlas import create_directed_atlas
plt.style.use('seaborn')

p = Path(os.getcwd())
nets = p.parent.parent


"""
Creating random graph with same degree distribution
"""

example_motifs = create_directed_atlas()
motifs = {}


for i in range(1, 4):
    motifs[i] = []
for motif in example_motifs:
    if motif.number_of_nodes() <=1:continue
    motifs[motif.number_of_nodes()].append(motif)

unique_motif_count = len(example_motifs)

def get_motif_counts(my_graph):
    subsets = dict()
    ss = [[edge[0], edge[1]] for edge in my_graph.edges()]
    subsets[2] = ss
    unique_subsets = dict()
    for i in [3]:
        for subset in tqdm(ss):
            for node in subset:
                for neb in my_graph.neighbors(node):
                    new_subset = subset + [neb]
                    if len(set(new_subset)) == i:
                        new_subset.sort()
                        unique_subsets[tuple(new_subset)] = 1
        ss = [list(k) for k, v in unique_subsets.items()]
        subsets[i] = ss
        unique_subsets = dict()

    motif_counts = {i: {j: 0 for j in range(len(motifs[i]))} for i in motifs.keys()}

    for size, node_lists in subsets.items():
        this_motifs = motifs[size]
        for nodes in tqdm(node_lists):
            sub_gr = my_graph.subgraph(list(nodes))
            for index, graph in enumerate(this_motifs):
                if nx.is_isomorphic(sub_gr, graph):
                    motif_counts[size][index] += 1
    return motif_counts

src = p.parent.parent
odiffs = {}
root = os.path.join(src, 'chosen_nets')
for match in next(os.walk(root))[1]:
    l1 = os.path.join(root, match)
    for team in next(os.walk(l1))[1]:
        l2 = os.path.join(l1, team)
        for split_method in next(os.walk(l2))[1]:
            fig, ax = plt.subplots(2, 1)
            l3 = os.path.join(l2, split_method)
            nets = [x for x in os.listdir(l3) if (os.path.isfile(l3+'\\'+x) and x.split('.')[-1]=='net')]
            assert len(list(nets)) == 2
            graphs = []
            for net in nets:
                G = nx.DiGraph(nx.read_pajek(l3+f'\\{net}'))
                graphs.append(G)
            # med1 = statistics.median([weight['weight'] for (a, b, weight) in graphs[0].edges.data()])
            # med2 = statistics.median([weight['weight'] for (a, b, weight) in graphs[1].edges.data()])
            # cutoff = min(med1, med2)
            pruned_graphs = []
            for graph in graphs:
                cutoff = statistics.median([weight['weight'] for (a, b, weight) in graph.edges.data()])
                G_pruned = nx.DiGraph()
                G_pruned.add_nodes_from(graph.nodes())
                G_pruned.add_edges_from([(a, b) for (a, b, c) in graph.edges.data() if c['weight'] > cutoff])
                pruned_graphs.append(G_pruned)

            sns.set_palette('Set2')
            colors = ['green', 'red']
            halfs = ['First Half', 'Second Half']
            plot_df = {}
            for half in range(len(pruned_graphs)):
                PG = pruned_graphs[half]
                din = list(d for n, d in PG.in_degree())
                dout = list(d for n, d in PG.out_degree())

                PG_motif_counts = get_motif_counts(PG)
                rand = pd.DataFrame()
                num_iters = 100
                for seed in range(num_iters):
                    PG_rand = nx.directed_configuration_model(din, dout, seed=seed)
                    counts = get_motif_counts(PG_rand)
                    vals = list(counts.values())[1:]
                    cols = list([x.keys() for x in vals])
                    c2 = []
                    for i in cols:
                        c2 += i
                    c2 = [c2[0]] + [x+1 for x in c2[1:]]
                    #c2[0] += f'_2'
                    r = list([x.values() for x in vals])
                    r2 = []
                    for i in r:
                        r2 += i
                    obss = {c:v for (c,v) in zip(c2,r2)}
                    df = pd.DataFrame(obss, index=[0])
                    if seed==0:
                        rand = df
                        continue
                    rand = rand.append(df)
                    rand = rand.reset_index(drop=True)

                vals = list(PG_motif_counts.values())[1:]
                cols = list([x.keys() for x in vals])
                c2 = []
                for i in cols:
                    c2 += i
                c2 = [c2[0]] + [x+1 for x in c2[1:]]
                #c2[0] += f'_2'
                r = list([x.values() for x in vals])
                r2 = []
                for i in r:
                    r2 += i
                gtd = {c:v for (c,v) in zip(c2,r2)}
                gt_df = pd.DataFrame(gtd, index=[0])
                r_means = rand.mean()
                r_sds = rand.std()
                #print(gt_df)
                zs = {}
                for i in gt_df.columns:
                    zs[i] = (gt_df[i][0] - r_means[i])/r_sds[i]
                zs = pd.DataFrame(zs, index=['Z-score'])
                ##currently replacing infs
                zs = zs.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
                denominator = sum([z**2 for z in zs])
                significance = {}
                for i in zs.columns:
                    significance[i] = zs[i]['Z-score']/denominator
                significance = pd.DataFrame(significance, index=[halfs[half]])
                significance = significance.transpose()
                plot_df.update({halfs[half] : significance[halfs[half]]})
            plot_df = pd.DataFrame(plot_df)
            print(plot_df)
            sns.lineplot(data=plot_df, ax=ax[0])
            diff = plot_df[halfs[0]] - plot_df[halfs[1]]
            odiffs[(match, team, split_method)] =  diff
            sns.lineplot(x=range(len(diff)), y=diff, ax=ax[1])
            ax[1].hlines(0, 0, 13, ls='--')

            ax[0].set_xlabel(r'Motif ordinal number $i$')
            ax[0].set_ylabel(f'Motif Significance')
            ax[0].set_title(f'Per-half significance')
            ax[0].set_title(f'Between-half significance difference')
            ax[1].set_xlabel(r'Motif ordinal number $i$')
            ax[1].set_ylabel(f'Motif Significance difference \n((+) is first half)')
            ax[0].set_xticks(list(range(len(diff))))
            ax[1].set_xticks(list(range(len(diff))))
            plt.suptitle(f'{team} motif significance profiles: {split_method} split')
            plt.savefig(str(l3)+f'/MotifSignificance_{split_method}', bbox_inches='tight')
            plt.subplots_adjust(hspace=0.5)
            #plt.show()


df =pd.DataFrame(odiffs).transpose()
melted = pd.melt(df)
melted2 = melted.copy()
melted2.value = [abs(val) for val in melted2.value]
print(df.columns)
fig, ax = plt.subplots(2,1)
col_line = [list(df[col]) for col in df.columns]
col_line2 = [[abs(x) for x in list(df[col])] for col in df.columns]
SPLIT_KIND = "halftime split"
sns.barplot(data=melted, x='variable', y='value', ax=ax[0])
sns.barplot(data=melted2, x='variable', y='value', ax=ax[1])
ax[0].set_title(f'Mean Between-{SPLIT_KIND[0]} Difference - pruned')
ax[1].set_title(f'Mean Absolute Between-{SPLIT_KIND[0]} Significance difference - pruned')
ax[1].set_xlabel(r'Motif ordinal number $i$')
ax[0].set_xlabel(r'Motif ordinal number $i$')
ax[0].set_ylabel(f'Motif Significance difference \n((+) is first half)')
ax[1].set_ylabel(f'Absolute Motif Significance difference')
ax[0].hlines(0, 0, 14, ls='--', color='black')
plt.suptitle(f'halftime split, pruned (median of each graph), 24 matches, {num_iters} random iters')
plt.subplots_adjust(hspace=0.5)
plt.savefig('././OVERALL SIGNIFICANCE DIFFERENCE')
plt.show()
