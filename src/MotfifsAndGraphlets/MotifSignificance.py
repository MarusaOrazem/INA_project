import networkx as nx
import os
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm
from math import floor
import statistics
import math
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from create_directed_atlas import create_directed_atlas
import scipy.stats as stats
plt.style.use('seaborn')
from copy import deepcopy

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
        for subset in ss:
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
        for nodes in node_lists:
            sub_gr = my_graph.subgraph(list(nodes))
            for index, graph in enumerate(this_motifs):
                if nx.is_isomorphic(sub_gr, graph):
                    motif_counts[size][index] += 1
    return motif_counts

src = p.parent.parent
half_diffs_for = {}
goal_diffs_for = {}
cards_diffs_for = {}
half_diffs_against = {}
goal_diffs_against = {}
cards_diffs_against = {}

goals_for = {'before' : {i : [] for i in range(-1, 14, 1)}, 'after' : {i : [] for i in range(-1, 14, 1)}, 'sds' : []}
goals_against = {'before' : {i : [] for i in range(-1, 14, 1)}, 'after' : {i : [] for i in range(-1, 14, 1)}, 'sds' : [], }
hhalfs = {'before' : {i : [] for i in range(-1, 14, 1)}, 'after' : {i : [] for i in range(-1, 14, 1)}, 'sds' : [], }
cards_for = {'before' : {i : [] for i in range(-1, 14, 1)}, 'after' : {i : [] for i in range(-1, 14, 1)}, 'sds' : [], }
cards_against = {'before' : {i : [] for i in range(-1, 14, 1)}, 'after' : {i : [] for i in range(-1, 14, 1)}, 'sds' : [], }

root = os.path.join(src, 'chosen_nets')
for match in next(os.walk(root))[1]:
    l1 = os.path.join(root, match)
    with open(l1+ f"\\{match}.txt", "r") as f:
        line = f.readline()
        while(line[:4] != 'Goal'):
            line = f.readline()
        try:
            goal_for = line.split('\'')[-2]
        except IndexError:
            pass
        line = f.readline()
        try:
            card_for = line.split('\'')[-2]
        except IndexError:
            pass
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
            halfs = ['Before split', 'After split']
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
                    c2 = [-1, 0] + [x+1 for x in c2[2:]]
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
                c2 = [-1, 0] + [x+1 for x in c2[2:]]
                #c2[0] += f'_2'
                r = list([x.values() for x in vals])
                r2 = []
                for i in r:
                    r2 += i
                gtd = {c:v for (c,v) in zip(c2,r2)}
                gt_df = pd.DataFrame(gtd, index=[0])
                r_means = rand.mean()
                r_sds = rand.std()

                zs = {}
                for i in gt_df.columns:
                    zs[i] = (gt_df[i][0] - r_means[i])/r_sds[i]
                zs = pd.DataFrame(zs, index=['Z-score'])
                print(zs)
                ##currently replacing infs
                zs = zs.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
                print(zs)
                try:
                    denominator = sum([z**2 for z in zs.values.tolist()[0]])
                    variance = (1/denominator)**2
                    sigma = 1/denominator
                except ZeroDivisionError:
                    print('**ERROR HERE!**')
                    print([z for z in zs.values.tolist() if not np.isnan(z)][0])
                    print('**ERROR HERE!**')
                    continue
                significance = {}
                for i in zs.columns:
                    significance[i] = zs[i]['Z-score']/denominator
                    #significance[i] = zs[i]['Z-score']
                significance = pd.DataFrame(significance, index=[halfs[half]])
                significance = significance.transpose()
                plot_df.update({halfs[half] : significance[halfs[half]]})
            plot_df = pd.DataFrame(plot_df)

            try:
                plot_df.index = plot_df.index.rename(r'Motif ordinal number $i$')
                sns.lineplot(data=plot_df, ax=ax[0])
                ax[0].set_xticks(list(range(-1, 14, 1)))
                plot_df = plot_df.reset_index()
                diff = plot_df[halfs[0]] - plot_df[halfs[1]]
            except KeyError:
                print(f'No after found in {match}, {team}, {split_method}')
                continue
            if split_method == 'periods':
                for (i, before, after) in zip(plot_df[r'Motif ordinal number $i$'], plot_df['Before split'], plot_df['After split']):
                    if not np.isnan(before) and not np.isnan(after):
                        hhalfs['before'][i].append(before)
                        hhalfs['after'][i].append(after)
                        hhalfs['sds'].append(sigma)
                if goal_for[:4] == team[:4]:
                    half_diffs_for[(match, team, split_method)] =  diff
                else:
                    half_diffs_against[(match, team, split_method)] =  diff
            elif split_method == 'goals':
                if goal_for[:4] == team[:4]:
                    print(f'Goal for {team} in match {match}')
                    goal_diffs_for[(match, team, split_method)] =  diff
                    for (i, before, after) in zip(plot_df[r'Motif ordinal number $i$'], plot_df['Before split'], plot_df['After split']):
                        if not np.isnan(before) and not np.isnan(after):
                            goals_for['before'][i].append(before)
                            goals_for['after'][i].append(after)
                            goals_for['sds'].append(sigma)
                else:
                    print(f'Goal against {team} in match {match}')
                    goal_diffs_against[(match, team, split_method)] =  diff
                    for (i, before, after) in zip(plot_df[r'Motif ordinal number $i$'], plot_df['Before split'], plot_df['After split']):
                        if not np.isnan(before) and not np.isnan(after):
                            goals_against['before'][i].append(before)
                            goals_against['after'][i].append(after)
                            goals_against['sds'].append(sigma)
            elif split_method == 'cards':
                if card_for[:4] == team[:4]:
                    print(f'card given to {team} in match {match}')
                    cards_diffs_for[(match, team, split_method)] =  diff
                    for (i, before, after) in zip(plot_df[r'Motif ordinal number $i$'], plot_df['Before split'], plot_df['After split']):
                        if not np.isnan(before) and not np.isnan(after):
                            cards_for['before'][i].append(before)
                            cards_for['after'][i].append(after)
                            cards_for['sds'].append(sigma)
                else:
                    print(f'card given to {team}\'s OPPONENT in match {match}')
                    cards_diffs_against[(match, team, split_method)] =  diff
                    for (i, before, after) in zip(plot_df[r'Motif ordinal number $i$'], plot_df['Before split'], plot_df['After split']):
                        if not np.isnan(before) and not np.isnan(after):
                            cards_against['before'][i].append(before)
                            cards_against['after'][i].append(after)
                            cards_against['sds'].append(sigma)
            sns.lineplot(x=range(len(diff)), y=diff, ax=ax[1])
            ax[1].set_xticks(list(range(-1, 14, 1)))
            ax[1].hlines(0, -1, 15, ls='--')
            ax[0].set_xlabel(r'Motif ordinal number $i$')
            ax[0].set_ylabel(f'Motif Significance')
            ax[0].set_title(f'Per-half significance')
            #ax[0].set_xticks(range(0,16))
            #ax[0].set_xticklabels(['0a', '0b',] + list(range(1,14,1)))
            ax[0].set_title(f'Between-{split_method} significance difference')
            #ax[1].set_xticks(range(0,16))
            #ax[1].set_xticklabels(['0a', '0b',] + list(range(1,14,1)))
            ax[1].set_xlabel(r'Motif ordinal number $i$')
            ax[1].set_ylabel(f'Motif Significance difference \n((+) is first half)')
            ax[0].set_xticks(list(range(len(diff))))
            ax[1].set_xticks(list(range(len(diff))))
            plt.suptitle(f'{team} motif significance profiles: {split_method} split')
            plt.subplots_adjust(hspace=0.5)
            plt.savefig(str(l3)+f'/MotifSignificance_{split_method}', bbox_inches='tight')
            fig,ax = plt.subplots(3,1, figsize=(5,10))
            mu = 0
            x = np.linspace(mu - 3*sigma, mu + 22*sigma, 300)
            sns.lineplot(x=x, y=stats.norm.pdf(x,mu,sigma), ax=ax[0])
            #ax[0].set_xlim([-1/denominator, 22/denominator])
            sns.boxplot(data=plot_df.reset_index(), x='Before split', y=r'Motif ordinal number $i$', orient='h', ax=ax[1], showfliers=False)
            #ax[1].set_xlim([-1/denominator, 22/denominator])
            #ax[0].vlines(3/denominator, 0, 220, alpha=0.2, ls='--', label=r'$3*\sigma$')
            ax[1].vlines(3/denominator, 14, -2, alpha=0.2, ls='--', label=r'$3*\sigma$')

            ax[1].legend()

            ax[0].set_xlabel(r'Normalized $z$ score')
            ax[0].set_title('Normal distribution for normalized z-score comparison')
            ax[1].set_title('Before split')
            ax[1].set_xlabel(r'Normalized $z$ score')
            ax[2].set_title('After split')
            sns.boxplot(data=plot_df.reset_index(), x='After split', y=r'Motif ordinal number $i$', orient='h', ax=ax[2], showfliers=False)
            ax[2].vlines(3/denominator, 14, -2, alpha=0.2, ls='--', label=r'$3*\sigma$')
            ax[2].legend()
            ax[2].set_xlabel(r'Normalized $z$ score')
            #ax[2].set_xlim([-1/denominator, 22/denominator])
            plt.subplots_adjust(hspace=0.4)
            plt.savefig(str(l3)+f'/DifferentMotifSignificance_{split_method}-boxplot', bbox_inches='tight')
            plt.show()



for dd in [hhalfs, goals_for, goals_against, cards_for,cards_against]:
    print(dd)
    for key in ['before', 'after']:
        if not dd[key][i]:
            dd[key][i] = [0]
    dd['sd'] = np.mean(dd['sds'])

fig,ax = plt.subplots(1,2, figsize=(8,4))
_hhalfs = deepcopy(hhalfs)
_hhalfs.pop('sd', None)
_hhalfs.pop('counts', None)
_hhalfs.pop('sds', None)
_hhalfs_df = pd.DataFrame(_hhalfs)
_hhalfs_df.index = _hhalfs_df.index.rename(r'Motif ordinal number $i$')
_hhalfs_df = _hhalfs_df.reset_index()
print(_hhalfs_df)
_hhalfs_df.before = [j if j else [0] for j in _hhalfs_df.before]
_hhalfs_df.after = [j if j else [0] for j in _hhalfs_df.after]
maxlen = max([len(i) for i in _hhalfs_df.before])
_hhalfs_df.before = [j + (maxlen-len(j))*[np.mean(j)] for j in _hhalfs_df.before]
_hhalfs_df.after = [j + (maxlen-len(j))*[np.mean(j)] for j in _hhalfs_df.after]
_hhalfs_df_bef = pd.DataFrame({_hhalfs_df[r'Motif ordinal number $i$'][i] : _hhalfs_df['before'][i] for i in _hhalfs_df.index})
_hhalfs_df_aft = pd.DataFrame({_hhalfs_df[r'Motif ordinal number $i$'][i] : _hhalfs_df['after'][i] for i in _hhalfs_df.index})
print(_hhalfs_df)
sns.boxplot(data=_hhalfs_df_bef, orient='h', ax=ax[0], showfliers=False)
sns.boxplot(data=_hhalfs_df_aft, orient='h', ax=ax[1], showfliers=False)
#ax[1].set_xlim([-1*hhalfs['sd'], 12*hhalfs['sd']])
#ax[0].set_xlim([-1*hhalfs['sd'], 12*hhalfs['sd']])
ax[0].vlines(3*hhalfs['sd'], 14, -2, alpha=0.2, ls='--', label=r'$3*\sigma$')
ax[1].vlines(3*hhalfs['sd'], 14, -2, alpha=0.2, ls='--', label=r'$3*\sigma$')
ax[0].legend()
ax[1].legend()

ax[0].set_xlabel(r'Normalized $z$ score')
ax[0].set_ylabel(r'Motif ordinal number $i$')
ax[1].set_ylabel(r'Motif ordinal number $i$')
ax[0].set_title('Average Before-Halftime motif significance')
ax[1].set_title('Average After-Halftime motif significance')

ax[1].set_xlabel(r'Normalized $z$ score')
plt.subplots_adjust(wspace=0.3)
#plt.savefig(str(l3)+f'/DifferentMOtifSignificance_{split_method}', bbox_inches='tight')
plt.savefig(f'Avg OVERALL SIGNIFICANCE box - halftime - for-against-cis.png', bbox_inches='tight')
plt.show()

for (dds, kind) in zip([(goals_for, goals_against), (cards_for, cards_against)], ['First Goal Split', 'Dismissal Split']):
    fig,ax = plt.subplots(2,2, figsize=(10,10))
    _for = dds[0]
    print(_for)
    _against = dds[1]
    x_for = deepcopy(_for)
    x_for.pop('sd', None)
    x_for.pop('counts', None)
    x_for.pop('sds', None)
    x_against = deepcopy(_against)
    x_against.pop('counts', None)
    x_against.pop('sds', None)
    x_against.pop('sd', None)
    _for_df = pd.DataFrame(x_for)
    _against_df = pd.DataFrame(x_against)
    _for_df.index = _for_df.index.rename(r'Motif ordinal number $i$')
    _for_df = _for_df.reset_index()
    _against_df.index = _against_df.index.rename(r'Motif ordinal number $i$')
    _against_df = _against_df.reset_index()

    _against_df.before = [j if j else [0] for j in _against_df.before]
    _against_df.after = [j if j else [0] for j in _against_df.after]
    maxlen_bef = max([len(i) for i in _against_df.before])
    maxlen_aft = max([len(i) for i in _against_df.after])
    _against_df.before = [j + (maxlen_bef-len(j))*[np.mean(j)] for j in _against_df.before]
    _against_df.after = [j + (maxlen_aft-len(j))*[np.mean(j)] for j in _against_df.after]


    _for_df.before = [j if j else [0] for j in _for_df.before]
    _for_df.after = [j if j else [0] for j in _for_df.after]
    maxlen_bef = max([len(i) for i in _for_df.before])
    maxlen_aft = max([len(i) for i in _for_df.after])
    _for_df.before = [j + (maxlen_bef-len(j))*[np.mean(j)] for j in _for_df.before]
    _for_df.after = [j + (maxlen_aft-len(j))*[np.mean(j)] for j in _for_df.after]


    _for_df_bef = pd.DataFrame({_for_df[r'Motif ordinal number $i$'][i] : _for_df['before'][i] for i in _for_df.index})
    _for_df_aft = pd.DataFrame({_for_df[r'Motif ordinal number $i$'][i] : _for_df['after'][i] for i in _for_df.index})
    _against_df_bef = pd.DataFrame({_for_df[r'Motif ordinal number $i$'][i] : _against_df['before'][i] for i in _against_df.index})
    _against_df_aft = pd.DataFrame({_for_df[r'Motif ordinal number $i$'][i] : _against_df['after'][i] for i in _against_df.index})

    sns.boxplot(data=_for_df_bef, orient='h', ax=ax[0][0], showfliers=False)
    sns.boxplot(data=_for_df_aft, orient='h', ax=ax[1][0], showfliers=False)
    sns.boxplot(data=_against_df_bef, orient='h', ax=ax[0][1], showfliers=False)
    sns.boxplot(data=_against_df_aft, orient='h', ax=ax[1][1], showfliers=False)

    for axx in ax:
        #axx[1].set_xlim([-0.02, 0.1])
        #axx[0].set_xlim([-0.02, 0.1])
        axx[0].vlines(3 * _for['sd'], 14, -2, alpha=0.2, ls='--', label=r'$3*\sigma$')
        axx[1].vlines(3 * _against['sd'], 14, -2, alpha=0.2, ls='--', label=r'$3*\sigma$')
        axx[0].legend()
        axx[1].legend()
        axx[0].set_xlabel(r'Normalized $z$ score')
        axx[1].set_xlabel(r'Normalized $z$ score')
        axx[0].set_ylabel(r'Motif ordinal number $i$')
        axx[1].set_ylabel(r'Motif ordinal number $i$')
    ax[0][0].set_title(f'Event for Team - before {kind}')
    ax[1][0].set_title(f'Event for Team  - after {kind}')
    ax[0][1].set_title(f'Event for Opponent - before {kind}')
    ax[1][1].set_title(f'Event for Opponent - after {kind}')
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    #plt.savefig(str(l3)+f'/DifferentMOtifSignificance_{split_method}', bbox_inches='tight')
    plt.savefig(f'Avg OVERALL SIGNIFICANCE box - {kind} - for-against-cis.png', bbox_inches='tight')
    plt.show()




for (diffs, kind) in zip([(half_diffs_for, half_diffs_against), (goal_diffs_for, goal_diffs_against), (cards_diffs_for, cards_diffs_against)], ['Halftime split', 'First Goal Split', 'Dismissal split']):

    df_for =pd.DataFrame(diffs[0]).transpose()
    df_against =pd.DataFrame(diffs[1]).transpose()
    melted_for = pd.melt(df_for)
    melted_against = pd.melt(df_against)
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    col_line = [list(df[col]) for col in df.columns]
    minv = min(min(melted_for['value']), min(melted_against['value']))
    maxv = max(max(melted_for['value']), max(melted_against['value']))
    #col_line2 = [[abs(x) for x in list(df[col])] for col in df.columns]
    SPLIT_KIND = kind
    sns.boxplot(data=melted_for, x='variable', y='value', ax=ax[0], showfliers=False)
    sns.boxplot(data=melted_against, x='variable', y='value', ax=ax[1], showfliers=False)
    if kind == 'Dismissal split':
        ax[0].set_title(f'Between-{SPLIT_KIND} Difference: Team received card')
        ax[1].set_title(f'Between-{SPLIT_KIND} Difference: Opposition received card')
    else:
        ax[0].set_title(f'Between-{SPLIT_KIND} Difference: Team scored first')
        ax[1].set_title(f'Between-{SPLIT_KIND} Difference: Opposition scored first')
    ax[1].set_xlabel(r'Motif ordinal number $i$')
    #ax[0].set_xticklabels(['0a', '0b',] + list(range(1,14,1)))
    #ax[1].set_xticklabels(['0a', '0b',] + list(range(1,14,1)))
    ax[0].set_xlabel(r'Motif ordinal number $i$')
    ax[0].set_ylabel(f'Motif Significance difference \n((+) is before split)')
    ax[1].set_ylabel(f'Absolute Motif Significance difference')
    ax[0].set_ylim([minv, maxv])
    ax[1].set_ylim([minv, maxv])
    #ax[0].hlines(0, 0, 14, ls='--', color='black')
    #plt.suptitle(f'{kind}, pruned (median of each graph), 22 teams, {num_iters} random iters')
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(f'OVERALL SIGNIFICANCE DIFFERENCE box - {kind} - for-against.png', bbox_inches='tight')
    plt.show()
