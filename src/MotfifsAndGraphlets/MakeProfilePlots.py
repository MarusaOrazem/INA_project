import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import os
from pathlib import Path
from matplotlib import pyplot as plt
from texttable import Texttable
from tqdm import tqdm
import pandas as pd
import warnings
import statistics
from statistics import mean
plt.style.use('seaborn')


p=Path(os.getcwd())

src = p.parent.parent

#G2 = nx.read_pajek(str(nets) + '\\nets\\7430\\7430_period_1.net')

diffs_half = {}
diffs_goal = {}
diffs_card = {}
root = os.path.join(src, 'chosen_nets')
for match in next(os.walk(root))[1]:
    l1 = os.path.join(root, match)
    for team in next(os.walk(l1))[1]:
        l2 = os.path.join(l1, team)
        for split_method in next(os.walk(l2))[1]:
            l3 = os.path.join(l2, split_method)
            csvs = [x for x in os.listdir(l3) if (os.path.isfile(l3+'\\'+x) and x.split('.')[-1]=='csv') and 'pruned' in x]
            assert len(list(csvs)) == 2
            dfs = []
            for csv in csvs:
                df = pd.read_csv(str(l3)+'/'+csv)
                dfs.append(df)

            ###only do sums
            sums_0 = [sum(dfs[0][role]) for role in dfs[0].columns if role !='id']
            sums_1 = [sum(dfs[1][role]) for role in dfs[1].columns if role !='id']
            ppr0 = [sum(dfs[0][role])/11 for role in dfs[0].columns if role !='id']
            ppr1 = [sum(dfs[1][role])/11 for role in dfs[1].columns if role !='id']
            fig, ax = plt.subplots(2,1)
            sns.lineplot(x=range(len(sums_0)), y=ppr0, label='First Half', ax=ax[0])
            sns.lineplot(x=range(len(sums_1)), y=ppr1, label='Second Half', ax=ax[0])
            plt.suptitle(f'{team} motif profile differences: {split_method} split')
            ax[0].set_title('Players-per-orbit representation')
            ax[0].set_xlabel('Orbit')
            ax[0].set_ylabel('Number of player occurences\n in orbit (role) PPO')
            ax[0].set_xticks(list(range(len(sums_0))))
            diff = [b-a for (a,b) in zip(ppr0, ppr1)]
            sns.lineplot(x=range(len(sums_0)), y=diff, label='Difference', ax=ax[1], color='red')
            ax[1].set_title('Orbit Difference')
            ax[1].set_xlabel('Orbit')
            ax[1].set_ylabel('difference between graphs \n(in second half)')
            ax[1].hlines(0, 0, 33, ls='--', color='black')
            plt.subplots_adjust(hspace=0.5)
            ax[1].set_xticks(list(range(len(sums_0))))
            if split_method == 'periods':
                diffs_half[(match, team, split_method)] =  diff
            elif split_method == 'goals':
                diffs_goal[(match, team, split_method)] =  diff
            elif split_method == 'cards':
                diffs_card[(match, team, split_method)] =  diff
            else: raise ValueError(f'The split method "{split_method}" for match {match}, team {team} is unknown.')
            plt.savefig(str(l3)+f'/MotifProfile_{split_method}', bbox_inches='tight')

for (diffs, kind) in zip([diffs_half, diffs_goal, diffs_card], ['Halftime split', 'First Goal Split', 'Dismissal split']):
    df =pd.DataFrame(diffs).transpose()
    melted = pd.melt(df)
    melted2 = melted.copy()
    melted2.value = [abs(val) for val in melted2.value]
    print(df.columns)
    fig, ax = plt.subplots(2,1)
    col_line = [list(df[col]) for col in df.columns]
    col_line2 = [[abs(x) for x in list(df[col])] for col in df.columns]
    SPLIT_KIND = kind
    sns.boxplot(data=melted, x='variable', y='value', ax=ax[0], showfliers=False)
    sns.boxplot(data=melted2, x='variable', y='value', ax=ax[1],showfliers=False)
    #sns.lineplot(x=list(df.columns), y=col_line2, ax=ax[1], color='blue')
    ax[0].set_title(f'Mean orbit Difference for {SPLIT_KIND} - pruned')
    ax[1].set_title(f'Mean Absolute orbit representation Difference {SPLIT_KIND} - pruned')
    ax[1].set_xlabel('Orbit')
    ax[0].set_xlabel('Orbit')
    ax[0].set_ylabel('Number of \norbits more in 1st half')
    ax[1].set_ylabel('Split Disagreement')
    #ax[0].hlines(0, 0, 33, ls='--', color='black')
    if kind != 'Dismissal split':
        plt.suptitle(f'{kind}, pruned (median of each half), {17*2} teams')
    else: plt.suptitle(f'{kind}, pruned (median of each half), {7*2} teams')
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f'General {kind} Orbit profile differences.png', bbox_inches='tight')
    plt.show()

