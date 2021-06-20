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

diffs_half_for = {}
diffs_half = {}
diffs_goal_for = {}
diffs_card_for = {}

diffs_half_against = {}
diffs_goal_against = {}
diffs_card_against = {}
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
            l3 = os.path.join(l2, split_method)
            csvs = [x for x in os.listdir(l3) if (os.path.isfile(l3+'\\'+x) and x.split('.')[-1]=='csv') and 'pruned' in x]
            # print(f'split_method {split_method}')
            # print(f'team {team}')
            # print(f'match {match}')
            # print(len(list(csvs)))
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
            sns.lineplot(x=range(len(sums_0)), y=ppr0, label='Before split', ax=ax[0])
            sns.lineplot(x=range(len(sums_1)), y=ppr1, label='After split', ax=ax[0])
            plt.suptitle(f'{team} motif profile differences: {split_method} split')
            ax[0].set_title('Players-per-orbit representation')
            ax[0].set_xlabel(r'Orbit ordinal number $i$')
            ax[0].set_ylabel('Number of player occurences\n in orbit (role) PPO')
            ax[0].set_xticks(list(range(len(sums_0))))
            diff = [a-b for (a,b) in zip(ppr0, ppr1)]
            sns.lineplot(x=range(len(sums_0)), y=diff, label='Difference', ax=ax[1], color='red')
            ax[1].set_title('Orbit Difference')
            ax[1].set_xlabel(r'Orbit ordinal number $i$')
            ax[1].set_ylabel('difference between splits \n((+) if more in first)')
            ax[1].hlines(0, 0, 34, ls='--', color='black')
            plt.subplots_adjust(hspace=0.5)
            ax[1].set_xticks(list(range(len(sums_0))))
            if split_method == 'periods':
                diffs_half[(match, team, split_method)] =  diff
                if goal_for[:4] == team[:4]:
                    diffs_half_for[(match, team, split_method)] =  diff
                else:
                    diffs_half_against[(match, team, split_method)] =  diff
            elif split_method == 'goals':
                if goal_for[:4] == team[:4]:
                    diffs_goal_for[(match, team, split_method)] =  diff
                else:
                    diffs_goal_against[(match, team, split_method)] =  diff
            elif split_method == 'cards':
                if card_for[:4] == team[:4]:
                    diffs_card_for[(match, team, split_method)] =  diff
                else:
                    diffs_card_against[(match, team, split_method)] =  diff
            else: raise ValueError(f'The split method "{split_method}" for match {match}, team {team} is unknown.')
            plt.savefig(str(l3)+f'/OrbitalProfile_{split_method}', bbox_inches='tight')


df = pd.DataFrame(diffs_half).transpose()
print(df)
melted = pd.melt(df)
fig, ax = plt.subplots()
sns.boxplot(data=melted, x='variable', y='value', ax=ax, showfliers=False)
ax.set_xlabel(r'Orbit ordinal number $i$')
ax.set_ylabel('PPO difference \n((+) if more in before split')
ax.hlines(1, 0, 34, alpha=0.2, ls='--')
ax.hlines(-1, 0, 34, alpha=0.2, ls='--', label='1 PPO')
plt.legend()
plt.savefig(f'General HALFTIME Orbit profile differences.png', bbox_inches='tight')


for (diffs, kind) in zip([(diffs_half_for, diffs_half_against), (diffs_goal_for, diffs_goal_against), (diffs_card_for, diffs_card_against)], ['Halftime split', 'First Goal Split', 'Dismissal split']):
    df_for =pd.DataFrame(diffs[0]).transpose()
    df_against =pd.DataFrame(diffs[1]).transpose()
    melted_for = pd.melt(df_for)
    melted_against = pd.melt(df_against)
    minv = max(min(melted_for['value']), min(melted_against['value']))
    maxv = min(max(melted_for['value']), max(melted_against['value']))

    #melted2 = melted.copy()
    #melted2.value = [abs(val) for val in melted2.value]
    #print(df.columns)
    #fig, ax = plt.subplots(2,1)
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    #col_line = [list(df[col]) for col in df_for.columns]
    #col_line2 = [[abs(x) for x in list(df[col])] for col in df.columns]
    SPLIT_KIND = kind
    sns.boxplot(data=melted_for, x='variable', y='value', ax=ax[0], showfliers=False)
    sns.boxplot(data=melted_against, x='variable', y='value', ax=ax[1], showfliers=False)
    ax[0].set_xlabel(r'Orbit ordinal number $i$')
    ax[0].set_ylim([minv-0.5, maxv+0.5])
    ax[0].set_ylabel('PPO difference \n((+) if more in before split')
    ax[1].set_xlabel(r'Orbit ordinal number $i$')
    ax[1].set_ylabel('PPO difference \n((+) if more in before split')
    ax[1].set_ylim([minv-0.5, maxv+0.5])
    ax[0].hlines(1, 0, 34, alpha=0.2, ls='--')
    ax[0].hlines(-1, 0, 34, alpha=0.2, ls='--')
    ax[1].hlines(1, 0, 34, alpha=0.2, ls='--')
    ax[1].hlines(-1, 0, 34, alpha=0.2, ls='--')
    if kind == 'Dismissal split':
        ax[0].set_title(f'Between-{SPLIT_KIND} Difference: Team received card')
        ax[1].set_title(f'Between-{SPLIT_KIND} Difference: Opposition received card')
    else:
        ax[0].set_title(f'Between-{SPLIT_KIND} Difference: Team scored first')
        ax[1].set_title(f'Between-{SPLIT_KIND} Difference: Opposition scored first')

    #sns.boxplot(data=melted, x='variable', y='value', ax=ax[0], showfliers=False)
    #sns.boxplot(data=melted2, x='variable', y='value', ax=ax[1],showfliers=False)

    #sns.lineplot(x=list(df.columns), y=col_line2, ax=ax[1], color='blue')
    #ax[0].set_title(f'Orbit representation difference for {SPLIT_KIND} - pruned')
    #ax[1].set_title(f'Absolute orbit representation Difference {SPLIT_KIND} - pruned')
    #ax[1].set_xlabel('Orbit')
    #ax[0].set_xlabel('Orbit')
    #ax[0].set_ylabel('Number of \norbits more in 1st half')
    #ax[1].set_ylabel('Split Disagreement')
    #ax[0].hlines(0, 0, 33, ls='--', color='black')
    # if kind != 'Dismissal split':
    #     plt.suptitle(f'{kind}, pruned (median of each half), {17*2} teams')
    # else: plt.suptitle(f'{kind}, pruned (median of each half), {7*2} teams')
    plt.subplots_adjust(wspace=0.2)
    plt.savefig(f'General {kind} Orbit profile differences - foragainst.png', bbox_inches='tight')
    plt.show()

