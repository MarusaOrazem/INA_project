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
import statistics
from statistics import mean
plt.style.use('seaborn')


p=Path(os.getcwd())

src = p.parent.parent

#G2 = nx.read_pajek(str(nets) + '\\nets\\7430\\7430_period_1.net')

odiffs = {}
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
            odiffs[(match, team, split_method)] =  diff
            plt.savefig(str(l3)+f'/MotifProfile_{split_method}', bbox_inches='tight')

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
#sns.lineplot(x=list(df.columns), y=col_line2, ax=ax[1], color='blue')
ax[0].set_title(f'Mean orbit Difference for {SPLIT_KIND} - Pruned')
ax[1].set_title(f'Mean Absolute orbit representation Difference {SPLIT_KIND} - pruned')
ax[1].set_xlabel('Orbit')
ax[0].set_xlabel('Orbit')
ax[0].set_ylabel('Number of \norbits more in 1st half')
ax[1].set_ylabel('Split Disagreement')
ax[0].hlines(0, 0, 33, ls='--', color='black')
plt.suptitle('halftime split, pruned (min median), 24 matches.')
plt.subplots_adjust(hspace=0.5)
plt.show()

