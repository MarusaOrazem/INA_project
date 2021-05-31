import networkx as nx
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
plt.style.use('seaborn')

df0 = pd.read_csv('./output/cora_orbital_features_0.csv')
df1 = pd.read_csv('./output/cora_orbital_features_1.csv')
sums_0 = [sum(df0[role]) for role in df0.columns if role !='id']
sums_1 = [sum(df1[role]) for role in df1.columns if role !='id']
fig, ax = plt.subplots(2,1)
sns.lineplot(x=range(len(sums_0)), y=sums_0, label='First Half', ax=ax[0])
sns.lineplot(x=range(len(sums_1)), y=sums_1, label='Second Half', ax=ax[0])
plt.suptitle('Washington Spirit vs North Carolina Courage (2018)')
ax[0].set_title('Cummulative orbit representation')
ax[0].set_xlabel('Orbit')
ax[0].set_xlabel('Number of player occurences in orbit (role)')
ax[0].set_xticks(list(range(len(sums_0))))
diff = [b-a for (a,b) in zip(sums_0, sums_1)]

sns.lineplot(x=range(len(sums_0)), y=diff, label='Difference', ax=ax[1], color='red')
ax[1].set_title('Orbit Difference')
ax[1].set_xlabel('Orbit')
ax[1].set_xlabel('absolute difference between graphs')
ax[1].hlines(0, 0, 33, ls='--', color='black')
plt.subplots_adjust(hspace=0.5)
ax[1].set_xticks(list(range(len(sums_0))))

plt.show()