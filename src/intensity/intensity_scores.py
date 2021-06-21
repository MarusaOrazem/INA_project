import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd
import json
import numpy as np
import scipy.stats as st
import seaborn as sns
from matplotlib.legend import Legend
import matplotlib.patches as mpatches

def get_event_team(event_name, match_id):
    with open(f'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets/{match_id}/{match_id}.txt') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(event_name):
                for i in range(len(line)):
                    a = line.split('for')[-1].strip().split("'")[1]
                    if a == "Germany Women's ":
                        b = 3
                    return a
            if event_name == 'period':
                if line.startswith('Home team'):
                    a = line[10:].strip().split('(')[0]
                    if a == "Germany Women's ":
                        b = 3
                    return a



def intensity_scores(G, time):
    sum_weights = 0
    for i in list(G.nodes()):
        for j in list(G.successors(i)):
            sum_weights = int(G.get_edge_data(i,j)['weight'])
    intensity = 1/time *sum_weights
    return intensity

def get_time(team, match_id, name):
    a=3
    if name.endswith('card_0.net') or name.endswith('card_1.net'):
        q1 = 'Before card'
        q2 = 'After card'

    elif name.endswith('period_0.net') or name.endswith('period_1.net'):
        q1 = 'First period'
        q2 = 'Second period'

    elif name.endswith('goal_0.net') or name.endswith('goal_1.net'):
        q1 = 'Before goal'
        q2 = 'After goal'
    else:
        print('Not implemented')
        return -1

    times = [-1, -1]

    with open(f'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets/{match_id}/{match_id}.txt', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith(q1):
                if team in line:
                    times[0] = float(line.strip()[-6:])
            elif line.startswith(q2):
                if team in line:
                    times[1] = float(line.strip()[-6:])
    return times


def get_scores(type_net, type, team):
    # matches_ids = [303524, 303610, 303470, 18237, 7584, 7545, 7567, 303715, 22912, 2302764, 7565, 7542]
    data_dir = 'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/src/open-data/data/matches'
    nets_dir = 'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets'
    # nets_dir =  'C:/Users/Acer/Desktop/frizura'
    all_matches = []
    match_ids = []
    for competition_id in os.listdir(data_dir):
        # print(competition_id)
        for season_id in os.listdir(data_dir + '/' + competition_id):
            # print(season_id)
            with open(
                    f'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/src/open-data/data/matches/{competition_id}/{season_id}',
                    encoding='utf-8') as f:
                matches = json.load(f)
                for m in matches:
                    match_id = m['match_id']
                    match_ids.append(match_id)

    firsts = []
    seconds = []
    for match_id in match_ids:
        path = f'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets/{match_id}'
        nets = [f for f in os.listdir(path) if f.endswith('.net')]
        # print(nets)
        # find matches with periods
        for name in nets:
            try:
                if name.endswith(type_net):
                    print('---------')
                    event_team = get_event_team(type, match_id)
                    a = 3
                    print(event_team)
                    if team:
                        if not event_team in name:
                            # if True:
                            net1 = name
                            net2 = name[:-5] + '1' + name[-4:]
                            print(name)
                            G1 = nx.read_pajek(path + '/' + net1)
                            G1 = nx.DiGraph(G1)
                            G2 = nx.read_pajek(path + '/' + net2)
                            G2 = nx.DiGraph(G2)
                            if G2.number_of_nodes() == 0:
                                continue

                            time_home = get_time('home', match_id, name)
                            print(time_home)
                            if time_home == -1 or time_home == [-1, -1]:
                                print('NO TIME')
                                continue

                            s1 = intensity_scores(G1, time_home[0])
                            s2 = intensity_scores(G2, time_home[1])

                            print(s1)
                            print(s2)
                            firsts.append(s1)
                            seconds.append(s2)
                    elif type == 'Period':
                        net1 = name
                        net2 = name[:-5] + '1' + name[-4:]
                        print(name)
                        G1 = nx.read_pajek(path + '/' + net1)
                        G1 = nx.DiGraph(G1)
                        G2 = nx.read_pajek(path + '/' + net2)
                        G2 = nx.DiGraph(G2)
                        if G2.number_of_nodes() == 0:
                            continue

                        time_home = get_time('home', match_id, name)
                        print(time_home)
                        if time_home == -1 or time_home == [-1, -1]:
                            print('NO TIME')
                            continue

                        s1 = intensity_scores(G1, time_home[0])
                        s2 = intensity_scores(G2, time_home[1])

                        print(s1)
                        print(s2)
                        firsts.append(s1)
                        seconds.append(s2)
                    else:
                        if event_team in name:
                            # if True:
                            net1 = name
                            net2 = name[:-5] + '1' + name[-4:]
                            print(name)
                            G1 = nx.read_pajek(path + '/' + net1)
                            G1 = nx.DiGraph(G1)
                            G2 = nx.read_pajek(path + '/' + net2)
                            G2 = nx.DiGraph(G2)
                            if G2.number_of_nodes() == 0:
                                continue

                            time_home = get_time('home', match_id, name)
                            print(time_home)
                            if time_home == -1 or time_home == [-1, -1]:
                                print('NO TIME')
                                continue

                            s1 = intensity_scores(G1, time_home[0])
                            s2 = intensity_scores(G2, time_home[1])

                            print(s1)
                            print(s2)
                            firsts.append(s1)
                            seconds.append(s2)

            except:
                a = 3
                pass

    return firsts, seconds


if __name__ == "__main__":

    goal_team = get_scores('goal_0.net', 'Goal', True)
    goal_opp_team = get_scores('goal_0.net', 'Goal', False)
    card_team = get_scores('card_0.net', 'Card', True)
    card_opp_team = get_scores('card_0.net', 'Card', False)
    period = get_scores('period_0.net', 'Period', False)

    all_list = [goal_team, goal_opp_team, card_team, card_opp_team, period]
    all_list_names = ['goal team', 'goal opp team', 'card team', 'card opp team', 'period']
    all_diff = []
    for index, i in enumerate(all_list):
        diff = np.subtract(i[0],i[1])/np.mean(np.add(i[0],i[1]))
        all_diff.append(diff)

    means = []
    lower = []
    upper = []
    for i, c in enumerate(all_diff):
        m = np.mean(c)
        means.append(m)
        interval = st.t.interval(0.95, len(c) - 1, loc=np.mean(c), scale=st.sem(c))
        lower.append(m - interval[0])
        upper.append(interval[1] - m)

    plt.style.use('seaborn')
    plt.bar(['goal team', 'goal opp team', 'card team', 'card opp team', 'period'], means, yerr=[lower, upper])
    plt.ylabel('Share of intensity changed,\n ((+) if more in second)')
    plt.xlabel('Type of event')
    #sns.barplot(data=df)
    plt.savefig("final.png")

    plt.show()



