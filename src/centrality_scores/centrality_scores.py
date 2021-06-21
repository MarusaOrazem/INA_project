import networkx as nx
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
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
                    return a
            if event_name == 'period':
                if line.startswith('Home team'):
                    a = line[10:].strip().split('(')[0]
                    return a

def tops(G, centralities):
    nodes = list(G.nodes())
    nodes.sort(key = lambda node:centralities[node], reverse = True)
    print(f" top nodes: { nodes } ")

def calculate_centralies(G1, G2, match_id = "", event = ""):
    all_closeness = []
    all_betweeness = []

    for i, G in enumerate([G1,G2]):
        n = G.number_of_nodes()
        print(f'nodes: {n}')
        names = list(G.nodes())
        values_closeness = []
        values_betweeness = []

        # Closeness centrality
        closeness = nx.closeness_centrality(G)
        for node in names:
            if node == 'ON_goal' or node == 'OFF_goal':
                pass
            else:
                values_closeness.append(closeness[node])


        # Betweeness centrality
        betweeness = nx.betweenness_centrality(G, normalized=True)
        for node in names:
            if node == 'ON_goal' or node == 'OFF_goal':
                pass
            else:
                values_betweeness.append(betweeness[node])
        all_closeness.append(np.std(values_closeness))
        all_betweeness.append(np.std(values_betweeness))

    return all_closeness, all_betweeness



def get_scores(type_net, type, team):
    diff_c_all = []
    diff_b_all = []
    closeness_list = []
    betweeness_list = []
    i = 0
    data_dir = 'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/src/open-data/data/matches'

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
    for match_id in match_ids:
        path = f'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets/{match_id}'
        nets = [f for f in os.listdir(path) if f.endswith('.net')]
        # print(nets)
        # find matches with periods
        for name in nets:
            if i < 0:
                break
            try:
                if name.endswith(type_net):
                    i += 1
                    net1 = name
                    net2 = name[:-16] + '1' + name[-15:]
                    # net1='303524_Atlético Madrid (212)_period_0.net'
                    # net2='303524_Atlético Madrid (212)_period_1.net'
                    event_team = get_event_team(type, match_id)
                    print(event_team)
                    if team:
                        if event_team in name:
                            # if True:
                            G1 = nx.read_pajek(path + '/' + net1)
                            G1 = nx.DiGraph(G1)
                            G2 = nx.read_pajek(path + '/' + net2)
                            G2 = nx.DiGraph(G2)
                            if G2.number_of_nodes() == 0:
                                continue
                            if G1.number_of_nodes() == 0:
                                continue
                            c = calculate_centralies(G1, G2, match_id, name)
                            closeness, betweeness = c[0], c[1]
                            # print(closeness)
                            # print(betweeness)
                            closeness_diff = closeness[1] - closeness[0]
                            diff_c_all.append(closeness_diff)
                            closeness_list.append(closeness[0])
                            closeness_list.append(closeness[1])
                            betweeness_diff = betweeness[1] - betweeness[0]
                            diff_b_all.append(betweeness_diff)
                            betweeness_list.append(betweeness[0])
                            betweeness_list.append(betweeness[1])



                    elif type == 'Period':
                        G1 = nx.read_pajek(path + '/' + net1)
                        G1 = nx.DiGraph(G1)
                        G2 = nx.read_pajek(path + '/' + net2)
                        G2 = nx.DiGraph(G2)
                        if G2.number_of_nodes() == 0:
                            continue
                        if G1.number_of_nodes() == 0:
                            continue
                        c = calculate_centralies(G1, G2, match_id, name)
                        closeness, betweeness = c[0], c[1]
                        # print(closeness)
                        # print(betweeness)
                        closeness_diff = closeness[1] - closeness[0]
                        diff_c_all.append(closeness_diff)
                        closeness_list.append(closeness[0])
                        closeness_list.append(closeness[1])
                        betweeness_diff = betweeness[1] - betweeness[0]
                        diff_b_all.append(betweeness_diff)
                        betweeness_list.append(betweeness[0])
                        betweeness_list.append(betweeness[1])



                    else:
                        if event_team not in name:
                            # if True:
                            G1 = nx.read_pajek(path + '/' + net1)
                            G1 = nx.DiGraph(G1)
                            G2 = nx.read_pajek(path + '/' + net2)
                            G2 = nx.DiGraph(G2)
                            if G2.number_of_nodes() == 0:
                                continue
                            if G1.number_of_nodes() == 0:
                                continue
                            c = calculate_centralies(G1, G2, match_id, name)
                            closeness, betweeness = c[0], c[1]
                            # print(closeness)
                            # print(betweeness)
                            closeness_diff = closeness[1] - closeness[0]
                            diff_c_all.append(closeness_diff)
                            closeness_list.append(closeness[0])
                            closeness_list.append(closeness[1])
                            betweeness_diff = betweeness[1] - betweeness[0]
                            diff_b_all.append(betweeness_diff)
                            betweeness_list.append(betweeness[0])
                            betweeness_list.append(betweeness[1])


            except Exception as e:
                print(e)
                pass
    return diff_b_all, betweeness_list

if __name__ == "__main__":
    '''
    #matches_ids = [303524, 303610, 303470, 18237, 7584, 7545, 7567, 303715, 22912, 2302764, 7565, 7542]
    match_id = 9736
    path = f'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets/{match_id}'
    nets = [f for f in os.listdir(path) if f.endswith('.net')]
    for net in nets:
        if 'period_0' in net and 'with_goals' not in net:
            if 'Barcelona' in net:
                G1 = nx.read_pajek(path + '/' + net)
                G1 = nx.DiGraph(G1)
                b1 = nx.betweenness_centrality(G1)
        if 'period_1' in net and 'with_goals' not in net:
            if 'Barcelona' in net:
                G2 = nx.read_pajek(path + '/' + net)
                G2 = nx.DiGraph(G2)
                b2 = nx.betweenness_centrality(G2)

    #c = calculate_centralies(G1,G2)
    plt.style.use('seaborn')
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    plt.style.use('seaborn')
    n = ['first', 'second']
    t = [b1,b2]
    for i, cc in enumerate(t):
        sns.boxplot(ax=axes[i], data=pd.DataFrame({n[i]: cc}))
    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Betweness centralities')
    plt.title('Barcelona period split')
    plt.savefig('barcelona_period.png')
    plt.show()'''


    diff_goal = get_scores('goal_0_with_goals.net', 'Goal', True)
    diff_goal_opp = get_scores('goal_0_with_goals.net', 'Goal', False)
    diff_card = get_scores('card_0_with_goals.net', 'Card', True)
    diff_card_opp = get_scores('card_0_with_goals.net', 'Card', False)
    diff_period = get_scores('period_0_with_goals.net', 'Period', False)

    means = []
    lower = []
    upper = []
    all_diff = [diff_goal, diff_goal_opp, diff_card, diff_card_opp, diff_period]

    plt.style.use('seaborn')
    plt.ylabel('Share of difference in STD changed, (+) if more in first')
    plt.xlabel('Type of event')
    fig, axes = plt.subplots(1, 5, figsize=(15, 5), sharey=True)
    plt.style.use('seaborn')

    n = ['goal team', 'goal opp team', 'card team', 'card opp team', 'period']
    for i, cc in enumerate(all_diff):
        sns.boxplot(ax=axes[i], data = pd.DataFrame({n[i]: cc[0]/np.mean(cc[1])}))

    fig.add_subplot(111, frame_on=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.ylabel('Share of change in STD of closeness centralities,\n ((+) if more in first)')
    plt.xlabel('Type of event')
    plt.savefig('flow_final.png')
    plt.show()
