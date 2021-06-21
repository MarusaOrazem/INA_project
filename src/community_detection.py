import networkx as nx
from cdlib.algorithms import louvain, leiden, greedy_modularity, aslpaw, cpm, hierarchical_link_community
from cdlib.evaluation import normalized_mutual_information, variation_of_information
from cdlib import NodeClustering
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import copy
import seaborn as sns
import traceback

plt.style.use('seaborn')

game_ids = [
    '9620',
    '303524',
    '303610',
    '303470',
    '18237',
    '7584',
    '7545',
    '303715',
    '7530',
    '7544',
    '7565',
    '7542'
]

red_card_games = [
    '9717', # red card for Villeared, home team
    '9736', # red card for Real Madrid, home team
    '15986', # red card for barcelona, home team
    '303615',
    '19813', # red card for West ham, home team
    '2275030', # red card for Brighton & HOve, home team
    '7558', # red card for Russia, guest team
    '7551', # red card for Germany, home team
]

# 2 lines for each game, 1 for one team, and the other one for second team
communities_by_position = [
    {'home_team': [[8], [3, 6, 7], [1, 2, 4, 5, 10, 11], [9]], 'away_team': [[6], [1, 2, 4, 5], [3, 9, 10, 11], [7, 8]]},
    {'home_team': [[6], [1, 5, 7, 8], [2, 3, 10, 11], [4, 9]], 'away_team': [[1], [2, 4, 5, 11], [3, 8 ,10], [6, 7, 9]]},
    {'home_team': [[5], [3, 4, 6], [1, 2, 7, 8, 11], [9, 10]], 'away_team': [[4], [2, 3, 5, 6], [7, 8, 9, 10, 11], [1]]},
    {'home_team': [[8], [3, 4, 5, 6], [1, 2, 7, 9, 10], [11]], 'away_team': [[8], [3, 5, 6, 7], [2, 4, 9, 10], [1, 11]]},
    {'home_team': [[3], [2, 4, 6, 7], [1, 5, 8, 9, 11], [10]], 'away_team': [[5], [4, 6, 7, 9], [1, 3, 8, 10, 11], [2]]},
    {'home_team': [[6], [3, 4, 5], [2, 7, 8, 11], [1, 9, 10]], 'away_team': [[9], [3, 4, 5, 6], [2, 7, 8, 10, 11], [1]]},
    {'home_team': [[7], [4, 5, 6], [1, 2, 3, 8, 10, 11], [9]], 'away_team': [[7], [2, 3, 4, 5], [1, 6, 8, 9, 10], [11]]},
    {'home_team': [[6], [1, 2, 3, 7], [5, 8, 9, 10, 11], [4]], 'away_team': [[6], [1, 3, 5, 7, 10], [4, 8, 9, 11], [2]]},
    {'home_team': [[2], [1, 3, 4, 8], [5, 6, 11], [7, 9, 10]], 'away_team': [[6], [2, 3, 5, 7], [1, 4, 8, 10, 11], [9]]},
    {'home_team': [[1], [2, 4, 5, 6], [3, 9, 10, 11], [7, 8]], 'away_team': [[7], [2, 8, 4, 6], [3, 5, 1, 9, 10], [11]]},
    {'home_team': [[8], [2, 3, 4, 7], [1, 5, 6, 9, 10], [11]], 'away_team': [[4], [3, 9, 7, 8], [2, 10, 11], [5, 6, 1]]},
    {'home_team': [[8], [1, 4, 6, 7 ], [5, 9, 10, 11], [2, 3]], 'away_team': [[8], [4, 7, 2, 5], [1, 3, 6, 10, 11], [9]]},
]

rc_communities_by_position = [
    {'home_team': [[10], [1, 7, 9, 4], [3, 5, 6, 8, 11], [2]], 'away_team': [[8], [5, 6, 4, 9], [2, 3, 7, 11], [1, 10]]},
    {'home_team': [[4], [1, 2, 5, 9], [3, 7, 8, 10], [6, 11]], 'away_team': [[7], [3, 4, 5, 9], [2, 6, 8, 10], [1, 11]]},
    {'home_team': [[11], [4, 8, 9, 10], [2, 3, 6], [1, 5, 7]], 'away_team': [[9], [1, 3, 5, 8, 10], [2, 6, 11], [4, 7]]},
    {'home_team': [[10], [2, 4, 5, 11], [1, 3, 8, 9], [6, 7]], 'away_team': [[9], [4, 6, 8, 7, 11], [2, 3, 10], [5, 1]]},
    {'home_team': [[2], [1, 3, 6, 11], [4, 7, 10], [5, 8, 9]], 'away_team': [[4], [3, 5, 6, 7], [2, 8, 9, 10, 11], [1]]},
    {'home_team': [[6], [1, 8, 10, 11], [2, 3, 5, 9], [4, 7]], 'away_team': [[6], [3, 7, 9, 11], [2, 4, 5, 8, 10], [1]]},
    {'home_team': [[1], [2, 3, 4, 5], [6, 7, 9, 11], [8, 10]], 'away_team': [[5], [2, 4, 9, 11], [1, 6, 7, 8, 10], [3]]},
    {'home_team': [[9], [2, 3, 4, 5], [1, 6, 7, 10, 11], [8]], 'away_team': [[6], [2, 3, 7, 8], [1, 5, 9, 10], [4, 11]]}
]

rc_games = ['9717', '9736', '9924', '266440', '267590', '267492', '69224', '69262', '69240', '69213', '69270', '69245', '69263', '70219', '69325', '69319', '70260', '70287', '266916', '266770', '267561', '266794', '266921', '266653', '266986', '68354', '68351', '68325', '69176', '68334', '68335', '68332', '68347', '16095', '69139', '69228', '69211', '303615']
rc_events = ['h', 'h', 'h', 'a', 'a', 'a', 'a', 'h', 'h', 'a', 'a', 'h', 'h', 'h', 'a', 'a', 'a', 'a', 'a', 'h', 'h', 'h', 'h', 'h', 'h', 'a', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'a', 'h', 'h']

ggames = ['9592', '9870', '9783', '9700', '9860', '9695', '9717', '9673', '9620', '9827', '9837', '9642', '9602', '9948', '9682', '9581', '9726', '9754', '9575', '9765', '9889', '9609', '9636', '9661', '9736', '9799', '9924', '267212', '267220', '266989', '266357', '267039', '267058', '266477', '266440', '267660', '266273', '266874', '266731', '266280', '267076', '267590', '267373', '267077', '266952', '267464', '266299', '266191', '267492', '267569', '69243', '69257', '69253', '69244', '69277', '69229', '69219', '69218', '69250', '69242', '69256', '69298', '69221', '69259', '69224', '69210', '69220', '69237']
gevents = ['h', 'h', 'h', 'a', 'h', 'h', 'a', 'h', 'a', 'a', 'h', 'a', 'h', 'h', 'h', 'a', 'a', 'h', 'h', 'h', 'a', 'a', 'h', 'h', 'a', 'a', 'a', 'h', 'h', 'a', 'a', 'h', 'h', 'h', 'a', 'a', 'h', 'h', 'a', 'h', 'a', 'h', 'a', 'h', 'h', 'h', 'h', 'a', 'h', 'a', 'h', 'h', 'h', 'h', 'h', 'h', 'h', 'a', 'a', 'a', 'a', 'a', 'h', 'h', 'h', 'a', 'h', 'a', 'h', 'a', 'a', 'h']



def read_team_names(id):
    home = ''
    guests = ''

    with open('../nets/' + id + '/' + id + '.txt', 'r', encoding="utf-8") as File:
        line_count = 1
        for line in File:
            if line_count == 6:
                line_data = line.split(':')
                team_name = line_data[1]
                team_name = team_name[:team_name.index(',')]
                home = team_name[1:]
            elif line_count == 7:
                line_data = line.split(':')
                team_name = line_data[1]
                team_name = team_name[:team_name.index(',')]
                guests = team_name[1:]
            line_count += 1


    return home, guests


def who_did_what(event):
    # games = extract_games(100)
    event_data = []

    for id in rc_games:
        home_team, away_team = read_team_names(id)
        with open('../nets/' + id + '/' + id + '.txt', 'r', encoding="utf-8") as File:
            line_count = 1
            for line in File:
                if line_count == 8:
                    if event == 'goal':
                        line_segment = line.split(',')[3]
                        team = line_segment.replace(')', '')
                        team = team.replace("'", '')
                        team = team.strip()
                        # print(team)
                        if team in home_team:
                            event_data.append('h')
                        else:
                            event_data.append('a')
                if line_count == 9:
                    if event == 'card':
                        if len(line.split(',')) > 3:
                            line_segment = line.split(',')[3]
                            team = line_segment.replace(')', '')
                            team = team.replace("'", '')
                            team = team.strip()
                            # print(team)
                            if team in home_team:
                                event_data.append('h')
                            else:
                                event_data.append('a')


                line_count += 1


    return event_data

def read_game_result(id):

    with open('../nets/' + id + '/' + id + '.txt', 'r', encoding="utf-8") as File:
        line_count = 1
        hgoal = ''
        agoal = ''
        for line in File:
            if line_count == 6:
                line_data = line.split(':')
                hgoal = int(line_data[2])
            elif line_count == 7:
                line_data = line.split(':')
                agoal = int(line_data[2])
            line_count += 1


    return 'Result: ' + str(hgoal) + '-' + str(agoal)


def read_weighted_edge_list(file):
    nodes = list()
    node_names = list()
    edges = list()
    weights = list()
    reached_nodes_line = False
    reached_edges_line = False

    with open(file, 'r', encoding="utf-8") as File:
        line_count = 1
        for line in File:
            if line_count == 2 and not reached_nodes_line:
                reached_nodes_line = True
            if reached_nodes_line and len(line.split(' ')) > 2:
                line_data = line.split(' ')
                # print(line_data)
                nodes.append(line_data[0])
                cut = line_data[:-3]
                cut = cut[1:]
                node_names.append(line_data[0] + ' - ' + ' '.join(cut))
            if reached_edges_line == False and line.startswith('*arcs'):
                reached_edges_line = True
                reached_nodes_line = False
            if reached_edges_line and not line.startswith('*arcs'):
                elements = line.split(' ')
                edges.append([int(elements[0]), int(elements[1])])
                weights.append(int(elements[2]))
            line_count += 1

        return nodes, node_names, edges, weights

def parse_weighted_graph(file, with_node_names = True):
    nodes, node_names, edges, weights = read_weighted_edge_list(file)
    G = nx.DiGraph()
    for i, edge in enumerate(edges):
        if with_node_names == True:
            G.add_edge(node_names[int(edge[0]) - 1], node_names[int(edge[1]) - 1], weight = weights[i])
        else:
            G.add_edge(nodes[int(edge[0]) - 1], nodes[int(edge[1]) - 1], weight = weights[i])


    return G, weights

def normalize_array(arr):
    narr = list()
    maks = max(arr)
    for i, a in enumerate(arr):
        narr.append((a / maks) * 5)

    return narr

def recreate_graph_from_communities(g, communities):
    if len(communities) > 1:
        for i, comm in enumerate(communities[:-1]):
            for node in comm:
                for onode in communities[i+1]:

                    if g.has_edge(node, onode):
                        print('Edge removed')
                        g.remove_edge(node, onode)
                    if g.has_edge(onode, node):
                        print('reverse Edge removed')
                        g.remove_edge(onode, node)
        return g
    else:
        return g


def extract_games(n):
    collected_games = 0
    ids = []
    with open('../all_matches.txt', 'r', encoding="utf-8") as File:
        for line in File:
            id = str(int(line.split(',')[0].split(':')[1]))
            ids.append(id)
            collected_games += 1
            if collected_games >= n:
                break

        print(ids)

def extract_redcard_games(n):
    collected_games = 0
    ids = []
    with open('../all_matches.txt', 'r', encoding="utf-8") as File:
        for line in File:
            if 'Has dismissal: True' in line:
                id = str(int(line.split(',')[0].split(':')[1]))
                ids.append(id)
                collected_games += 1
                if collected_games >= n:
                    break

        print(ids)

def extract_goal_games(n):
    collected_games = 0
    ids = []
    with open('../all_matches.txt', 'r', encoding="utf-8") as File:
        for line in File:
            if line.count('goals: 0') < 2:
                id = str(int(line.split(',')[0].split(':')[1]))
                ids.append(id)
                collected_games += 1
                if collected_games >= n:
                    break

        print(ids)

def read_halftime_split_for_communities():
    team_h1_comm = []
    team_h2_comm = []
    team_a1_comm = []
    team_a2_comm = []
    teams = []
    game_results = []
    # algorithms = [leiden]

    for game_id in game_ids + red_card_games:
        home_team, away_team = read_team_names(game_id)
        gs = read_game_result(game_id)
        game_results.append(gs)
        game_results.append(gs)
        home_1hg, home_1hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_period_0.net')
        away_1hg, away_1hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_period_0.net')
        home_2hg, home_2hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_period_1.net')
        away_2hg, away_2hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_period_1.net')

        teams.append(home_team)
        teams.append(away_team)

        detected_communities = leiden(home_1hg, weights=home_1hweights)
        team_h1_comm.append(len(detected_communities.communities))

        detected_communities = leiden(away_1hg, weights=away_1hweights)
        team_a1_comm.append(len(detected_communities.communities))

        detected_communities = leiden(home_2hg, weights=home_2hweights)
        team_h2_comm.append(len(detected_communities.communities))

        detected_communities = leiden(away_2hg, weights=away_2hweights)
        team_a2_comm.append(len(detected_communities.communities))

    cms = pd.DataFrame({'first_half_comms': team_h1_comm + team_a1_comm,
                                     'second_half_comms': team_h2_comm + team_a2_comm},
									 )
    diffs = pd.DataFrame({'differences': cms['first_half_comms'] - cms['second_half_comms']})
    fig, ax = plt.subplots()
    ax.set_title('Boxplot for halftime split')
    ax.boxplot(diffs)
    # plt.xticks([1, 2], ['1st halftime', '2nd halftime'])
    plt.savefig('ht_boxplot.png')
    plt.show()
    # first_half_comms = pd.DataFrame({'team': teams,
    #                                  'first_half_comms': comm_1h_number,
    #                                  'second_half_team': comm_2h_number,
    #                                  'game_results': game_results})
    # first_half_comms.to_csv('half_split_communities.csv', index=False)

def read_goal_split_for_communities():
    team_h1_comm = []
    team_h2_comm = []
    team_a1_comm = []
    team_a2_comm = []
    teams = []
    game_results = []
    # algorithms = [leiden]

    for game_id in game_ids + red_card_games:
        home_team, away_team = read_team_names(game_id)
        gs = read_game_result(game_id)
        game_results.append(gs)
        game_results.append(gs)
        home_1hg, home_1hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_goal_0.net')
        away_1hg, away_1hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_goal_0.net')
        home_2hg, home_2hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_goal_1.net')
        away_2hg, away_2hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_goal_1.net')

        teams.append(home_team)
        teams.append(away_team)

        detected_communities = leiden(home_1hg, weights=home_1hweights)
        team_h1_comm.append(len(detected_communities.communities))

        detected_communities = leiden(away_1hg, weights=away_1hweights)
        team_a1_comm.append(len(detected_communities.communities))

        detected_communities = leiden(home_2hg, weights=home_2hweights)
        team_h2_comm.append(len(detected_communities.communities))

        detected_communities = leiden(away_2hg, weights=away_2hweights)
        team_a2_comm.append(len(detected_communities.communities))

    cms = pd.DataFrame({'first_goal_comms': team_h1_comm + team_a1_comm,
                                     'second_goal_comms': team_h2_comm + team_a2_comm},
									 )
    diffs = pd.DataFrame({'differences': cms['first_goal_comms'] - cms['second_goal_comms']})
    fig, ax = plt.subplots()
    ax.set_title('Boxplot for goal split')
    ax.boxplot(diffs)
    # plt.xticks([1, 2], ['1st halftime', '2nd halftime'])
    plt.savefig('goal_boxplot.png')
    plt.show()
    # goal_split_cooms = pd.DataFrame({'team': teams,
    #                                  'before_first_goal_comms': comm_1h_number,
    #                                  'after_first_goal_comms': comm_2h_number,
    #                                  'game_results': game_results})
    # goal_split_cooms.to_csv('goal_split_communities.csv', index=False)


def read_redcard_split_for_communities():
    team_recieving_card_t1_comm = []
    team_recieving_card_t2_comm = []
    team_opp_card_t1_comm = []
    team_opp_card_t2_comm = []
    teams = []
    game_results = []
    # algorithms = [leiden]

    for i, game_id in enumerate(rc_games):
        home_team, away_team = read_team_names(game_id)
        gs = read_game_result(game_id)
        game_results.append(gs)
        game_results.append(gs)
        home_1hg, home_1hweights = parse_weighted_graph(
            '../nets/' + game_id + '/' + game_id + '_' + home_team + '_card_0.net')
        away_1hg, away_1hweights = parse_weighted_graph(
            '../nets/' + game_id + '/' + game_id + '_' + away_team + '_card_0.net')
        home_2hg, home_2hweights = parse_weighted_graph(
            '../nets/' + game_id + '/' + game_id + '_' + home_team + '_card_1.net')
        away_2hg, away_2hweights = parse_weighted_graph(
            '../nets/' + game_id + '/' + game_id + '_' + away_team + '_card_1.net')

        teams.append(home_team)
        teams.append(away_team)

        try:
            detected_communities = leiden(home_1hg, weights=home_1hweights)
            if rc_events[i] == 'h':
                team_recieving_card_t1_comm.append(len(detected_communities.communities))
            else:
                team_opp_card_t1_comm.append(len(detected_communities.communities))

            detected_communities = leiden(away_1hg, weights=away_1hweights)
            if rc_events[i] == 'h':
                team_opp_card_t1_comm.append(len(detected_communities.communities))
            else:
                team_recieving_card_t1_comm.append(len(detected_communities.communities))

            detected_communities = leiden(home_2hg, weights=home_2hweights)
            if rc_events[i] == 'h':
                team_recieving_card_t2_comm.append(len(detected_communities.communities))
            else:
                team_opp_card_t2_comm.append(len(detected_communities.communities))

            detected_communities = leiden(away_2hg, weights=away_2hweights)
            if rc_events[i] == 'h':
                team_opp_card_t2_comm.append(len(detected_communities.communities))
            else:
                team_recieving_card_t2_comm.append(len(detected_communities.communities))

        except:
            print('Game id for which it was not possible: ' + str(game_id))

    print()
    cms = pd.DataFrame({'recieving_card_t1': team_recieving_card_t1_comm,
                        'recieving_card_t2': team_recieving_card_t2_comm,
                        'opp_card_t1': team_opp_card_t1_comm,
                        'opp_card_t2': team_opp_card_t2_comm,
                        }, )

    diffs = pd.DataFrame({'receiving_team': cms['recieving_card_t1'] - cms['recieving_card_t2'],
                          'opp_team': cms['opp_card_t1'] - cms['opp_card_t2']})
    fig, ax = plt.subplots()
    ax.set_title('Boxplot for red card split')
    ax.boxplot(diffs)
    plt.xticks([1, 2], ['Team recieving red card', 'Opposing team'])
    plt.savefig('card_boxplot_2_teams.png')
    plt.show()

    # red_card_splits = pd.DataFrame({'team': teams,
    #                                  'before_red_card_comms': comm_1h_number,
    #                                  'after_red_card_comms': comm_2h_number,
    #                                  'game_results': game_results})
    # red_card_splits.to_csv('red_card_split_communities.csv', index=False)

def transform_communities_to_int(communities):
    ncoms = []
    for i,com in enumerate(communities):
        c = []
        for member in com:
            c.append(int(member))

        ncoms.append(c)

    return ncoms

def transform_communities_to_str(communities):
    ncoms = []
    for i,com in enumerate(communities):
        c = []
        for member in com:
            c.append(str(member))

        ncoms.append(c)

    return ncoms

def add_missing_node_as_separate_community(communities):
    members = []
    for comm in communities:
        for c in comm:
            members.append(c)

    for n in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11']:
        if n not in members:
            communities.append([n])

    return communities

def see_community_changes(type, alg):
    # type can be 'period', 'goal', 'card'
    filenamestr = ''

    the_ids = game_ids
    if type == 'period':
        filenamestr = 'half'
    else:
        if type == 'card':
            the_ids = red_card_games
        filenamestr = type

    hteam_h1_nmi = []
    hteam_h2_nmi = []
    ateam_h1_nmi = []
    ateam_h2_nmi = []


    home_team_communities_after = []
    away_team_communities_after = []

    hometeam_1h_vs_2h_nmis = []
    awayteam_1h_vs_2h_nmis = []
    for i,game_id in enumerate(the_ids):
        home_team, away_team = read_team_names(game_id)
        print('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_0.net')
        home_1hg, home_1hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_0.net', with_node_names=False))
        away_1hg, away_1hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_' + type + '_0.net', with_node_names=False))
        home_2hg, home_2hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_1.net', with_node_names=False))
        away_2hg, away_2hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_' + type + '_1.net', with_node_names=False))

        try:
            detected_communities_hh1 = alg(home_1hg, weights=home_1hweights)
            detected_communities_hh1.communities == add_missing_node_as_separate_community(detected_communities_hh1.communities)
            nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['home_team']), graph=home_1hg)
            nmi_instance = normalized_mutual_information(nc, detected_communities_hh1).score
            hteam_h1_nmi.append(nmi_instance)


            detected_communities_ah1 = alg(away_1hg, weights=away_1hweights)
            detected_communities_ah1.communities == add_missing_node_as_separate_community(detected_communities_ah1.communities)
            nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['away_team']),
                                graph=away_1hg)
            nmi_instance = normalized_mutual_information(nc, detected_communities_ah1).score
            ateam_h1_nmi.append(nmi_instance)


            detected_communities_hh2 = alg(home_2hg, weights=home_2hweights)
            print('******', game_id,'\n', detected_communities_hh2.communities)
            print('Half 2 - home')
            detected_communities_hh2.communities == add_missing_node_as_separate_community(detected_communities_hh2.communities)
            nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['home_team']),
                                graph=home_2hg)
            # print(f'******{game_id}************', list(nc))
            nmi_instance = normalized_mutual_information(nc, detected_communities_hh2).score

            # print(transform_communities_to_str(communities_by_position[i]['home_team']))
            # print(detected_communities_hh2.communities)

            hteam_h2_nmi.append(nmi_instance)

            detected_communities_ah2 = alg(away_2hg, weights=away_2hweights)
            detected_communities_ah2.communities == add_missing_node_as_separate_community(detected_communities_ah2.communities)
            nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['away_team']),
                                graph=away_2hg)
            nmi_instance = normalized_mutual_information(nc, detected_communities_ah2).score

            ateam_h2_nmi.append(nmi_instance)

        except:
            print('Error occured for game')
            print(game_id)
            print('----------------')

    nmi_halftime_split_coms = pd.DataFrame({'game': the_ids,
                                     'first_' + filenamestr +'_NMI_hteam': hteam_h1_nmi,
                                     'second_' + filenamestr +'_NMI_hteam': hteam_h2_nmi,
                                    'first_' + filenamestr +'_NMI_ateam': ateam_h1_nmi,
                                    'second_' + filenamestr +'_NMI_ateam': ateam_h2_nmi
                                    # 'hometeam_1h_vs_2h_nmi': hometeam_1h_vs_2h_nmis,
                                    # 'awayteam_1h_vs_2h_nmi': awayteam_1h_vs_2h_nmis
                                    })

    # nmi_halftime_split_coms = pd.DataFrame({'game': the_ids + the_ids,
    #                                         'first_' + filenamestr + '_NMI': hteam_nmis_first_half + ateam_nmis_first_half,
    #                                         'second_' + filenamestr + '_NMI': hteam_nmis_second_half + ateam_nmis_second_half,
    #                                         '1h_vs_2h_nmi': hometeam_1h_vs_2h_nmis + awayteam_1h_vs_2h_nmis,
    #                                         })
    return nmi_halftime_split_coms
    # nmi_halftime_split_coms.to_csv('nmi_' + filenamestr +'time_split_coms.csv', index=False)

def tree_pruning(gw):
    g = gw[0]
    weights = gw[1]
    edata = g.edges(data=True)
    # print('Edges length before prunning:' + str(len(edata)))

    weights = []
    for edge in edata:
        weights.append(edge[2]['weight'])

    median = statistics.median(weights)
    # print('Median is: ' + str(median))

    ws = []
    for i, edge in enumerate(copy.deepcopy(g.edges(data=True))):
        if edge[2]['weight'] <= median:
            g.remove_edge(edge[0], edge[1])
        else:
            ws.append(weights[i])

    # print('Edges length after prunning:' + str(len(ws)))
    return g, ws

def edge_clustering(type):
    game_id = game_ids[0]
    home_team, away_team = read_team_names(game_id)
    print('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_0.net')
    home_1hg, home_1hweights = tree_pruning(
        parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_0.net',
                             with_node_names=False))

    return hierarchical_link_community(home_1hg)

def visualize_nmis():
    alg_names = ['Aslpaw', 'Louvain', 'Leiden', 'Greedy modularity', 'CPM']
    algorithms = [aslpaw, louvain, leiden, greedy_modularity, cpm]

    half_nmi_leiden = see_community_changes('goal', leiden)
    half_nmi_cpm = see_community_changes('goal', cpm)

    gms = []
    for i in range(1, len(half_nmi_cpm['game']) + 1):
        gms.append('Game #' + str(i))

    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel('Games', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)

    ax.plot(range(1, len(half_nmi_cpm['goal']) + 1), half_nmi_leiden['first_goal_NMI_hteam'],
            label='Leiden before goal Home team')
    # ax.plot(range(1, len(half_nmi_cpm['game']) + 1), half_nmi_leiden['second_card_NMI_hteam'],
    #         label='Leiden 2nd card Home team')

    ax.plot(range(1, len(half_nmi_cpm['game']) + 1), half_nmi_leiden['first_goal_NMI_ateam'],
            label='Leiden before goal Away team')
    # ax.plot(range(1, len(half_nmi_cpm['game']) + 1), half_nmi_leiden['second_card_NMI_ateam'],
    #         label='Leiden 2nd card Away team')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.title('NMI for 8 selected games')
    plt.savefig('NMI_leiden_card.png')
    plt.show()

def compute_ratios_for_postions(original_coms, detected_coms):

    originales = transform_communities_to_str(original_coms)
    gk_ratios = []
    def_ratios = []
    mid_ratios = []
    att_ratios = []
    com_lens = []

    for comm in detected_coms:
        # print(original_coms[1])
        # print(comm)
        gk_ratios.append(len([i for i in comm if i in originales[0]]) / len(comm))
        def_ratios.append(len([i for i in comm if i in originales[1]]) / len(comm))
        mid_ratios.append(len([i for i in comm if i in originales[2]]) / len(comm))
        att_ratios.append(len([i for i in comm if i in originales[3]]) / len(comm))
        com_lens.append(len(comm))

    df = pd.DataFrame({'gk_ratios': gk_ratios, 'def_ratios': def_ratios, 'mid_ratios': mid_ratios, 'att_ratios': att_ratios, 'com_lens': com_lens})
    # print(df)
    # print('-0---------->')
    return df

def some_f():
    type = 'period'
    hteam_ratios_t1 = pd.DataFrame()
    hteam_ratios_t2 = pd.DataFrame()
    ateam_ratios_t1 = pd.DataFrame()
    ateam_ratios_t2 = pd.DataFrame()
    original_comms = communities_by_position + rc_communities_by_position

    for j, game_id in enumerate(game_ids + red_card_games):
        home_team, away_team = read_team_names(game_id)
        home_1hg, home_1hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_0.net',with_node_names=False))
        home_2hg, home_2hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_1.net',with_node_names=False))
        away_1hg, away_1hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_' + type + '_0.net',with_node_names=False))
        away_2hg, away_2hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_' + type + '_1.net',with_node_names=False))

        detected_communities_hh1 = leiden(home_1hg, weights=home_1hweights)
        detected_communities_hh2 = leiden(home_2hg, weights=home_2hweights)
        detected_communities_ah1 = leiden(away_1hg, weights=away_1hweights)
        detected_communities_ah2 = leiden(away_2hg, weights=away_2hweights)

        hteam_ratios_t1 = hteam_ratios_t1.append(compute_ratios_for_postions(original_comms[j]['home_team'], detected_communities_hh1.communities), ignore_index=True)
        hteam_ratios_t2 = hteam_ratios_t2.append(compute_ratios_for_postions(original_comms[j]['home_team'], detected_communities_hh2.communities), ignore_index=True)
        ateam_ratios_t1 = ateam_ratios_t1.append(compute_ratios_for_postions(original_comms[j]['away_team'], detected_communities_ah1.communities), ignore_index=True)
        ateam_ratios_t2 = ateam_ratios_t2.append(compute_ratios_for_postions(original_comms[j]['away_team'], detected_communities_ah2.communities), ignore_index=True)

    all_together = hteam_ratios_t2.append(ateam_ratios_t2, ignore_index=True)
    # .append(ateam_ratios_t1, ignore_index=True).append(ateam_ratios_t2, ignore_index=True)
    hteam_ratios_t1_avgs = all_together.groupby('com_lens').mean()
    hteam_ratios_t1_stds = all_together.groupby('com_lens').std().fillna(0)

    labels = hteam_ratios_t1_avgs.index
    gk_means = list(hteam_ratios_t1_avgs['gk_ratios'])
    def_means = list(hteam_ratios_t1_avgs['def_ratios'])
    mid_means = list(hteam_ratios_t1_avgs['mid_ratios'])
    att_means = list(hteam_ratios_t1_avgs['att_ratios'])

    gk_stds = list(hteam_ratios_t1_stds['gk_ratios'])
    def_stds = list(hteam_ratios_t1_stds['def_ratios'])
    mid_stds = list(hteam_ratios_t1_stds['mid_ratios'])
    att_stds = list(hteam_ratios_t1_stds['att_ratios'])

    width = 0.35  # the width of the bars: can also be len(x) sequence

    fig, ax = plt.subplots(figsize=(15,10))
    df_gk_means = [gk_means[i] + def_means[i] for i in range(len(def_means))]
    md_df_gk_means = [gk_means[i] + def_means[i] + + mid_means[i] for i in range(len(def_means))]

    ax.bar(labels, gk_means, label='Goalkeeper')
    ax.bar(labels, def_means, bottom=gk_means, label='Defense')
    ax.bar(labels, mid_means, bottom=df_gk_means, label='Middle')
    ax.bar(labels, att_means, bottom=md_df_gk_means, label='Attack')

    ax.set_ylabel('Ratios')
    ax.set_xlabel('Size of the community')
    ax.set_title('Ratios of positons by community size (2nd half) hometeam')
    ax.legend(loc='upper right', bbox_to_anchor=(1.13, 0.8), prop={'size': 12})
    plt.savefig('stack_bar_positions_second_half.png')
    plt.show()

    # hteam_ratios_t1_avgs = hteam_ratios_t2.groupby('com_lens').mean()
    # hteam_ratios_t1_stds = hteam_ratios_t2.groupby('com_lens').std().fillna(0)
    #
    # labels = hteam_ratios_t1_avgs.index
    # gk_means = list(hteam_ratios_t1_avgs['gk_ratios'])
    # def_means = list(hteam_ratios_t1_avgs['def_ratios'])
    # mid_means = list(hteam_ratios_t1_avgs['mid_ratios'])
    # att_means = list(hteam_ratios_t1_avgs['att_ratios'])
    #
    # gk_stds = list(hteam_ratios_t1_stds['gk_ratios'])
    # def_stds = list(hteam_ratios_t1_stds['def_ratios'])
    # mid_stds = list(hteam_ratios_t1_stds['mid_ratios'])
    # att_stds = list(hteam_ratios_t1_stds['att_ratios'])
    #
    # width = 0.35  # the width of the bars: can also be len(x) sequence
    #
    # fig, ax = plt.subplots()
    # df_gk_means = [gk_means[i] + def_means[i] for i in range(len(def_means))]
    # md_df_gk_means = [gk_means[i] + def_means[i] + + mid_means[i] for i in range(len(def_means))]
    #
    # ax.bar(labels, gk_means, label='Goalkeeper')
    # ax.bar(labels, def_means, bottom=gk_means, label='Defense')
    # ax.bar(labels, mid_means, bottom=df_gk_means, label='Middle')
    # ax.bar(labels, att_means, bottom=md_df_gk_means, label='Attack')
    #
    # ax.set_ylabel('Ratios')
    # ax.set_xlabel('Size of the community')
    # ax.set_title('Ratios of positons by community size (2nd half) hometeam')
    # ax.legend()
    # plt.savefig('stack_bar_positions_ht2.png')
    # plt.show()
    #
    # hteam_ratios_t1_avgs = ateam_ratios_t1.groupby('com_lens').mean()
    # hteam_ratios_t1_stds = ateam_ratios_t1.groupby('com_lens').std().fillna(0)
    #
    # labels = hteam_ratios_t1_avgs.index
    # gk_means = list(hteam_ratios_t1_avgs['gk_ratios'])
    # def_means = list(hteam_ratios_t1_avgs['def_ratios'])
    # mid_means = list(hteam_ratios_t1_avgs['mid_ratios'])
    # att_means = list(hteam_ratios_t1_avgs['att_ratios'])
    #
    # gk_stds = list(hteam_ratios_t1_stds['gk_ratios'])
    # def_stds = list(hteam_ratios_t1_stds['def_ratios'])
    # mid_stds = list(hteam_ratios_t1_stds['mid_ratios'])
    # att_stds = list(hteam_ratios_t1_stds['att_ratios'])
    #
    # width = 0.35  # the width of the bars: can also be len(x) sequence
    #
    # fig, ax = plt.subplots()
    # df_gk_means = [gk_means[i] + def_means[i] for i in range(len(def_means))]
    # md_df_gk_means = [gk_means[i] + def_means[i] + + mid_means[i] for i in range(len(def_means))]
    #
    # ax.bar(labels, gk_means, label='Goalkeeper')
    # ax.bar(labels, def_means, bottom=gk_means, label='Defense')
    # ax.bar(labels, mid_means, bottom=df_gk_means, label='Middle')
    # ax.bar(labels, att_means, bottom=md_df_gk_means, label='Attack')
    #
    # ax.set_ylabel('Ratios')
    # ax.set_xlabel('Size of the community')
    # ax.set_title('Ratios of positons by community size (1st half) awayteam')
    # ax.legend()
    # plt.savefig('stack_bar_positions_away_ht1.png')
    # plt.show()
    #
    # hteam_ratios_t1_avgs = ateam_ratios_t2.groupby('com_lens').mean()
    # hteam_ratios_t1_stds = ateam_ratios_t2.groupby('com_lens').std().fillna(0)
    #
    # labels = hteam_ratios_t1_avgs.index
    # gk_means = list(hteam_ratios_t1_avgs['gk_ratios'])
    # def_means = list(hteam_ratios_t1_avgs['def_ratios'])
    # mid_means = list(hteam_ratios_t1_avgs['mid_ratios'])
    # att_means = list(hteam_ratios_t1_avgs['att_ratios'])
    #
    # gk_stds = list(hteam_ratios_t1_stds['gk_ratios'])
    # def_stds = list(hteam_ratios_t1_stds['def_ratios'])
    # mid_stds = list(hteam_ratios_t1_stds['mid_ratios'])
    # att_stds = list(hteam_ratios_t1_stds['att_ratios'])
    #
    # width = 0.35  # the width of the bars: can also be len(x) sequence
    #
    # fig, ax = plt.subplots()
    # df_gk_means = [gk_means[i] + def_means[i] for i in range(len(def_means))]
    # md_df_gk_means = [gk_means[i] + def_means[i] + + mid_means[i] for i in range(len(def_means))]
    #
    # ax.bar(labels, gk_means, label='Goalkeeper')
    # ax.bar(labels, def_means, bottom=gk_means, label='Defense')
    # ax.bar(labels, mid_means, bottom=df_gk_means, label='Middle')
    # ax.bar(labels, att_means, bottom=md_df_gk_means, label='Attack')
    #
    # ax.set_ylabel('Ratios')
    # ax.set_xlabel('Size of the community')
    # ax.set_title('Ratios of positons by community size (2nd half) awayteam')
    # ax.legend()
    # plt.savefig('stack_bar_positions_away_ht2.png')
    # plt.show()





def read_goal_lcommunities():
    team_performing_t1_comm = []
    team_performing_t2_comm = []
    team_opp_t1_comm = []
    team_opp_t2_comm = []
    teams = []
    game_results = []
    ngames = []

    for i, game_id in enumerate(ggames):
        home_team, away_team = read_team_names(game_id)
        gs = read_game_result(game_id)
        game_results.append(gs)
        game_results.append(gs)

        try:
            home_1hg, home_1hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_goal_0.net', with_node_names=False))
            away_1hg, away_1hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_goal_0.net', with_node_names=False))
            home_2hg, home_2hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_goal_1.net', with_node_names=False))
            away_2hg, away_2hweights = tree_pruning(parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_goal_1.net', with_node_names=False))
        except:
            print(game_id)
        teams.append(home_team)
        teams.append(away_team)

        try:
            detected_communities = leiden(home_1hg, weights=home_1hweights)
            largest_com = len(detected_communities.communities)
            if gevents[i] == 'h':
                team_performing_t1_comm.append(largest_com)
            else:
                team_opp_t1_comm.append(largest_com)

            detected_communities = leiden(away_1hg, weights=away_1hweights)
            largest_com = len(detected_communities.communities)
            if gevents[i] == 'h':
                team_opp_t1_comm.append(largest_com)
            else:
                team_performing_t1_comm.append(largest_com)

            detected_communities = leiden(home_2hg, weights=home_2hweights)
            largest_com = len(detected_communities.communities)
            if gevents[i] == 'h':
                team_performing_t2_comm.append(largest_com)
            else:
                team_opp_t2_comm.append(largest_com)

            detected_communities = leiden(away_2hg, weights=away_2hweights)
            largest_com = len(detected_communities.communities)
            if gevents[i] == 'h':
                team_opp_t2_comm.append(largest_com)
            else:
                team_performing_t2_comm.append(largest_com)
        except:
            print(game_id)
            # rc_games.remove(game_id)

    print('----- Duzine --------------')
    print(len(team_performing_t1_comm))
    print(len(team_performing_t2_comm))
    print(len(team_opp_t1_comm))
    print(len(team_opp_t2_comm))
    print('------------')
    cms = pd.DataFrame({'team_scoring_goal_t1': team_performing_t1_comm,
        'team_scoring_goal_t2': team_performing_t2_comm,
        'team_opp_goal_t1': team_opp_t1_comm,
        'team_opp_goal_t2': team_opp_t2_comm})

    avgs = [cms['team_scoring_goal_t1'].mean(), cms['team_scoring_goal_t2'].mean(), cms['team_opp_goal_t1'].mean(),
            cms['team_opp_goal_t2'].mean()]
    stds = [cms['team_scoring_goal_t1'].std(), cms['team_scoring_goal_t2'].std(), cms['team_opp_goal_t1'].std(),
            cms['team_opp_goal_t2'].std()]
    x = ['Before scoring goal', 'After scoring goal', 'Before conceding goal', 'After conceding goal']

    diffs = pd.DataFrame({'x': x, 'avg': avgs})
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(x="x", y="avg", yerr=stds, data=diffs)
    plt.xticks(rotation=90);
    plt.savefig('ncom_goal.png')
    plt.show()

if __name__ == "__main__":
    # read_halftime_split_for_communities()
    # read_goal_split_for_communities()
    # read_redcard_split_for_communities()
    # print(len(rc_games))
    # extract_goal_games(90)
    # extract_redcard_games(90)

    # df = see_community_changes('goal')
    #
    # dteam1 = df['first_goal_NMI_did']
    #
    #
    #
    # sns.set_theme(style="whitegrid")
    # # tips = sns.load_dataset("tips")
    # ax = sns.barplot(x="day", y="total_bill", data=df)
    # who_did_what('goal')
    # gevents = who_did_what('goal')
    # read_goal_lcommunities()

    # some_f()
    # print(len(rc_games))
    # rc_events = who_did_what('card')
    # print(len(rc_events))
    # print(rc_events)

    some_f()










