import networkx as nx
from cdlib.algorithms import louvain, leiden, greedy_modularity, aslpaw, cpm, hierarchical_link_community
from cdlib.evaluation import normalized_mutual_information, variation_of_information
from cdlib import NodeClustering
import matplotlib.pyplot as plt
import pandas as pd
import statistics
import copy


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


def read_halftime_split_for_communities():
    comm_1h_number = []
    comm_2h_number = []
    teams = []
    game_results = []
    # algorithms = [leiden]

    for game_id in game_ids:
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

        try:
            detected_communities = leiden(home_1hg, weights=home_1hweights)
            comm_1h_number.append(len(detected_communities.communities))
        except:
            print('Game: ' + game_id)
            print('G edges: ' + str(len(home_1hg.edges)))
            print('G weights: ' + str(len(home_1hweights)))


        detected_communities = leiden(away_1hg, weights=away_1hweights)
        comm_1h_number.append(len(detected_communities.communities))

        detected_communities = leiden(home_2hg, weights=home_2hweights)
        comm_2h_number.append(len(detected_communities.communities))

        detected_communities = leiden(away_2hg, weights=away_2hweights)
        comm_2h_number.append(len(detected_communities.communities))

    first_half_comms = pd.DataFrame({'first_half_comms': comm_1h_number,
                                     'second_half_team': comm_2h_number})

    fig, ax = plt.subplots()
    ax.set_title('Boxplot for halftime split')
    ax.boxplot(first_half_comms)
    plt.xticks([1, 2], ['1st halftime', '2nd halftime'])
    plt.savefig('halftime_split_boxplot.png')
    plt.show()
    # first_half_comms = pd.DataFrame({'team': teams,
    #                                  'first_half_comms': comm_1h_number,
    #                                  'second_half_team': comm_2h_number,
    #                                  'game_results': game_results})
    # first_half_comms.to_csv('half_split_communities.csv', index=False)

def read_goal_split_for_communities():
    comm_1h_number = []
    comm_2h_number = []
    teams = []
    game_results = []
    # algorithms = [leiden]

    for game_id in game_ids:
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

        try:
            detected_communities = leiden(home_1hg, weights=home_1hweights)
            comm_1h_number.append(len(detected_communities.communities))
        except:
            print('Graph is disconnected')
            print(game_id)
            print(home_team)
            print(away_team)
            print(read_game_result(game_id))

        detected_communities = leiden(away_1hg, weights=away_1hweights)
        comm_1h_number.append(len(detected_communities.communities))

        try:
            detected_communities = leiden(home_2hg, weights=home_2hweights)
            comm_2h_number.append(len(detected_communities.communities))
        except:
            print('Graph is disconnected')
            print(game_id)
            print(home_team)
            print(away_team)
            print(read_game_result(game_id))
            # print(list(nx.connected_components(home_2hg)))
            # detected_communities = leiden(nx.connected_components(home_2hg), weights=home_2hweights)
            # comm_2h_number.append(len(detected_communities.communities))

        detected_communities = leiden(away_2hg, weights=away_2hweights)
        comm_2h_number.append(len(detected_communities.communities))

    first_half_comms = pd.DataFrame({'before_first_goal_comms': comm_1h_number,
                                     'after_first_goal_comms': comm_2h_number})
    fig, ax = plt.subplots()
    ax.set_title('Boxplot for goal split')
    ax.boxplot(first_half_comms)
    plt.xticks([1, 2], ['Before 1st goal', 'After 1st goal'])
    plt.savefig('goal_split_boxplot.png')
    plt.show()
    # goal_split_cooms = pd.DataFrame({'team': teams,
    #                                  'before_first_goal_comms': comm_1h_number,
    #                                  'after_first_goal_comms': comm_2h_number,
    #                                  'game_results': game_results})
    # goal_split_cooms.to_csv('goal_split_communities.csv', index=False)

def read_redcard_split_for_communities():
    comm_1h_number = []
    comm_2h_number = []
    teams = []
    game_results = []
    # algorithms = [leiden]

    for game_id in red_card_games:
        home_team, away_team = read_team_names(game_id)
        gs = read_game_result(game_id)
        game_results.append(gs)
        game_results.append(gs)
        home_1hg, home_1hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_card_0.net')
        away_1hg, away_1hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_card_0.net')
        home_2hg, home_2hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_card_1.net')
        away_2hg, away_2hweights = parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + away_team + '_card_1.net')

        teams.append(home_team)
        teams.append(away_team)

        try:
            detected_communities = leiden(home_1hg, weights=home_1hweights)
            comm_1h_number.append(len(detected_communities.communities))
        except:
            print('Graph is disconnected')
            print(game_id)
            print(home_team)
            print(away_team)
            print(read_game_result(game_id))

        detected_communities = leiden(away_1hg, weights=away_1hweights)
        comm_1h_number.append(len(detected_communities.communities))

        try:
            detected_communities = leiden(home_2hg, weights=home_2hweights)
            comm_2h_number.append(len(detected_communities.communities))
        except:
            print('Graph is disconnected')
            print(game_id)
            print(home_team)
            print(away_team)
            print(read_game_result(game_id))
            # print(list(nx.connected_components(home_2hg)))
            # detected_communities = leiden(nx.connected_components(home_2hg), weights=home_2hweights)
            # comm_2h_number.append(len(detected_communities.communities))

        detected_communities = leiden(away_2hg, weights=away_2hweights)
        comm_2h_number.append(len(detected_communities.communities))

    first_half_comms = pd.DataFrame({'before_first_card_comms': comm_1h_number,
                                     'after_first_card_comms': comm_2h_number})
    fig, ax = plt.subplots()
    ax.set_title('Boxplot for red card split')
    ax.boxplot(first_half_comms)
    plt.xticks([1, 2], ['Before red card', 'After red card'])
    plt.savefig('rcard_split_boxplot.png')
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
    hteam_nmis_first_half = []
    ateam_nmis_first_half = []

    hteam_nmis_second_half = []
    ateam_nmis_second_half = []

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

        detected_communities_hh1 = alg(home_1hg, weights=home_1hweights)
        detected_communities_hh1.communities == add_missing_node_as_separate_community(detected_communities_hh1.communities)
        nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['home_team']), graph=home_1hg)
        nmi_instance = normalized_mutual_information(nc, detected_communities_hh1).score
        hteam_nmis_first_half.append(nmi_instance)



        detected_communities_ah1 = alg(away_1hg, weights=away_1hweights)
        detected_communities_ah1.communities == add_missing_node_as_separate_community(detected_communities_ah1.communities)
        nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['away_team']),
                            graph=away_1hg)
        nmi_instance = normalized_mutual_information(nc, detected_communities_ah1).score
        ateam_nmis_first_half.append(nmi_instance)

        try:
            detected_communities_hh2 = alg(home_2hg, weights=home_2hweights)
            detected_communities_hh2.communities == add_missing_node_as_separate_community(detected_communities_hh2.communities)
            nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['home_team']),
                                graph=home_2hg)
            nmi_instance = normalized_mutual_information(nc, detected_communities_hh2).score
        except:
            print(game_id)



        # print(transform_communities_to_str(communities_by_position[i]['home_team']))
        # print(detected_communities_hh2.communities)

        hteam_nmis_second_half.append(nmi_instance)

        detected_communities_ah2 = alg(away_2hg, weights=away_2hweights)
        detected_communities_ah2.communities == add_missing_node_as_separate_community(detected_communities_ah2.communities)
        nc = NodeClustering(communities=transform_communities_to_str(communities_by_position[i]['away_team']),
                            graph=away_2hg)
        nmi_instance = normalized_mutual_information(nc, detected_communities_ah2).score
        ateam_nmis_second_half.append(nmi_instance)

        # Comparing differences in communitites between first and second half
        nmi_instance = normalized_mutual_information(detected_communities_hh1, detected_communities_hh2).score
        hometeam_1h_vs_2h_nmis.append(nmi_instance)

        nmi_instance = normalized_mutual_information(detected_communities_ah1, detected_communities_ah2).score
        awayteam_1h_vs_2h_nmis.append(nmi_instance)


        # detected_communities_hh2 = leiden(home_2hg, weights=home_2hweights)
        # detected_communities_ah2 = leiden(away_2hg, weights=away_2hweights)

    nmi_halftime_split_coms = pd.DataFrame({'game': the_ids,
                                     'first_' + filenamestr +'_NMI_hteam': hteam_nmis_first_half,
                                     'second_' + filenamestr +'_NMI_hteam': hteam_nmis_second_half,
                                    'first_' + filenamestr +'_NMI_ateam': ateam_nmis_first_half,
                                    'second_' + filenamestr +'_NMI_ateam': ateam_nmis_second_half,
                                    'hometeam_1h_vs_2h_nmi': hometeam_1h_vs_2h_nmis,
                                    'awayteam_1h_vs_2h_nmi': awayteam_1h_vs_2h_nmis
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
    print('Edges length before prunning:' + str(len(edata)))

    weights = []
    for edge in edata:
        weights.append(edge[2]['weight'])

    median = statistics.median(weights)
    print('Median is: ' + str(median))

    ws = []
    for i, edge in enumerate(copy.deepcopy(g.edges(data=True))):
        if edge[2]['weight'] < median:
            g.remove_edge(edge[0], edge[1])
        else:
            ws.append(weights[i])

    print('Edges length after prunning:' + str(len(ws)))
    return g, ws

def edge_clustering(type):
    game_id = game_ids[0]
    home_team, away_team = read_team_names(game_id)
    print('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_0.net')
    home_1hg, home_1hweights = tree_pruning(
        parse_weighted_graph('../nets/' + game_id + '/' + game_id + '_' + home_team + '_' + type + '_0.net',
                             with_node_names=False))

    return hierarchical_link_community(home_1hg)


if __name__ == "__main__":
    # print(read_team_names(game_ids[0]))

    # print(edge_clustering('period').communities)

    # read_halftime_split_for_communities()
    # read_goal_split_for_communities()
    # read_redcard_split_for_communities()

    alg_names = ['Aslpaw', 'Louvain', 'Leiden', 'Greedy modularity', 'CPM']
    algorithms = [aslpaw, louvain, leiden, greedy_modularity, cpm]

    # half_nmi_aslpaw = see_community_changes('period', aslpaw)
    # half_nmi_louvain = see_community_changes('period', louvain)
    half_nmi_leiden = see_community_changes('card', leiden)
    # half_nmi_greedy_modularity = see_community_changes('period', greedy_modularity)
    half_nmi_cpm = see_community_changes('card', cpm)

    gms = []
    for i in range(1, len(half_nmi_cpm['game']) + 1):
        gms.append('Game #' + str(i))


    fig = plt.figure()
    ax = plt.axes()
    plt.xlabel('Games', fontsize=14)
    plt.ylabel('NMI', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=14)


    # ax.plot(range(1, 25), half_nmi_leiden['first_half_NMI'], label='Leiden 1st half')
    # ax.plot(range(1, 25), half_nmi_leiden['second_half_NMI'], label='Leiden 2nd half')

    # ax.plot(range(1, 25), half_nmi_greedy_modularity['first_half_NMI'], label='Greedy modularity 1st half')
    # ax.plot(range(1, 25), half_nmi_greedy_modularity['second_half_NMI'], label='Greedy modularity  2nd half')

    ax.plot(range(1, len(half_nmi_cpm['game']) + 1), half_nmi_leiden['first_card_NMI_hteam'], label='Leiden 1st card Home team')
    ax.plot(range(1, len(half_nmi_cpm['game']) + 1), half_nmi_leiden['second_card_NMI_hteam'], label='Leiden 2nd card Home team')

    ax.plot(range(1, len(half_nmi_cpm['game']) + 1), half_nmi_leiden['first_card_NMI_ateam'], label='Leiden 1st card Away team')
    ax.plot(range(1, len(half_nmi_cpm['game']) + 1), half_nmi_leiden['second_card_NMI_ateam'], label='Leiden 2nd card Away team')

    # ax.plot(range(1, 25), half_nmi_walktrap['second_half_NMI'], label='Walktrap 2nd half')
    # ax.plot(mu_list, nmis[1], label='Walktrap')
    # ax.plot(mu_list, nmis[2], label='Label Propagation')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    plt.title('NMI for 8 selected games')
    plt.savefig('NMI_leiden_card.png')
    plt.show()





