import networkx as nx
from cdlib.algorithms import louvain, leiden, significance_communities, label_propagation
from cdlib.evaluation import normalized_mutual_information, variation_of_information
from cdlib import NodeClustering
import matplotlib.pyplot as plt
import pandas as pd

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

def parse_weighted_graph(file):
    nodes, node_names, edges, weights = read_weighted_edge_list(file)
    G = nx.DiGraph()
    for i, edge in enumerate(edges):

        G.add_edge(node_names[int(edge[0]) - 1], node_names[int(edge[1]) - 1], weight = weights[i])

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

        detected_communities = leiden(home_1hg, weights=home_1hweights)
        comm_1h_number.append(len(detected_communities.communities))

        detected_communities = leiden(away_1hg, weights=away_1hweights)
        comm_1h_number.append(len(detected_communities.communities))

        detected_communities = leiden(home_2hg, weights=home_2hweights)
        comm_2h_number.append(len(detected_communities.communities))

        detected_communities = leiden(away_2hg, weights=away_2hweights)
        comm_2h_number.append(len(detected_communities.communities))



    first_half_comms = pd.DataFrame({'team': teams,
                                     'first_half_comms': comm_1h_number,
                                     'second_half_team': comm_2h_number,
                                     'game_results': game_results})
    first_half_comms.to_csv('half_split_communities.csv', index=False)


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



    goal_split_cooms = pd.DataFrame({'team': teams,
                                     'before_first_goal_comms': comm_1h_number,
                                     'after_first_goal_comms': comm_2h_number,
                                     'game_results': game_results})
    goal_split_cooms.to_csv('goal_split_communities.csv', index=False)

if __name__ == "__main__":
    # print(read_team_names(game_ids[0]))

    # read_halftime_split_for_communities()
    read_goal_split_for_communities()

    # g, weights = parse_weighted_graph('../nets/303524/303524_Barcelona (217)_period_0.net')
    #
    # algorithms = [leiden]
    #
    # for i, alg in enumerate(algorithms):
    #     detected_communities = alg(g)
    #     # print(detected_communities.communities)
    #
    # # print(weights)
    #
    # pos = nx.spring_layout(g)
    # colors = ['red', 'blue', 'green']
    # color_map = []
    #
    # normalized = normalize_array(weights)
    # nodes = g.nodes()
    # for node in nodes:
    #     for i, comm in enumerate(detected_communities.communities):
    #         if node in comm:
    #             color_map.append(colors[i])
    #
    # print(color_map)
    # nx.set_node_attributes(g, color_map, 'color')
    # nx.draw(G=g, pos=pos, node_color= color_map,with_labels=True)
    # nx.draw_networkx_edges(G=g, pos=pos, width=normalized)
    # plt.savefig('before_detection.png')
    #
    # plt.show()


    # gc = recreate_graph_from_communities(g, detected_communities.communities)
    #
    #
    #
    # nx.draw(G=gc, pos=pos, with_labels=True)
    # nx.draw_networkx_edges(G=gc, pos=pos, width=normalized)
    # plt.savefig('after_detection.png')
    # plt.show()




