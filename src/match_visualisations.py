from utils import *
from classes import *
import networkx as nx

def rename_substitution_nodes(weighted_edges, substitutions_out, substitutions_in):
    '''
    Renames nodes - if one player was substituted, it is represented with the same node but it is named as player1.name/player2.name
    :param weighted_edges: list of edges, [(name1,name2,weight)]
    :param substitutions_out: dictionary, {player_name_went_out : player_name_that_went_in}
    :param substitutions_in: dictionary, {player_name_that_went_in : player_name_went_out}
    :return: list of edges, that has renamed labels
    '''
    for sub in substitutions_out.keys():
        iter = weighted_edges.copy()
        for edge in iter:
            if edge[0] == sub:
                edge_new = (edge[0] + '/' + substitutions_out[sub], edge[1])
                weighted_edges[edge_new] = weighted_edges.pop(edge)
            if edge[1] == sub:
                edge_new = (edge[0], edge[1] + '/' + substitutions_out[sub] )
                weighted_edges[edge_new] = weighted_edges.pop(edge)

    for sub in substitutions_in.keys():
        iter = weighted_edges.copy()
        for edge in iter:
            if edge[0] == sub:
                edge_new = (substitutions_in[sub] + '/' + edge[0], edge[1])
                weighted_edges[edge_new] = weighted_edges.pop(edge)
            if edge[1] == sub:
                edge_new = (edge[0], substitutions_in[sub] + '/' + edge[1])
                weighted_edges[edge_new] = weighted_edges.pop(edge)
    return weighted_edges

def create_pajek(match, team, filename):
    '''
    Writes pajek format of graph, which nodes are players id, and weighted edges for passe between them. File is saves as match_id.net.
    :param match: match you want to visualise, type Match()
    :param team: which team you want to visualise events for, type Team()
    '''
    G = nx.DiGraph()

    events = match.events
    weighted_edges = {}
    substitutions_out = {}
    substitutions_in = {}
    for event in events:
        if event['type']['id'] == 30: #pass
            if 'outcome' in list(event['pass'].keys()):
                #if object outcome exists, pass was not successfull
                continue
            elif event['team']['id'] != team.id:
                continue
            else:
                player1, player2 = get_pass_players(event)
                #check if the player was substituted, so we join labels
                edge = (player1.name, player2.name)
                if edge in list(weighted_edges.keys()):
                    weighted_edges[edge] += 1
                else:
                    weighted_edges[edge] = 1
        elif event['type']['id'] == 19:  # substitution
            out_player = event['player']['name']
            in_player = event['substitution']['replacement']['name']
            substitutions_out[out_player] = in_player
            substitutions_in[in_player] = out_player

    #relabel the nodes if the substitution happened
    weighted_edges = rename_substitution_nodes(weighted_edges, substitutions_out, substitutions_in)

    #reorganize structure of edges (node1,node2,weight)
    edges = []
    for edge in list(weighted_edges.keys()):
        edges.append((edge[0], edge[1], weighted_edges[edge]))

    G.add_weighted_edges_from(edges)
    nx.write_pajek(G, f'{filename}.net')


def print_goals(match):
    '''
    Prints all goals of the match
    '''
    events = match.events
    for event in events:
        if event['type']['id'] == 16:
            if event['shot']['outcome']['id'] == 97: #goal:
                team_scored = event['team']['name']
                time = event['timestamp']
                print(f'GOAL for: {team_scored} at {time}')

def separate_events_by_periods(events):
    '''
    Separates events by period: first half, second half, first extensions, second extensions, penalties
    :param events: list of events
    :return: five lists of events, each for each period
    '''
    first, second, third, forth, fifth = [], [], [], [], []
    for event in events:
        if event['period'] == 1:
            first.append(event)
        elif event['period'] == 2:
            second.append(event)
        elif event['period'] == 3:
            third.append(event)
        elif event['period'] == 4:
            forth.append(event)
        elif event['period'] == 5:
            fifth.append(event)

    return first, second, third, forth, fifth


if __name__ == "__main__":
    competition_id = 2
    season_id = 44
    match_id = 3749257

    data = get_match_data(competition_id, season_id, match_id)
    events = get_match_events(match_id)
    first, second, _, _, _ = separate_events_by_periods(events)

    home_team = get_home_team(data)
    away_team = get_away_team(data)

    match = Match(match_id, home_team, data['home_score'], away_team, data['away_score'])
    print(match)

    for i, period in enumerate([first, second]):
        match.events = period
        create_pajek(match, home_team, str(match_id) + f'_{i}')

    match.events = events
    print_goals(match)
