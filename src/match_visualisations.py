from utils import *
from classes import *
import networkx as nx



def create_pajek(match, team):
    '''
    Writes pajek format of graph, which nodes are players id, and weighted edges for passe between them. File is saves as match_id.net.
    :param match: match you want to visualise, type Match()
    :param team: which team you want to visualise events for, type Team()
    '''
    G = nx.DiGraph()

    events = match.events
    weighted_edges = {}
    for event in events:
        if event['type']['id'] == 30: #pass
            if 'outcome' in list(event['pass'].keys()):
                #if object outcome exists, pass was not successfull
                continue
            elif event['team']['id'] != team.id:
                continue
            else:
                player1, player2 = get_pass_players(event)
                edge = (player1.name, player2.name)
                if edge in list(weighted_edges.keys()):
                    weighted_edges[edge] += 1
                else:
                    weighted_edges[edge] = 1

    #reorganize structure of edges (node1,node2,weight)
    edges = []
    for edge in list(weighted_edges.keys()):
        edges.append((edge[0], edge[1], weighted_edges[edge]))

    G.add_weighted_edges_from(edges)
    nx.write_pajek(G, f'{match_id}.net')

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
    first, _, _, _, _ = separate_events_by_periods(events)

    home_team = get_home_team(data)
    away_team = get_away_team(data)

    match = Match(match_id, home_team, data['home_score'], away_team, data['away_score'])
    match.events = first

    create_pajek(match, home_team)
