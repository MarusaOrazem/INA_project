from utils import *
from classes import *
import networkx as nx
from pathlib import Path
import os
import json

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

def create_pajek(match, team, filename, dir, with_goals):
    '''
    Writes pajek format of graph, which nodes are players id, and weighted edges for passe between them. File is saves as match_id.net.
    :param match: match you want to visualise, type Match()
    :param team: which team you want to visualise events for, type Team()
    '''

    Path(f'{dir}/{match.id}').mkdir(parents=True, exist_ok=True)

    G = nx.DiGraph()

    events = match.events

    weighted_edges = {}
    substitutions_out = {}
    substitutions_in = {}
    for event in events:
        if event['team']['id'] != team.id:
            continue
        if event['type']['id'] == 30: #pass
            if 'outcome' in list(event['pass'].keys()):
                #if object outcome exists, pass was not successfull
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

        elif with_goals and event['type']['id'] == 16: #shot
            goal, player = get_shot_players(event)
            edge = (goal.name, player.name)
            if edge in list(weighted_edges.keys()):
                weighted_edges[edge] += 1
            else:
                weighted_edges[edge] = 1

    #relabel the nodes if the substitution happened
    weighted_edges = rename_substitution_nodes(weighted_edges, substitutions_out, substitutions_in)

    #reorganize structure of edges (node1,node2,weight)
    edges = []
    count = 0
    for edge in list(weighted_edges.keys()):
        edges.append((edge[0], edge[1], weighted_edges[edge]))
        count += weighted_edges[edge]
    #print(f'All weights {count} ')

    G.add_weighted_edges_from(edges)
    if with_goals:
        filename += '_with_goals'
    #nx.write_pajek(G, f'{dir}/{match.id}/{filename}.net')


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

def separate_events_by_periods(events, extensions = False):
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

    if extensions:
        return first, second, third, forth, fifth
    else:
        return first, second, 0, 0

def separate_events_by_first_goal(events):
    '''
    Separates events by first goal
    :param events: list of events
    :return: two lists of events, one before goal and second after goal
    '''
    before_goal = []
    after_goal = []
    minute = -1
    second = -1
    goal = False
    for event in events:
        if event['period'] == 3:
            print('Match has more than 2 periods!')
            return before_goal, after_goal, minute, second
        if event['type']['id'] == 16 and not goal:
            if event['shot']['outcome']['id'] == 97: #goal:
                goal = True
                minute = event['minute']
                second = event['second']
        if goal:
            after_goal.append(event)
        else:
            before_goal.append(event)

    if not goal:
        print('Match has no goals.')

    return before_goal, after_goal, minute, second

def separate_events_by_cards(events):
    '''
    Separates events by first card - second yellow or red
    :param events: list of events
    :return: two lists of events, one before goal and second after goal
    '''
    before_card = []
    after_card = []
    minute = -1
    second = -1
    card = False
    for event in events:
        if event['period'] == 3:
            print('Match has more than 2 periods!')
            return before_card, after_card, minute, second
        if event['type']['id'] == 22 and not card:
            if 'foul_committed' in event.keys():
                if 'card' in event['foul_committed']:
                    if event['foul_committed']['card']['id'] == 6: #second yellow
                        print('Second yellow card')
                        card = True
                        # print(event)
                        minute = event['minute']
                        second = event['second']
                    elif event['foul_committed']['card']['id'] == 5: #red card
                        # print(event)
                        print('Red card')
                        card = True
                        minute = event['minute']
                        second = event['second']
        if card:
            after_card.append(event)
        else:
            before_card.append(event)

    if not card:
        print('Match has no second yellow or red cards.')

    return before_card, after_card, minute, second


def separate_match(competition_id, season_id, match_id, dir, match_data, with_goals = False):

    data = get_match_data(competition_id, season_id, match_id)
    events = get_match_events(match_id)

    home_team = get_home_team(data)
    away_team = get_away_team(data)

    match = Match(match_id, home_team, data['home_score'], away_team, data['away_score'])
    print(match)


    separations = [separate_events_by_periods, separate_events_by_first_goal, separate_events_by_cards]
    names = ['period', 'goal', 'card']
    times = []
    possessions = []
    for i, sep in enumerate(separations):
        separation = sep(events)
        first, second, minute, seconds = separation[0], separation[1], separation[2], separation[3]
        times.append((minute, seconds))
        time_possessions = []
        for j, period in enumerate([first, second]):
            match.events = period
            if type(period) == int:
                a=3
            create_pajek(match, home_team, str(match_id) + f'_{home_team}' + f'_{names[i]}_{j}', dir, with_goals)
            create_pajek(match, away_team, str(match_id) + f'_{away_team}' + f'_{names[i]}_{j}', dir, with_goals)
            time_home = possession_counter(period, home_team)/60
            time_away = possession_counter(period, away_team)/60
            time_possessions.append((time_home, time_away))
        possessions.append(time_possessions)

    with open(f'{dir}/{match_id}/{match_id}.txt', 'w', encoding='utf-8') as f:
        f.write(f'ID: {match.id}   \n')
        f.write(f'Competition: {match_data["competition"]["competition_name"]}    \n')
        f.write(f'Country: {match_data["competition"]["country_name"]}    \n')
        f.write(f'Season: {match_data["season"]["season_name"]}    \n')
        f.write(f'Gender: {match_data["home_team"]["home_team_gender"]}     \n')
        f.write(f'Home team: {match.home_team}, goals: {match.home_score}   \n')
        f.write(f'Away team: {match.away_team}, goals: {match.away_score}    \n')
        f.write(f'Goal time: {times[1]}   \n')
        f.write(f'Card time: {times[2]}   \n')
        f.write(f'Before goal: Time in possession for home team {home_team.id}: {possessions[1][0][0]}    \n')
        f.write(f'Before goal: Time in possession for away team {away_team.id}: {possessions[1][0][1]}    \n')
        f.write(f'After goal: Time in possession for home team {home_team.id}: {possessions[1][1][0]}    \n')
        f.write(f'After goal: Time in possession for away team {away_team.id}: {possessions[1][1][1]}    \n')
        f.write(f'Before card: Time in possession for home team {home_team.id}: {possessions[2][0][0]}    \n')
        f.write(f'Before card: Time in possession for away team {away_team.id}: {possessions[2][0][1]}    \n')
        f.write(f'After card: Time in possession for home team {home_team.id}: {possessions[2][1][0]}    \n')
        f.write(f'After card: Time in possession for away team {away_team.id}: {possessions[2][1][1]}    \n')



if __name__ == "__main__":
    competition_id = 2
    season_id = 44
    match_id = 3749257

    '''
    data = get_match_data(competition_id, season_id, match_id)
    events = get_match_events(match_id)
    first, second = separate_events_by_periods(events)


    home_team = get_home_team(data)
    away_team = get_away_team(data)

    match = Match(match_id, home_team, data['home_score'], away_team, data['away_score'])
    print(match)

    for i, period in enumerate([first, second]):
        match.events = period
        create_pajek(match, home_team, str(match_id) + home_team + f'_{i}', 'C:/Users/Acer/Desktop/frizura', True)
        create_pajek(match, away_team, str(match_id) + f'_{i}', 'C:/Users/Acer/Desktop/frizura', True)
    
    match.events = events
    print_goals(match)'''
    #separate_match(competition_id, season_id, match_id, 'C:/Users/Acer/Desktop/frizura' )

    data_dir = 'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/src/open-data/data/matches'
    nets_dir = 'C:/Users/Acer/Desktop/Data science/1 letnik/Introduction to network analysis/INA_project/nets'
    #nets_dir =  'C:/Users/Acer/Desktop/frizura'
    all_matches = []

    for competition_id in os.listdir(data_dir):
        print(competition_id)
        for season_id in os.listdir(data_dir+'/'+competition_id):
            print(season_id)
            with open(f'open-data/data/matches/{competition_id}/{season_id}', encoding='utf-8') as f:
                matches = json.load(f)
                for m in matches:
                    match_id = m['match_id']
                    print(match_id)
                    separate_match(competition_id, season_id[:-5], match_id, nets_dir, m)
                    separate_match(competition_id, season_id[:-5], match_id, nets_dir, m, True)
                    all_matches.append(f'ID: {match_id}, Country: {m["competition"]["country_name"]}, Competition: {m["competition"]["competition_name"]}, Season: {m["season"]["season_name"]}, Gender: {m["home_team"]["home_team_gender"]}, Has dismissal: {has_dismissal(match_id)}, Home team: {m["home_team"]["home_team_name"]}, goals: {m["home_score"]}, Away team: {m["away_team"]["away_team_name"]}, goals: {m["away_score"]}   ')

    with open(f'{nets_dir[:-5]}/all_matches.txt', 'w', encoding='utf-8') as f:
        for match in all_matches:
            f.write(f'{match}  \n')

