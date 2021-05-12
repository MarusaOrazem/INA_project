import os
import json
from classes import *
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import write_dot
import pydot


def get_match_data(competition_id, season_id, match_id):
    with open(f'open-data/data/matches/{competition_id}/{season_id}.json', encoding='utf-8') as f:
        data = json.load(f)
        for m in data:
            if m['match_id'] == match_id:
                return m
    print('Match not found.')
    return False

def get_match_events(match_id):
    with open(f'open-data/data/events/{match_id}.json', encoding='utf-8') as f:
        data = json.load(f)
        return data

def get_away_team(data):
    away_data = data['away_team']
    away_team = Team(away_data['away_team_id'], away_data['away_team_name'], away_data['away_team_gender'],
                     away_data['country']['id'], away_data['country']['name'])
    return away_team

def get_home_team(data):
    home_data = data['home_team']
    home_team = Team(home_data['home_team_id'], home_data['home_team_name'], home_data['home_team_gender'],
                     home_data['country']['id'], home_data['country']['name'])
    return home_team

def get_pass_players(event):
    player1_id = event['player']['id']
    player1_name = event['player']['name']
    player2_id = event['pass']['recipient']['id']
    player2_name = event['pass']['recipient']['name']

    return Player(player1_id, player1_name), Player(player2_id, player2_name)

def visualise(match, team):
    G = nx.MultiDiGraph()

    events = match.events
    print(len(events))
    for event in events:
        if event['type']['id'] == 30: #pass
            if 'outcome' in list(event['pass'].keys()):
                #if object outcome exists, pass was not successfull
                continue
            elif event['team']['id'] != team.id:
                a = 3
                continue
            else:
                player1, player2 = get_pass_players(event)
                G.add_edge(player1.id, player2.id)

    print(G.number_of_nodes())
    print(G.number_of_edges())
    write_dot(G, f'{match_id}.dot')

    (graph,) = pydot.graph_from_dot_file(f'{match_id}.dot')
    graph.write_png(f'{match_id}.png')




if __name__ == "__main__":
    competition_id = 2
    season_id = 44
    match_id = 3749257

    data = get_match_data(competition_id, season_id, match_id)
    events = get_match_events(match_id)

    home_team = get_home_team(data)
    away_team = get_away_team(data)

    match = Match(match_id, home_team, data['home_score'], away_team, data['away_score'])
    match.events = events

    visualise(match, home_team)