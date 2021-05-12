from classes import *
import json

def get_match_data(competition_id, season_id, match_id):
    '''
    Gets match data for a specific match
    :param competition_id: int
    :param season_id: int
    :param match_id: int
    :return: list of json data
    '''
    with open(f'open-data/data/matches/{competition_id}/{season_id}.json', encoding='utf-8') as f:
        data = json.load(f)
        for m in data:
            if m['match_id'] == match_id:
                return m
    print('Match not found.')
    return False

def get_match_events(match_id):
    '''
    Gets events for the match
    :param match_id: int, match id
    :return: list of json events
    '''
    with open(f'open-data/data/events/{match_id}.json', encoding='utf-8') as f:
        data = json.load(f)
        return data

def get_away_team(data):
    '''
    From given data extracts the away team of the match.
    :param data: match data
    :return: Team()
    '''
    away_data = data['away_team']
    away_team = Team(away_data['away_team_id'], away_data['away_team_name'], away_data['away_team_gender'],
                     away_data['country']['id'], away_data['country']['name'])
    return away_team

def get_home_team(data):
    '''
    From given data extracts the home team of the match.
    :param data: match data
    :return: Team()
    '''
    home_data = data['home_team']
    home_team = Team(home_data['home_team_id'], home_data['home_team_name'], home_data['home_team_gender'],
                     home_data['country']['id'], home_data['country']['name'])
    return home_team

def get_pass_players(event):
    '''
    For a given event, extracts players which the pass was made between
    :param event: pass event
    :return: Player(), Player()
    '''
    player1_id = event['player']['id']
    player1_name = event['player']['name']
    player2_id = event['pass']['recipient']['id']
    player2_name = event['pass']['recipient']['name']

    return Player(player1_id, player1_name), Player(player2_id, player2_name)