from classes import *
import json
from match_visualisations import *
from datetime import datetime

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

def get_shot_players(event):
    shot = event['shot']
    if shot['outcome']['id'] in [97, 100, 116]: #on goal
        player2_id = None
        player2_name = f'ON_goal'
    elif shot['outcome']['id'] in [96, 98, 99, 101, 115]: #off goal
        player2_id = None
        player2_name = f'OFF_goal'

    player1_id = event['player']['id']
    player1_name = event['player']['name']

    return Player(player1_id, player1_name), Player(player2_id, player2_name)

def has_dismissal(match_id):
    events = get_match_events(match_id)
    for event in events:
        if event['period'] == 3:
            print('Match has more than 2 periods!')
            return False
        if event['type']['id'] == 22:
            if 'foul_committed' in event.keys():
                if 'card' in event['foul_committed']:
                    if event['foul_committed']['card']['id'] == 6:  # second yellow
                        return True
                    elif event['foul_committed']['card']['id'] == 5:  # red card
                        return True
    return False

def count_passes(events, team):
    count = 0
    for event in events:
        if event['team']['id'] == team and event['type']['id'] == 30:
            count +=1
    print(count)

def possession_counter(events, team):
    count = 0
    if len(events) == 0:
        return -1
    in_possession = events[0]['possession_team']['id'] == team.id
    if in_possession:
        timestamp = events[0]['timestamp'].split('.')[0]
        start_time = datetime.strptime(timestamp,"%H:%M:%S")
    first_half = True

    for i in range(len(events)-1):
        #print(event['possession'], event['possession_team'])
        if first_half and events[i]['period'] == 2:
            first_half = False
            start_time = datetime.strptime('0:0:0', "%H:%M:%S")
        if  events[i+1]['possession_team']['id'] != team.id and events[i]['possession_team']['id'] == team.id:
            timestamp = events[i+1]['timestamp'].split('.')[0]
            stop_time = datetime.strptime(timestamp,"%H:%M:%S")
            count_add = (stop_time-start_time).seconds
            #print(count_add)
            count+=count_add
        elif  events[i+1]['possession_team']['id'] == team.id and events[i]['possession_team']['id'] != team.id:
            timestamp = events[i + 1]['timestamp'].split('.')[0]
            start_time = datetime.strptime(timestamp, "%H:%M:%S")
    return count

    




if __name__ == "__main__":
    match_id = 18242
    events = get_match_events(match_id)
    home_team = 217 #barcelona
    away_team = 224 #juventus
    count_passes(events, home_team)
    count_passes(events, away_team)


    events = get_match_events(match_id)

    home_team = Team(217, "Barcelona", None, None, None)
    away_team = Team(224, "Juventus", None,None,None)

    match = Match(match_id, home_team, 3, away_team, 1)
    match.events = events

    #create_pajek(match, away_team, "test", "test", False)
    first = possession_counter(events, home_team)
    second = possession_counter(events, away_team)
    print((first+second)/60)