from classes import *
import os
import json


def get_num_matches_with_events(n):
    '''
    Gets all matches with event n
    :param n: int, event type
    :return: list of all matches (type of Match())
    '''
    yellow_cards = 0
    second_yellow_cards = 0
    red_cards = 0
    autogoal = 0
    lost_after_first_scored = 0
    num_matches = 0
    competitions_ids = [f for f in os.listdir(f'open-data/data/matches')]
    for competition_id in competitions_ids:
        seasons_files = [f for f in os.listdir(f'open-data/data/matches/{competition_id}')]
        for season in seasons_files:
            with open(f'open-data/data/matches/{competition_id}/{season}', encoding='utf-8') as f:
                data = json.load(f)
                for m in data:
                    num_matches += 1
                    match_id = m['match_id']
                    home_data = m['home_team']
                    home_team = Team(home_data['home_team_id'], home_data['home_team_name'], home_data['home_team_gender'], home_data['country']['id'], home_data['country']['name'])
                    away_data = m['away_team']
                    away_team = Team(away_data['away_team_id'], away_data['away_team_name'], away_data['away_team_gender'], away_data['country']['id'], away_data['country']['name'])
                    match = Match(id, home_team, m['home_score'], away_team, m['away_score'])
                    with open(f'open-data/data/events/{match_id}.json', encoding='utf-8') as events_data:
                        events = json.load(events_data)
                        with open(f'open-data/data/matches/{competition_id}/{season}', encoding='utf-8') as ff:
                            #get winner id of this match
                            match = json.load(ff)
                            for m in match:
                                if m['match_id'] == match_id:
                                    home_score = m['home_score']
                                    away_score = m['away_score']
                                    if home_score > away_score:
                                        winner_id = m['home_team']['home_team_id']
                                    elif away_score > home_score:
                                        winner_id = m['away_team']['away_team_id']
                                    else:
                                        winner_id = -1 #tied score
                                    first_goal = True

                        for event in events:
                            if event['type']['id'] == 22:
                                if 'foul_committed' in event.keys():
                                    if 'card' in event['foul_committed']:
                                        if event['foul_committed']['card']['id'] == 7:
                                            yellow_cards +=1
                                            #print(event)
                                        elif event['foul_committed']['card']['id'] == 6:
                                            second_yellow_cards += 1
                                            #print(event)
                                        elif event['foul_committed']['card']['id'] == 5:
                                            #print(event)
                                            red_cards += 1

                            if event['type']['id'] == 25:
                                autogoal+=1

                            if event['type']['id'] == 16:
                                #shot
                                if event['shot']['outcome']['id'] == 97:
                                    #goal
                                    if event['possession_team']['id'] != winner_id and first_goal:
                                        lost_after_first_scored += 1
                                    else:
                                        first_goal = False


    print(f'Number of yellow cards: {yellow_cards}')
    print(f'Number of second yellow cards: {second_yellow_cards}')
    print(f'Number of red cards: {red_cards}')
    print(f'Number of auto goals: {autogoal}')
    print(f'Number of matches where team lost but scored first: {lost_after_first_scored}')
    print(f'Numbe rof all matches: {num_matches}')


if __name__ == "__main__":
    '''
    A list of event types
    42 - Ball Receipt
    2 - Ball Recovery
    3 - Dispossessed
    4 - Duel
    5 - Camera On
    6 - Block
    8 - Offside
    9 - Clearance
    10 - Interception
    14 - Dribble
    16 - Shot
    17 - Pressure
    18 - Half start
    19 - Substitution
    20 - Own Goal Against
    21 - Foul Won
    22 - Foul Committed
         5 - Red card
         6 - Second Yellow
         7 - Yellow Card
    23 - Goal Keeper
    24 - Bad Behaviour
            Receives a card because of a bad behaviour
    25 - Own Goal For
    26 - Player On
    27 - Player Off
    28 - Shield
    30 - Pass
    33 - 50/50
    34 - Half End
    35 - Starting XI
    36 - Tactical Shift
    37 - Error
    38 - Miscontrol
    39 - Dribbled Past
    40 - Injury Stoppage
    41 - Referee Ball-Drop
    43 - Carry
    '''
    get_num_matches_with_events()
