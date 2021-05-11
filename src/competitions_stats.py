import json
import os
from classes import *



def get_competitions():
    '''
    Reads all the competitions from competitions.json, creates Competitions() objects
    :return: list of Competitions() objects
    '''
    competitions = []
    with open('open-data/data/competitions.json') as f:
        data = json.load(f)
        for i in range(len(data)):
            com = data[i]
            t = Competition(com['competition_id'], com['competition_name'], com['country_name'], com['competition_gender'])
            season = Season(com['season_id'], com['season_name'], com['competition_id'])
            if t not in competitions:
                t.add_season(season)
                competitions.append(t)
            else:
                #if competition already added, found it and just add season
                competition = competitions[competitions.index(t)]
                competition.add_season(season)
    return competitions

def get_matches_for_competitions(competitions):
    '''
    For each competitions in competitions list, it adds matches for every season
    :param competitions: list of Competitions
    :return: list of Competitions
    '''
    for competition in competitions:
        competition_id = competition.id
        seasons_ids = [f for f in os.listdir(f'open-data/data/matches/{competition_id}') ]

        for season_id in seasons_ids:
            with open(f'open-data/data/matches/{competition_id}/{season_id}', encoding='utf-8') as f:
                matches = json.load(f)
                for m in matches:
                    season = Season(m['season']['season_id'], m['season']['season_name'], competition)
                    #if season not in competition add it
                    if season not in competition.seasons:
                        competition.add_season(season)
                    else:
                        #get already exsisted season from seasons list
                        season = competition.seasons[competition.seasons.index(season)]

                    home_data = m['home_team']
                    home_team = Team(home_data['home_team_id'], home_data['home_team_name'], home_data['home_team_gender'], home_data['country']['id'], home_data['country']['name'])
                    away_data = m['away_team']
                    away_team = Team(away_data['away_team_id'], away_data['away_team_name'], away_data['away_team_gender'], away_data['country']['id'], away_data['country']['name'])
                    match = Match(m['match_id'], home_team, m['home_score'], away_team, m['away_score'])
                    season.add_match(match)

    return competitions




if __name__ == "__main__":
    competitions = get_competitions()
    com = get_matches_for_competitions(competitions)
    for c in com:
        print(c)
        c.print_num_matches()
        print()
