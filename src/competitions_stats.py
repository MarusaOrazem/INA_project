import json
import os

class Competition():
    '''Competition class represent each competition for several seasons, it has unique id'''
    def __init__(self, id, name, country, gender):
        '''
        :param id: int, unique id
        :param name: string, name of a competition e.g. Champions League
        :param country: string, country of where competition was held
        :param gender: string, gender of the players, female or male
        '''
        self.id = id
        self.country = country
        self.name = name
        self.gender = gender
        self.seasons = []


    def __str__(self):
        return f'Competition id: {self.id}, name: {self.name}, \n    country: {self.country}, gender: {self.gender}, \n    seasons: {len(self.seasons)}'

    def print_num_matches(self):
        print('     Seasons: ')
        for season in self.seasons:
            print(f'     {season}')

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False

    def add_season(self, season):
        '''
        Adds seasons to seasons list.
        :param season: type Season()
        '''
        self.seasons.append(season)

class Season():
    '''
    Season class represents specific season in some competition
    '''
    def __init__(self, id, year, competition):
        '''
        :param id: int, unique id
        :param year: int, year of the season
        :param competition: which competition this was, type Competition()
        '''
        self.id = id
        self.year = year
        self.competition = competition
        self.matches = []

    def add_match(self, match):
        '''
        Adds match to matches list
        :param match: type of Match()
        '''
        self.matches.append(match)

    def __str__(self):
        return f'id: {self.id}, year: {self.year}, matches: {len(self.matches)}'

    def __eq__(self, other):
        if self.id == other.id:
            return True
        return False


class Match():
    '''
    Match class represents a single match between two Teams
    '''
    def __init__(self, id, home_team, home_score, away_team, away_score):
        '''
        :param id: int, unique id
        :param home_team: team that played at home, type of Team()
        :param home_score: int, number of goals home_team scored
        :param away_team: team that did not play at home, type of Team()
        :param away_score: int, number of goals away_team scored
        '''
        self.id = id
        self.home_team = home_team
        self.away_team = away_team
        self.home_score = home_score
        self.away_score = away_score

    def __str__(self):
        return f'id: {self.id} -- {self.home_team} {self.home_score} : {self.away_score} {self.away_team}'

class Team():
    '''
    Team class represents a single team
    '''
    def __init__(self, id, name, gender, country_id, country_name):
        '''
        :param id: int, unique id
        :param name: string, name of the team e.g. Real Madrid
        :param gender: string, gender of the players, female or male
        :param country_id: int, unique id
        :param country_name: string, country of where the team comes from
        '''
        self.id = id
        self.name = name
        self.gender = gender
        self.country_id = country_id
        self.country_name = country_name

    def __str__(self):
        return f'{self.name} ({self.id})'

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
