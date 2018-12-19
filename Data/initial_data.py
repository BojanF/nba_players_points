import datetime
from Models.team import Team
from Models.player import Player
from Models.season import Season
from Models.player_season import Player_Season

# active franchises from 2015-16 onwards
teams = [
    Team('Toronto Raptors', 'TOR', 1),
    Team('Boston Celtics', 'BOS', 1),
    Team('Philadelphia 76ers', 'PHI', 1),
    Team('Cleveland Cavaliers', 'CLE', 1),
    Team('Indiana Pacers', 'IND', 1),
    Team('Miami Heat', 'MIA', 1),
    Team('Milwaukee Bucks', 'MIL', 1),
    Team('Washington Wizards', 'WAS', 1),
    Team('Detroit Pistons', 'DET', 1),
    Team('Charlotte Hornets', 'CHO', 1),
    Team('New York Knicks', 'NYK', 1),
    Team('Brooklyn Nets', 'BRK', 1),
    Team('Chicago Bulls', 'CHI', 1),
    Team('Orlando Magic', 'ORL', 1),
    Team('Atlanta Hawks', 'ATL', 1),
    Team('Houston Rockets', 'HOU', 1),
    Team('Golden State Warriors', 'GSW', 1),
    Team('Portland Trail Blazers', 'POR', 1),
    Team('Oklahoma City Thunder', 'OKC', 1),
    Team('Utah Jazz', 'UTA', 1),
    Team('New Orleans Pelicans', 'NOP', 1),
    Team('San Antonio Spurs', 'SAS', 1),
    Team('Minnesota Timberwolves', 'MIN', 1),
    Team('Denver Nuggets', 'DEN', 1),
    Team('Los Angeles Clippers', 'LAC', 1),
    Team('Los Angeles Lakers', 'LAL', 1),
    Team('Sacramento Kings', 'SAC', 1),
    Team('Dallas Mavericks', 'DAL', 1),
    Team('Memphis Grizzlies', 'MEM', 1),
    Team('Phoenix Suns', 'PHO', 1)
]


seasons = [
    Season('2015-16', '2016'),
    Season('2016-17', '2017'),
    Season('2017-18', '2018'),
    Season('2018-19', '2019')
]


players = [
    Player('LeBron', 'James', datetime.date(1984, 12, 30), 'jamesle01'),
    Player('DeMar', 'DeRozan', datetime.date(1989, 8, 7), 'derozde01'),
    Player('Russell', 'Westbrook', datetime.date(1988, 11, 12), 'westbru01')
]


player_season = [
    Player_Season(1, 1, 4, 31),
    Player_Season(1, 2, 4, 32),
    Player_Season(1, 3, 4, 33),
    Player_Season(1, 4, 26, 34),

    Player_Season(2, 1, 1, 26),
    Player_Season(2, 2, 1, 27),
    Player_Season(2, 3, 1, 28),
    Player_Season(2, 4, 22, 29),

    Player_Season(3, 1, 19, 27),
    Player_Season(3, 2, 19, 28),
    Player_Season(3, 3, 19, 29),
    Player_Season(3, 4, 19, 30),
]

# team db IDs
team_index_by_name = {}
team_index_by_code = {}
for x in range(1, 31):
    team_index_by_name[teams[x-1].name] = x
    team_index_by_code[teams[x-1].code] = x

# seasons db IDs
season_index_by_short = {}
for x in range(seasons.__len__()):
    season_index_by_short[seasons[x].short] = x+1
