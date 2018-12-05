import datetime
from Models.team import Team
from Models.player import Player
from Models.season import Season
from Models.player_season import Player_Season


teams = [
    Team('Toronto Raptors', 'TOR'),
    Team('Boston Celtics', 'BOS'),
    Team('Philadelphia 76ers', 'PHI'),
    Team('Cleveland Cavaliers', 'CLE'),
    Team('Indiana Pacers', 'IND'),
    Team('Miami Heat', 'MIA'),
    Team('Milwaukee Bucks', 'MIL'),
    Team('Washington Wizards', 'WAS'),
    Team('Detroit Pistons', 'DET'),
    Team('Charlotte Hornets', 'CHO'),
    Team('New York Knicks', 'NYK'),
    Team('Brooklyn Nets', 'BRK'),
    Team('Chicago Bulls', 'CHI'),
    Team('Orlando Magic', 'ORL'),
    Team('Atlanta Hawks', 'ATL'),
    Team('Houston Rockets', 'HOU'),
    Team('Golden State Warriors', 'GSW'),
    Team('Portland Trail Blazers', 'POR'),
    Team('Oklahoma City Thunder', 'OKC'),
    Team('Utah Jazz', 'UTA'),
    Team('New Orleans Pelicans', 'NOP'),
    Team('San Antonio Spurs', 'SAS'),
    Team('Minnesota Timberwolves', 'MIN'),
    Team('Denver Nuggets', 'DEN'),
    Team('Los Angeles Clippers', 'LAC'),
    Team('Los Angeles Lakers', 'LAL'),
    Team('Sacramento Kings', 'SAC'),
    Team('Dallas Mavericks', 'DAL'),
    Team('Memphis Grizzlies', 'MEM'),
    Team('Phoenix Suns', 'PHO')
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
