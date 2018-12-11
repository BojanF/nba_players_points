import numpy as np
import Services.repository as repo
import Services.csv_parse as csv
from Data.initial_data import team_index_by_code, team_index_by_name
from selenium import webdriver

# Create a new instance of the Chrome driver
driver = webdriver.Chrome('..\\Drivers\\Chrome\\chromedriver.exe')

# def games_for_team_in_given_seasons_old(driver, seasons, team_codes, xpathTable, range_from=1, range_to=82):
#     for season in seasons:
#         print('    Season: ' + season.full)
#         for code in team_codes:
#             print('         Team: ' + code)
#             url = 'https://www.basketball-reference.com/teams/' + code + '/' + season.short + '_games.html'
#             driver.get(url)
#
#             # get the table
#             table = driver.find_element_by_xpath(xpathTable)
#             # rows in selected table
#             rows = table.find_elements_by_xpath('./tr')
#
#
#             # if range_from!
#             count = 0
#
#             for row in rows:
#                 ths = row.find_elements_by_tag_name('th')
#                 tds = row.find_elements_by_tag_name('td')
#                 game = [td.text for td in tds]
#                 count += 1
#                 if game.__len__()!=0 and (count>=range_from and count<=range_to):
#                     game = np.insert(game, 0, ths[0].text)
#                     game[1] = tds[0].get_attribute('csk')
#                     # for win streak - default is W 1
#                     #  so we want to be without space - W1
#                     game[13] = game[13].replace(' ', '')
#                     if game[5] == '@':
#                         teamHA = 'A'
#                     else:
#                         teamHA = 'H'
#                     game[5] = teamHA
#                     repo.insert_game(season_index_by_short[season.short], team_index_by_code[code], team_index_by_name[game[6]], game)
#
#                 if game.__len__() == 0:
#                     # rows thad don`t represent game
#                     count -= 1
#
#                 # if count == rows_limit:
#                 if count == range_to:
#                     break


def games_for_players_in_given_seasons(players, seasons, xpath_table, range_from=1, range_to=82):
    for season in seasons:
        print('Insert games for season ', str(season))
        for player in players:
            games_for_player(player, season, xpath_table, range_from, range_to)


def games_for_player(player_slug, season_p, xpath_table, range_from=1, range_to=82):
    print('     Insert games for player ', str(player_slug), ' -> [' + str(range_from) + ':' + str(range_to) + ']')
    url = 'https://www.basketball-reference.com/players/a/' + player_slug + '/gamelog/' + season_p
    driver.get(url)
    # xpathTable = '//*[@id="pgl_basic"]/tbody'

    # get the table
    table = driver.find_element_by_xpath(xpath_table)
    # rows in selected table
    rows = table.find_elements_by_xpath('./tr')

    # eliminating rows that are not representation of games
    rows = rows[0:20] + rows[21:41] + rows[42:62] + rows[63:83] + rows[84:86]

    # get player id
    player_id = repo.get_player_id_by_slug(player_slug)

    # get season
    season = repo.get_season_by_short_representation(season_p)

    player_season = None

    previous_game = []

    # range of games
    if range_from != 1:
        # range > 1
        # get previous game
        previous_game = player_game_html_row_parse(rows[range_from-2])[0]
        # get player_season id
        player_season = repo.get_player_season_id(player_id, season[0], previous_game[4])

    # csv file path
    file = '..\\Margins_And_Odds\\' + season_p + '\\' + player_slug + '.csv'

    if not (range_from == 1 and range_to == 82):
        rows = rows[range_from - 1: range_to]
        odds = csv.parse_csv_file(file, range_from, range_to)
    else:
        odds = csv.parse_csv_file(file)

    for row in rows:
        ths = row.find_elements_by_tag_name('th')
        team_game = ths[0].text

        odd_index = int(team_game) - range_from
        parsed_data = player_game_html_row_parse(row, odds[odd_index][4:7])
        game = parsed_data[0]
        points = parsed_data[1]
        pts_margin_and_odds = parsed_data[2]

        team_win_pct_streak = ()
        opponent_previous_game = ()
        opponent_win_pct_streak = ()

        if game[0] != '1':

            if game[4] != previous_game[4]:
                player_season = repo.get_player_season_id(player_id, season[0], game[4])

            team_win_pct_streak = repo.get_team_win_pct_and_streak_after_game(int(game[0]) - 1,
                                                                              player_season[3],
                                                                              season[0])

            opponent_previous_game = repo.find_game(game[6], player_season[3], game[2])

            opponent_win_pct_streak = repo.get_team_win_pct_and_streak_after_game(opponent_previous_game[2] - 1,
                                                                                  game[6],
                                                                                  season[0])

        if game[0] != '1' and game[0] != '2':
            # print('game > 3')
            opponent_win_pct_streak_lg = repo.get_team_win_pct_and_streak_after_game(opponent_previous_game[2]-2,
                                                                                     game[6],
                                                                                     season[0])
            if previous_game[1] is not None:
                persist_game = [game[0], game[2], game[5], game[6], player_season[0], game[1], 1, previous_game[8],
                                previous_game[9].split(':')[0], previous_game[10], previous_game[11], previous_game[12],
                                previous_game[13], previous_game[14], previous_game[15], previous_game[16],
                                previous_game[17], previous_game[18], previous_game[19], previous_game[20],
                                previous_game[22], previous_game[23], previous_game[24], previous_game[25],
                                previous_game[26], previous_game[27], previous_game[28], previous_game[29],
                                team_win_pct_streak[0], team_win_pct_streak[1], opponent_win_pct_streak[0],
                                opponent_win_pct_streak[1], opponent_win_pct_streak_lg[1], points] + pts_margin_and_odds
            else:
                blank = [None] * 21
                persist_game = [game[0], game[2], game[5], game[6], player_season[0], game[1], 0] + blank +\
                                [team_win_pct_streak[0], team_win_pct_streak[1], opponent_win_pct_streak[0],
                                opponent_win_pct_streak[1], opponent_win_pct_streak_lg[1], points] + pts_margin_and_odds
            previous_game = game
        elif game[0] == '2':
            # print('game 2')
            if previous_game[1] is not None:
                persist_game = [game[0], game[2], game[5], game[6], player_season[0], game[1], 1, previous_game[8],
                                previous_game[9].split(':')[0], previous_game[10], previous_game[11], previous_game[12],
                                previous_game[13], previous_game[14], previous_game[15], previous_game[16],
                                previous_game[17], previous_game[18], previous_game[19], previous_game[20],
                                previous_game[22], previous_game[23], previous_game[24], previous_game[25],
                                previous_game[26], previous_game[27], previous_game[28], previous_game[29],
                                team_win_pct_streak[0], team_win_pct_streak[1], opponent_win_pct_streak[0],
                                opponent_win_pct_streak[1], None, points] + pts_margin_and_odds
            else:
                blank = [None] * 21
                persist_game = [game[0], game[2], game[5], game[6], player_season[0], game[1], 0]\
                                + blank + \
                                [team_win_pct_streak[0], team_win_pct_streak[1], opponent_win_pct_streak[0],
                                opponent_win_pct_streak[1], None, points] + pts_margin_and_odds
            previous_game = game
        else:
            # print('game 1')
            previous_game = game
            blank = [None] * 25
            player_season = repo.get_player_season_id(player_id, season[0], game[4])
            persist_game = [game[0], game[2], game[5], game[6], player_season[0], game[1], 0] + blank + [None, points] + pts_margin_and_odds

        repo.insert_player_game(persist_game)


def games_for_team_in_given_seasons(seasons, team_codes, xpath_table, range_from=1, range_to=82):
    for season in seasons:
        print('    Season: ' + season[1])
        for code in team_codes:
            print('         Team: ' + code + ' -> ' + repr(range_from) + ':' + repr(range_to))
            url = 'https://www.basketball-reference.com/teams/' + code + '/' + season[2] + '_games.html'
            driver.get(url)

            # get the table
            table = driver.find_element_by_xpath(xpath_table)
            # rows in selected table
            rows = table.find_elements_by_xpath('./tr')

            # eliminating rows that are not representation of games
            rows = rows[0:20] + rows[21:41] + rows[42:62] + rows[63:83] + rows[84:86]

            # range of games
            if not(range_from == 1 and range_to == 82):
                rows = rows[range_from-1:range_to]
            # print(rows.__len__())

            for row in rows:
                # get th and td from html table row
                ths = row.find_elements_by_tag_name('th')
                tds = row.find_elements_by_tag_name('td')

                # every td cell value from html row in stored in array cell
                game = [td.text for td in tds]

                # set number of in season for team
                game = np.insert(game, 0, ths[0].text)

                # get date form attribute in form YYYY-MM-DD
                game[1] = tds[0].get_attribute('csk')

                # for win streak - default is W 1
                #  so we want to be without space - W1
                game[13] = game[13].replace(' ', '')

                if game[5] == '@':
                    team_ha = 'A'
                else:
                    team_ha = 'H'
                game[5] = team_ha

                # if this cell is empty => game is not played yet
                # cycle needs to end here
                if game[4] == '':
                    break

                repo.insert_game(season[0], team_index_by_code[code], team_index_by_name[game[6]], game)


def player_game_html_row_parse(row, pts_margin_and_odds=None):

    # if pts_margin_and_odds is None:
    #     pts_margin_and_odds = []
    ths = row.find_elements_by_tag_name('th')
    tds = row.find_elements_by_tag_name('td')

    # every td cell value from html row in stored in array cell
    game = [td.text for td in tds]

    # set game in season for team
    game = [ths[0].text] + game

    # set opponent team id
    game[6] = repo.get_team_id_by_code(game[6])

    if game[5] == '@':
        team_ha = 'A'
    else:
        team_ha = 'H'
    game[5] = team_ha

    points = None

    if game[1] == '':
        game[1] = None
        pts_margin_and_odds += [None]
    else:
        # FG%
        if game[12] == '':
            game[12] = None
        # 3PT%
        if game[15] == '':
            game[15] = None
        # FT%
        if game[18] == '':
            game[18] = None
        points = game[27]
        # if pts_margin_and_odds.__len__() > 0:
        if pts_margin_and_odds is not None:
            if float(points) > float(pts_margin_and_odds[0]):
                pts_margin_and_odds += ['OVER']
            else:
                pts_margin_and_odds += ['UNDER']

    return game, points, pts_margin_and_odds
