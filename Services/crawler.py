import numpy as np
import Services.repository as repo
from Data.initial_data import team_index_by_code, team_index_by_name
from selenium import webdriver

# Create a new instance of the Chrome driver
driver = webdriver.Chrome('..\\Drivers\\Chrome\\chromedriver.exe')

# def games_for_team_in_given_seasons_old(connection, driver, seasons, team_codes, xpathTable, range_from=1, range_to=82):
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
#                     repo.insert_game(connection, season_index_by_short[season.short], team_index_by_code[code], team_index_by_name[game[6]], game)
#
#                 if game.__len__() == 0:
#                     # rows thad don`t represent game
#                     count -= 1
#
#                 # if count == rows_limit:
#                 if count == range_to:
#                     break


def games_for_team_in_given_seasons(connection, seasons, team_codes, xpathTable, range_from=1, range_to=82):
    for season in seasons:
        print('    Season: ' + season[1])
        for code in team_codes:
            print('         Team: ' + code + ' -> ' + repr(range_from) + ':' + repr(range_to))
            url = 'https://www.basketball-reference.com/teams/' + code + '/' + season[2] + '_games.html'
            driver.get(url)

            # get the table
            table = driver.find_element_by_xpath(xpathTable)
            # rows in selected table
            rows = table.find_elements_by_xpath('./tr')

            # eliminating rows that are not representation of games
            rows = rows[0:20] + rows[21:41] + rows[42:62] + rows[63:83] + rows[84:86]

            # range of games
            if not(range_from==1 and range_to==82):
                rows = rows[range_from-1 : range_to]
            # print(rows.__len__())

            for row in rows:
                # get th and td from html table row
                ths = row.find_elements_by_tag_name('th')
                tds = row.find_elements_by_tag_name('td')

                # every cell value from html row in stored in array cell
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

                # if this cell is empty game is not played
                # cycle needs to end here
                if game[4] == '':
                    break

                repo.insert_game(connection, season[0], team_index_by_code[code], team_index_by_name[game[6]], game)
