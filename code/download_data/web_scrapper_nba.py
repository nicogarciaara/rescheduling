import urllib.request
import calendar
import pandas as pd
from bs4 import BeautifulSoup
import os


''' auxiliar functions '''

def game_info(soup):
    '''gets a soup object and returns a dataframe with the game data'''
    games = soup.tbody.find_all('tr')
    rows = []

    for game in games:
        if len(game) == 1:
            pass
        else:
            row = []
            
            game_date = game.find('th', {"data-stat":"date_game"}).string
            game_date = game_date.split(',')
            
            game_day_week = game_date[0]
            game_season = game_date[2]
            
            game_time = game.find('td', {"data-stat":"game_start_time"}).string
            visitor = game.find('td', {"data-stat":"visitor_team_name"}).string
            home = game.find('td', {"data-stat":"home_team_name"}).string
            visitor_pts = game.find('td', {"data-stat":"visitor_pts"}).string
            home_pts = game.find('td', {"data-stat":"home_pts"}).string
            overtime = game.find('td', {"data-stat":"overtimes"}).string

            att = game.find('td', {"data-stat":"attendance"}).string
            try:
                att = int(att.replace(',',''))
            except:
                att = 0
            
            row.extend([game_day_week,
                        game_date[1],
                        game_season,
                        game_time,
                        visitor,
                        visitor_pts,
                        home,
                        home_pts,
                        overtime,
                        att])
            
            rows.append(row)

    df = pd.DataFrame(rows, columns= ['game_week_day',
                                          'game_date',
                                          'game_year',
                                          'game_time',
                                          'visitor',
                                          'visitor_pts',
                                          'home',
                                          'home_pts',
                                          'overtime',
                                          'att' ])
    return df

''' scrapper '''
if __name__ == '__main__':

    user_agent = 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
    headers = {'User-Agent': user_agent}
    df = pd.DataFrame()



    season_names = {2021: '2020/2021',
                    2019: '2018/19'}


    for year in season_names:
        if year == 2021:
            months = calendar.month_name[12:] + calendar.month_name[1:7]  # game seasons go from oct to jun
        else:
            months = calendar.month_name[10:] + calendar.month_name[1:7]  # game seasons go from oct to jun

        for month in months:
            url = 'https://www.basketball-reference.com/leagues/NBA_' + str(year) + '_games-' + month.lower() + '.html'
            try:
                req = urllib.request.Request(url, headers=headers)

                with urllib.request.urlopen(req) as response:
                    raw = response.read()
                print('request ok: ', url)
            except:
                print('error en url: ' + url)
                pass

            soup = BeautifulSoup(raw, "lxml")
            print('parsing completed')

            game_data = game_info(soup)
            game_data['season'] = season_names[year]

            df = pd.concat([df, game_data], axis = 0)
            print('game scrap completed: ' + str(year) + month)
            cmd = os.getcwd()
            cmd = cmd.replace('code/download_data', 'data/schedules/nba')
            df.to_csv(f"{cmd}/{year}_schedule.csv", index=False, encoding="utf-8 sig")