import pandas as pd
import numpy as np
import datetime


if __name__ == '__main__':
    file_dir = 'C:/Users/HP/Documents/Sports Analytics/Re Scheduling/resources'
    df = pd.read_csv(f'{file_dir}/nhl_rescheduled_games.txt')

    original_dates_months = []
    original_dates_days = []
    homes = []
    visitors = []
    game_dates_months = []
    game_dates_days = []

    for index, row in df.iterrows():
        if len(row['Games']) < 15:
            original_date = row['Games']
            original_date_month = original_date.split(' ')[0]
            original_date_day = original_date.split(' ')[1]
        else:
            original_dates_days.append(int(original_date_day))
            original_dates_months.append(original_date_month)

            teams = row['Games'].split(':')[0]
            home = teams.split(' at ')[1]
            away = teams.split(' at ')[0]
            homes.append(home)
            visitors.append(away)
            try:
                full_date = row['Games'].split(' to ')[1]
                game_month = full_date.split(' ')[0]
                game_day = full_date.split(' ')[1]
                game_dates_months.append(game_month)
                game_dates_days.append(int(game_day))
            except:
                game_dates_months.append(np.nan)
                game_dates_days.append(np.nan)

    df_reschedules = pd.DataFrame(data={
        'original_month': original_dates_months,
        'original_day': original_dates_days,
        'home_team': homes,
        'visitor_team': visitors,
        'game_month': game_dates_months,
        'game_day': game_dates_days
    })
    months = list(df_reschedules['game_month'].unique()) + list(df_reschedules['original_month'].unique())
    months = list(set(months))
    months_df_o = pd.DataFrame(data={
        'original_month': ['Jan.', 'May', 'April', 'Feb.', 'March'],
        'original_month_number': [1, 5, 4, 2, 3]
    })
    months_df_g = pd.DataFrame(data={
        'game_month': ['Jan.', 'May', 'April', 'Feb.', 'March'],
        'game_month_number': [1, 5, 4, 2, 3]
    })
    # Merge
    df_reschedules = pd.merge(df_reschedules, months_df_o, how='left', on='original_month')
    df_reschedules = pd.merge(df_reschedules, months_df_g, how='left', on='game_month')
    df_reschedules = df_reschedules.dropna()
    for col in ['game_month_number', 'original_month_number', 'game_day', 'original_day']:
        df_reschedules[col] = df_reschedules[col].astype(int)
    # Read teams
    df_teams = pd.read_csv('C:/Users/HP/Documents/Sports Analytics/Re Scheduling/data/teams/nhl/nhl_teams.csv')
    teams = list(df_teams['team'])

    visitors_corr = []
    homes_corr = []
    original_dates = []
    game_dates = []
    for index, row in df_reschedules.iterrows():
        original_dates.append(datetime.datetime(2021, row['original_month_number'], row['original_day']))
        game_dates.append(datetime.datetime(2021, row['game_month_number'], row['game_day']))
        visitor = row['visitor_team']
        home = row['home_team']
        for t in teams:
            if visitor in t:
                visitor_corr = t
            if home in t:
                home_corr = t
        visitors_corr.append(visitor_corr)
        homes_corr.append(home_corr)
    df_reschedules['original_date'] = original_dates
    df_reschedules['game_date'] = game_dates
    df_reschedules['home'] = homes_corr
    df_reschedules['visitor'] = visitors_corr
    df_reschedules['source'] = 'https://www.sportsnet.ca/nhl/article/nhl-covid-19-schedule-changes-every-postponed-game-2020-21/'
    df_reschedules
    df_reschedules.to_csv(f'{file_dir}/nhl_reschedules.csv', index=False, encoding='utf-8 sig')

