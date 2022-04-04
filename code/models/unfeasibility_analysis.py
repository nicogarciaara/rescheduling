import pandas as pd
import pickle
import pandas as pd
from model_utils import League

if __name__ == '__main__':
    objective = 'basic'
    distance_mode = 'low'
    league = 'nba'
    team = 'Memphis Grizzlies'
    df = pd.read_csv(f'./output/BasicModel_{league}_{objective}_{distance_mode}.csv')
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['original_date'] = pd.to_datetime(df['original_date'])
    df.rename(columns={'original_date': 'new_date'}, inplace=True)

    # Read the original schedule
    L = League(league)
    schedule = L.load_schedule()
    schedule_team = schedule[(schedule['home'] == team) | (schedule['visitor'] == team)]
    schedule_team
    # Merge both dataframes and rename columns

    df = pd.merge(df, schedule[['home', 'visitor', 'game_date', 'original_date']], how='left',
                  on=['home', 'visitor', 'game_date'])
    df.rename(columns={'game_date': 'final_date', 'new_date': 'game_date'}, inplace=True)

    df_team = df[(df['home'] == team) | (df['visitor'] == team)]
    dates = df_team.groupby(['game_date']).size().reset_index()
    dates