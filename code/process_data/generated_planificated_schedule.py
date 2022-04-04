import pandas as pd
import os
import numpy as np


def load_data(league):
    """
    Function that loads the processed schedule of the desired league

    Parameters
    ----------
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'

    Returns
    -------
    df: pd.DataFrame
        Dataframe with the historical results of the desired schedule
    """
    # Set up directory where we will look for the data
    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\process_data', f'data\\schedules\\{league}')

    # Load dataframe and format date column
    df = pd.read_csv(f'{file_dir}\\2021_processed_schedule.csv')
    df.loc[:, 'game_date'] = pd.to_datetime(df['game_date'])
    return df


def load_rescheduled_games(league):
    """
    Load rescheduled games for a particular league

    Parameters
    ----------
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'

    Returns
    -------
    df_rescheduled: pd.DataFrame
        Dataframe with information of rescheduled games
    """
    # Set up directory where we will look for the data
    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\process_data', f'resources')

    # Load dataframe and format date column
    if league == 'nba':
        df_rescheduled = pd.read_excel(f'{file_dir}\\{league}_reschedules.xlsx', sheet_name='matches')
    else:
        df_rescheduled = pd.read_csv(f'{file_dir}\\{league}_reschedules.csv')
    df_rescheduled.loc[:, 'game_date'] = pd.to_datetime(df_rescheduled['game_date'])
    df_rescheduled.loc[:, 'original_date'] = pd.to_datetime(df_rescheduled['original_date'])
    df_rescheduled.loc[:, 'reschedule'] = 1
    return df_rescheduled


if __name__ == '__main__':
    for league in ['nba', 'nhl']:
        df = load_data(league)

        file_dir = os.getcwd()
        file_dir = file_dir.replace('code\\process_data', f'data\\schedules\\{league}')

        df_rescheduled = load_rescheduled_games(league)
        df = pd.merge(df, df_rescheduled, how='left', on=['home', 'visitor', 'game_date'])
        df.loc[:, 'reschedule'] = df['reschedule'].fillna(0)
        df['day_difference'] = (df['game_date'] - df['original_date']).dt.days
        df.to_csv(f'{file_dir}/{league}_original_and_true_schedule.csv', index=False, encoding='utf-8 sig')

        # Generate new date
        df.loc[df['reschedule'] == 1, 'game_date'] = df['original_date']
        df.loc['game_date'] = df['game_date'].fillna(df['original_date'])

        df = df.sort_values(by=['game_date']).reset_index(drop=True)
        df.to_csv(f'{file_dir}\\{league}_original_schedule.csv', index=False, encoding='utf-8 sig')
