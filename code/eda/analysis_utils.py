import os
import pandas as pd

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
    file_dir = file_dir.replace('code\\eda', f'data\\schedules\\{league}')

    # Load dataframe and format date column
    df = pd.read_csv(f'{file_dir}\\2021_processed_schedule.csv')
    df.loc[:, 'game_date'] = pd.to_datetime(df['game_date'])
    return df
