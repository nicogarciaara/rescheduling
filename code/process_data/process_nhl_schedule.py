import pandas as pd
import os


def load_data(year):
    """
    Loads a dataframe with the schedule of a particular year

    Parameters
    ----------
    year: int
        Year of the schedule that wants to be loaded

    Returns
    -------
    df: pd.DataFrame
        Pandas dataframe with the schedule of the season that wants to be analyzed

    """
    # Set up directory
    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\process_data', 'data\\schedules\\nhl')

    # Read file
    return pd.read_csv(f'{file_dir}\\{year}_schedule.csv')


def output_data(df, year):
    """
    Saves processed information in disk

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe with the schedule of the season that wants to be analyzed, with formatted dates
    year: int
        Year of the schedule that wants to be loaded
    """
    # Set up directory
    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\process_data', 'data\\schedules\\nhl')
    df.to_csv(f'{file_dir}\\{year}_processed_schedule.csv', index=False, encoding='utf-8 sig')

    # Also output a teams df
    teams = list(df['visitor'].unique())
    df_teams = pd.DataFrame(data={'team': teams})
    file_dir = file_dir.replace('data\\schedules\\nhl', 'data\\teams\\nhl')
    df_teams.to_csv(f'{file_dir}\\nhl_teams.csv', index=False, encoding='utf-8 sig')


if __name__ == '__main__':
    for year in [2019, 2021]:
        df = load_data(year)
        df.loc[:, 'game_date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y').dt.date
        # Change column names to have the same format as in the nba
        df.rename(columns={'Visitor': 'visitor', 'Home': 'home'}, inplace=True)
        output_data(df, year)
