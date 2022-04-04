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
    file_dir = file_dir.replace('code\\process_data', 'data\\schedules\\nba')

    # Read file
    return pd.read_csv(f'{file_dir}\\{year}_schedule.csv')


def process_dates(df):
    """
    Formats the date column so that we can have a datetime in the game_date column

    Parameters
    ----------
    df: pd.DataFrame
        Pandas dataframe with the schedule of the season that wants to be analyzed

    Returns
    -------
    df: pd.DataFrame
        Pandas dataframe with the schedule of the season that wants to be analyzed, with formatted dates
    """
    # Take away unnecessary initial space
    df.loc[:, 'game_date'] = df['game_date'].str[1:]

    # Split column as we have month and day number
    month_and_day = df['game_date'].str.split(" ", n=1, expand=True)
    df.loc[:, 'game_month'] = month_and_day[0]
    df.loc[:, 'game_day'] = month_and_day[1]

    # Generate a dataframe with month name and numbers
    month_df = pd.DataFrame(data={
        'game_month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Oct', 'Nov', 'Dec'],
        'game_month_number': [1, 2, 3, 4, 5, 6, 10, 11, 12]
    })

    # Merge df
    df = pd.merge(df, month_df, how='left', on='game_month')

    # Generate final date column
    df_date = df[['game_year', 'game_month_number', 'game_day']]
    df_date.columns = ['year', 'month', 'day']
    df.loc[:, 'game_date'] = pd.to_datetime(df_date)
    df.loc[:, 'game_date'] = df['game_date'].dt.date

    return df


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
    file_dir = file_dir.replace('code\\process_data', 'data\\schedules\\nba')
    df.to_csv(f'{file_dir}\\{year}_processed_schedule.csv', index=False, encoding='utf-8 sig')

    # Also output a teams df
    teams = list(df['visitor'].unique())
    df_teams = pd.DataFrame(data={'team': teams})
    file_dir = file_dir.replace('data\\schedules\\nba', 'data\\teams\\nba')
    df_teams.to_csv(f'{file_dir}\\nba_teams.csv', index=False, encoding='utf-8 sig')


if __name__ == '__main__':
    for year in [2019, 2021]:
        df = load_data(year)
        df_processed = process_dates(df=df)
        output_data(df_processed, year)
