import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def load_original_schedule(league):
    """
    Loads original schedule, pre-covid reschedules

    Parameters
    ----------
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'

    Returns
    -------
    df: pd.DataFrame
        Dataframe with original schedule
    """
    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\eda', f'data\\schedules\\{league}')
    df = pd.read_csv(f'{file_dir}\\{league}_original_schedule.csv')
    return df


def reschedules_by_team(df, league):
    """
    Calculates number of reschedules by team

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with original schedule
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'

    """
    # We do the calculation twice, one with home games and one with away ones
    df_home_reschedules = df.groupby('home')['reschedule'].sum().reset_index()
    df_home_reschedules.rename(columns={'home': 'team', 'reschedule': 'home_reschedule'}, inplace=True)

    df_visitor_reschedules = df.groupby('visitor')['reschedule'].sum().reset_index()
    df_visitor_reschedules.rename(columns={'visitor': 'team', 'reschedule': 'away_reschedule'}, inplace=True)

    df_reschedules = pd.merge(df_home_reschedules, df_visitor_reschedules, how='outer', on='team')
    df_reschedules = df_reschedules.fillna(0)
    df_reschedules.loc[:, 'reschedules'] = df_reschedules['home_reschedule'] + df_reschedules['away_reschedule']
    df_reschedules.to_csv(f'./results/{league}_reschedules_by_team.csv', index=False, encoding='utf-8 sig')


def teams_with_reschedules(df, league):
    """
    Calculates number of teams with reschedules

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with original schedule
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'

    """
    # We do the calculation twice, one with home games and one with away ones
    df_home_reschedules = df.groupby('home')['reschedule'].max().reset_index()
    df_home_reschedules.rename(columns={'home': 'team', 'reschedule': 'home_reschedule'}, inplace=True)

    df_visitor_reschedules = df.groupby('visitor')['reschedule'].max().reset_index()
    df_visitor_reschedules.rename(columns={'visitor': 'team', 'reschedule': 'away_reschedule'}, inplace=True)

    df_reschedules = pd.merge(df_home_reschedules, df_visitor_reschedules, how='outer', on='team')
    df_reschedules = df_reschedules.fillna(0)
    df_reschedules.loc[:, 'reschedules'] = df_reschedules[['home_reschedule', 'away_reschedule']].max(axis=1)
    df_reschedules.to_csv(f'./results/{league}_teams_with_reschedules.csv', index=False, encoding='utf-8 sig')


def calculate_probability_of_reschedule(df, league):
    """
    Calculates the probability of a match being rescheduled, given that the previous match was rescheduled

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with original schedule
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'
    """
    teams = list(df['home'].unique())

    df_rescheduled = pd.DataFrame()

    for team in teams:
        df_team = df[(df['home'] == team) | (df['visitor'] == team)].sort_values(by='game_date')
        df_team.loc[:, 'prev_reschedule'] = df_team['reschedule'].shift(1)
        df_team_reschedule = df_team[df_team['reschedule'] == 1]
        if len(df_team_reschedule) > 0:
            df_team_reschedule.loc[:, 'team'] = team
            df_rescheduled = pd.concat([df_rescheduled, df_team_reschedule[['team', 'reschedule', 'prev_reschedule']]], ignore_index=True)
    df_rescheduled = df_rescheduled.fillna(0)
    prev_re = np.sum(df_rescheduled['prev_reschedule'])
    print(f'{league}: probability of prev suspended given suspended: {prev_re/len(df_rescheduled)}')


def calculate_probability_of_next_reschedule(df, league):
    """
    Calculates the probability of a match being rescheduled, given that the previous match was rescheduled

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with original schedule
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'
    """
    teams = list(df['home'].unique())

    df_rescheduled = pd.DataFrame()

    for team in teams:
        df_team = df[(df['home'] == team) | (df['visitor'] == team)].sort_values(by='game_date')
        df_team.loc[:, 'next_reschedule'] = df_team['reschedule'].shift(-1)
        df_team_reschedule = df_team[df_team['reschedule'] == 1]
        if len(df_team_reschedule) > 0:
            df_team_reschedule.loc[:, 'team'] = team
            df_rescheduled = pd.concat([df_rescheduled, df_team_reschedule[['team', 'reschedule', 'next_reschedule']]], ignore_index=True)
    df_rescheduled = df_rescheduled.fillna(0)
    next_re = np.sum(df_rescheduled['next_reschedule'])
    print(f'{league}: probability of next suspended given suspended: {next_re/len(df_rescheduled)}')


if __name__ == '__main__':
    for league in ['nba']:
        df = load_original_schedule(league)

        # Calculate number of reschedules per team
        reschedules_by_team(df, league)

        # Calculate number of teams with reschedules
        teams_with_reschedules(df, league)

        calculate_probability_of_reschedule(df, league)
        calculate_probability_of_next_reschedule(df, league)
