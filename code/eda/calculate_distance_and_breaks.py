import pandas as pd
import numpy as np
from analysis_utils import load_data
import os


def load_distance_matrix(league):
    """
    Load the distance matrix of the desired league

    Parameters
    ----------
    league: str
        String indicating whose distance matrix we want to load

    Returns
    -------
    dist_matrix_df: pd.DataFrame
        Dataframe containing the distance matrix of a particular league
    """
    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\eda', f'data\\teams\\{league}')

    # Load dataframe and format date column
    dist_matrix_df = pd.read_csv(f'{file_dir}\\{league}_distances_matrix.csv')
    return dist_matrix_df


def turn_dist_matrix_into_dict(dist_matrix_df, teams):
    """
    We create a dictionary whose key are a tuple of a pair of teams and the values, the distance between them

    Parameters
    ----------
    dist_matrix_df: pd.DataFrame
        Dataframe containing the distance matrix of a particular league

    teams: list
        List of teams

    Returns
    -------
    dist_matrix: dict
        Dictionary with the distances between teams
    """
    dist_matrix = {}

    # We populate the dictionary
    for team_i in teams:
        for j in range(len(dist_matrix_df)):
            team_j = dist_matrix_df['Equipo'][j]
            dist_matrix[(team_i, team_j)] = dist_matrix_df[team_i][j]

    return dist_matrix


def calculate_distance_per_team(df, distance_matrix, team):
    """
    Calculate distance travelled by a particular team

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the historical results of the desired schedule
    dist_matrix: dict
        Dictionary with the distances between teams
    team: str
        String indicating the team whose stats we want to calculate

    Returns
    -------
    distance: float
        Total distance traveled by a team
    """
    df_filt = df[(df['home'] == team) | (df['visitor'] == team)].sort_values(by='game_date').reset_index(drop=True)

    # Calculate the day difference between a game and the previous
    df_filt.loc[:, 'previous_game'] = df_filt['game_date'].shift(1)
    df_filt.loc[:, 'day_diff'] = (df_filt['game_date'] - df_filt['previous_game']).dt.days

    # We calculate the first distance, the team and the location of the first team
    distance = distance_matrix[(team, df_filt['home'][0])]

    for i in range(1, len(df_filt) - 1):
        home_team = df_filt['home'][i]
        home_team_prev = df_filt['home'][i - 1]

        # If there are more than 3 days in between games, teams go home in between
        if df_filt['day_diff'][i] > 3:
            distance += distance_matrix[(team, home_team_prev)]
            distance += distance_matrix[(team, home_team)]
        else:
            distance += distance_matrix[(home_team_prev, home_team)]

    # We add one more distance, the distance between the last match home team and the team being analyzed
    distance += distance_matrix[(team, df_filt['home'][len(df_filt) - 1])]
    return distance


def calculate_distance(df, distance_matrix, teams):
    """
    Calculates total distance per team and for all teams

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the historical results of the desired schedule
    dist_matrix: dict
        Dictionary with the distances between teams
    teams: list
        List whose items are the teams of the league that is being analyzed

    Returns
    -------
    df_distance: pd.DataFrame
        Dataframe of distance per team and total
    """
    distances = []
    # Calculate distance per team
    for team in teams:
        distance = calculate_distance_per_team(df, distance_matrix, team)
        distances.append(distance)
    df_distance = pd.DataFrame(data={
        'Team': teams,
        'Distance': distances
    })
    # Calculate a total distance
    df_total_distance = pd.DataFrame(data={
        'Team': ['all'],
        'Distance': [np.sum(df_distance['Distance'])]
    })
    # Concat both dataframes
    df_distance = pd.concat([df_distance, df_total_distance], ignore_index=True)
    return df_distance


def calculate_breaks_per_team(df, team):
    """
    Calculate number of breaks per team

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the historical results of the desired schedule
    team: str
        String indicating the team whose stats we want to calculate

    Returns
    -------
    n_breaks: int
        Number of breaks a team has had
    """
    df_filt = df[(df['home'] == team) | (df['visitor'] == team)].sort_values(by='game_date').reset_index(drop=True)

    n_breaks = 0

    for i in range(1, len(df_filt) - 1):
        home_team = df_filt['home'][i]
        home_team_prev = df_filt['home'][i - 1]
        away_team = df_filt['visitor'][i]
        away_team_prev = df_filt['visitor'][i - 1]

        # If the home team or away team of two consecutive games is the same, then we add a break
        if home_team == home_team_prev and home_team == team:
            n_breaks += 1
        if away_team == away_team_prev and away_team == team:
            n_breaks += 1

    return n_breaks


def calculate_breaks(df, teams):
    """
    Calculates total distance per team and for all teams

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the historical results of the desired schedule
    teams: list
        List whose items are the teams of the league that is being analyzed

    Returns
    -------
    df_breaks: pd.DataFrame
        Dataframe of breaks per team and total
    """
    breaks = []
    # Calculate distance per team
    for team in teams:
        break_team = calculate_breaks_per_team(df, team)
        breaks.append(break_team)

    df_breaks = pd.DataFrame(data={
        'Team': teams,
        'Breaks': breaks
    })
    # Calculate total breaks
    df_total_breaks = pd.DataFrame(data={
        'Team': ['all'],
        'Breaks': [np.sum(df_breaks['Breaks'])]
    })
    # Concat both dataframes
    df_breaks = pd.concat([df_breaks, df_total_breaks], ignore_index=True)
    return df_breaks


if __name__ == '__main__':
    for league in ['nba', 'nhl']:
        df = load_data(league)
        dist_matrix_df = load_distance_matrix(league)
        teams = list(df['home'].unique())
        dist_matrix = turn_dist_matrix_into_dict(dist_matrix_df, teams)

        df_distance = calculate_distance(df, dist_matrix, teams)
        df_breaks = calculate_breaks(df, teams)
        df_distance_breaks = pd.merge(df_distance, df_breaks, how='left', on='Team')
        df_distance_breaks.to_csv(f'./results/{league}_distance_breaks.csv', index=False, encoding='utf-8 sig')
