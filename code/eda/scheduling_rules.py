import pandas as pd
import numpy as np
from analysis_utils import load_data


def calculate_max_number_of_games_per_team_and_days(df, tournament_days, team, n_days, games='all'):
    """
    Calculates the maximum number of game plays in a range of days defined by n_days. The user can also decide if
    all games are considered for the calculation or only home/awau games

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the full schedule of a regular season
    tournament_days: list
        List of all days in which the competition takes place
    team: str
        String indicating the team whose stats we want to calculate
    n_days: int
        Number of days that will be considered for our interval
    games: str
        String indicating whose teams we want to consider. Must be one of the following:
            - 'home'
            - 'away'
            - 'all'

    Returns
    -------
    df_max_games: pd.DataFrame
        Dataframe which has the maximum amount of games that were played by a particular team
    """
    # Filter according to the games we want to consider
    if games == 'home':
        df_filt = df[df['home'] == team].reset_index(drop=True)
    elif games == 'away':
        df_filt = df[df['visitor'] == team].reset_index(drop=True)
    else:
        df_filt = df[(df['visitor'] == team) | (df['home'] == team)].reset_index(drop=True)

    max_games = 0

    # For each day in the tournament
    for i in range(len(tournament_days) - n_days + 1):
        # We check the range that we want to consider
        start = tournament_days[i]
        end = tournament_days[i + n_days - 1]

        # Filter dates
        df_filt_dates = df_filt[(df_filt['game_date'] >= start) & (df_filt['game_date'] <= end)]

        # Calculate number of games and check if it is the max
        n_games = df_filt_dates.shape[0]

        if n_games > max_games:
            max_games = n_games

            if n_days == 7 and max_games > 3:
                max_games

    # Create an output dataframw which has the team name and the maximum number of games
    df_max_games = pd.DataFrame({
        'Team': [team],
        f'Max_games_{n_days}_{games}': [max_games]
    })
    return df_max_games


def calculate_max_games_per_team(df, tournament_days, teams):
    """
    Calculate per team, the maximum number of home, away and overall games that are played

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the full schedule of a regular season
    tournament_days: list
        List of all days in which the competition takes place
    teams: list
        List whose items are the teams of the league that is being analyzed

    Returns
    -------
    df_max_days = pd.DataFrame
        DataFrame that has information per team, indicating the maximum of games that are being played within a
        range of days
    """
    # First we generate, initial dataframes for the trivial version of one day
    df_home_one = pd.DataFrame()
    df_away_one = pd.DataFrame()
    df_all_one = pd.DataFrame()
    for team in teams:
        df_max_days_home = calculate_max_number_of_games_per_team_and_days(df, tournament_days,
                                                                           team=team, n_days=1, games='home')
        df_max_days_away = calculate_max_number_of_games_per_team_and_days(df, tournament_days,
                                                                           team=team, n_days=1, games='away')
        df_max_days_all = calculate_max_number_of_games_per_team_and_days(df, tournament_days,
                                                                          team=team, n_days=1, games='all')
        df_home_one = pd.concat([df_home_one, df_max_days_home], ignore_index=True)
        df_away_one = pd.concat([df_away_one, df_max_days_away], ignore_index=True)
        df_all_one = pd.concat([df_all_one, df_max_days_all], ignore_index=True)

    # Merge initial results
    df_max_days = pd.merge(df_home_one, df_away_one, how='left', on='Team')
    df_max_days = pd.merge(df_max_days, df_all_one, how='left', on='Team')

    # Calculate for days 2 to 7
    for day in range(2, 8):
        df_home_day = pd.DataFrame()
        df_away_day = pd.DataFrame()
        df_all_day = pd.DataFrame()
        for team in teams:
            df_max_days_home = calculate_max_number_of_games_per_team_and_days(df, tournament_days,
                                                                               team=team, n_days=day, games='home')
            df_max_days_away = calculate_max_number_of_games_per_team_and_days(df, tournament_days,
                                                                               team=team, n_days=day, games='away')
            df_max_days_all = calculate_max_number_of_games_per_team_and_days(df, tournament_days,
                                                                              team=team, n_days=day, games='all')
            df_home_day = pd.concat([df_home_day, df_max_days_home], ignore_index=True)
            df_away_day = pd.concat([df_away_day, df_max_days_away], ignore_index=True)
            df_all_day = pd.concat([df_all_day, df_max_days_all], ignore_index=True)
        df_max_days = pd.merge(df_max_days, df_home_day, how='left', on='Team')
        df_max_days = pd.merge(df_max_days, df_away_day, how='left', on='Team')
        df_max_days = pd.merge(df_max_days, df_all_day, how='left', on='Team')

    return df_max_days


def calculate_number_of_back_to_backs_per_team(df, tournament_days, team, games):
    """
    Calculates the number of back to backs that a teams has

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the full schedule of a regular season
    tournament_days: list
        List of all days in which the competition takes place
    team: str
        String indicating the team whose stats we want to calculate
    games: str
        String indicating whose teams we want to consider. Must be one of the following:
            - 'home'
            - 'away'
            - 'all'

    Returns
    -------
    df_b2b_team: pd.DataFrame
        Dataframe which has the number of back to backs for a particular team
    """
    # Filter according to the games we want to consider
    if games == 'home':
        df_filt = df[df['home'] == team].reset_index(drop=True)
    elif games == 'away':
        df_filt = df[df['visitor'] == team].reset_index(drop=True)
    else:
        df_filt = df[(df['visitor'] == team) | (df['home'] == team)].reset_index(drop=True)

    b2b = 0

    # For each day in the tournament
    for i in range(len(tournament_days) - 1):
        # We check the range that we want to consider
        start = tournament_days[i]
        end = tournament_days[i + 1]

        # Filter dates
        df_filt_dates = df_filt[(df_filt['game_date'] >= start) & (df_filt['game_date'] <= end)]

        # Calculate number of games and check if it is the max
        n_games = df_filt_dates.shape[0]

        if n_games == 2:
            b2b += 1

    # Create an output dataframw which has the team name and the number of back to backs
    df_b2b_team = pd.DataFrame({
        'Team': [team],
        f'Back2Backs_{games}': [b2b]
    })
    return df_b2b_team


def calculate_number_of_back_to_backs(df, tournament_days, teams):
    """
    Calculate per team, the number of back to backs

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the full schedule of a regular season
    tournament_days: list
        List of all days in which the competition takes place
    teams: list
        List whose items are the teams of the league that is being analyzed

    Returns
    -------
    df_b2b = pd.DataFrame
        DataFrame that has information per team, indicating the number of back to backs
    """
    # Calculate per game condition
    df_home_one = pd.DataFrame()
    df_away_one = pd.DataFrame()
    df_all_one = pd.DataFrame()
    for team in teams:
        df_max_days_home = calculate_number_of_back_to_backs_per_team(df, tournament_days, team, games='home')
        df_max_days_away = calculate_number_of_back_to_backs_per_team(df, tournament_days, team, games='away')
        df_max_days_all = calculate_number_of_back_to_backs_per_team(df, tournament_days, team, games='all')
        df_home_one = pd.concat([df_home_one, df_max_days_home], ignore_index=True)
        df_away_one = pd.concat([df_away_one, df_max_days_away], ignore_index=True)
        df_all_one = pd.concat([df_all_one, df_max_days_all], ignore_index=True)

    # Merge results
    df_b2b = pd.merge(df_home_one, df_away_one, how='left', on='Team')
    df_b2b = pd.merge(df_b2b, df_all_one, how='left', on='Team')
    return df_b2b


if __name__ == '__main__':
    for league in ['nba', 'nhl']:
        df = load_data(league)

        # Get list of teams
        teams = list(df['home'].unique())

        # Get date range of the days of the tournament
        tournament_days = list(pd.date_range(
            start=np.min(df['game_date']),
            end=np.max(df['game_date']),
        ))

        df_max_days = calculate_max_games_per_team(df, tournament_days, teams)
        df_b2b = calculate_number_of_back_to_backs(df, tournament_days, teams)
        df_stats = pd.merge(df_max_days, df_b2b, how='left', on='Team')
        df_stats.to_csv(f'./results/{league}_schedule_rules.csv', index=False, encoding='utf-8 sig')
