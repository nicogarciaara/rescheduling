import pandas as pd
import numpy as np
from analysis_utils import load_data
import statistics


def calculate_k_balance_team_day(df, team, day, games):
    """
    Calculates, per day, the number of games a team has played and the difference between the maximum amount of games
    played and the mode of that variable

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the full schedule of a regular season
    team: str
        String indicating the team being analyzed
    day: date
        Date up to we want to analyze
    games: str
        String indicating whose teams we want to consider. Must be one of the following:
            - 'home'
            - 'all'
    Returns
    -------
    n_games: int
        Returns the number of games a particular team has played until day

    """
    # Filter according to the games we want to consider
    if games == 'home':
        df_filt = df[df['home'] == team].reset_index(drop=True)
    else:
        df_filt = df[(df['visitor'] == team) | (df['home'] == team)].reset_index(drop=True)

    # Filter dates and count games
    df_filt = df_filt[df_filt['game_date'] <= day]
    n_games = df_filt.shape[0]
    return n_games


def calculate_k_balance(df, league, teams, tournament_days, games):
    """
    Calculates, per day, the number of games a team has played and the difference between the maximum amount of games
    played and the mode of that variable

    Parameters
    ----------
    df: pd.DataFrame
        Dataframe with the full schedule of a regular season

    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'
    teams: list
        List whose items are the teams of the league that is being analyzed
    tournament_days: list
        List of all days in which the competition takes place
    games: str
        String indicating whose teams we want to consider. Must be one of the following:
            - 'home'
            - 'all'
    """
    # Create a dictionary in which we will save the different results
    results_dict = {}
    for team in teams:
        results_dict[team] = []
    results_dict['day'] = []
    results_dict['mode'] = []

    # Calculate number of games played by each team until each day
    for day in tournament_days:
        results_dict['day'].append(day)
        games_per_team = []
        for team in teams:
            results_dict[team].append(calculate_k_balance_team_day(df, team, day, games))
            games_per_team.append(calculate_k_balance_team_day(df, team, day, games))
        try:
            results_dict['mode'].append(statistics.mode(games_per_team))
        except:
            games_per_team_df = pd.DataFrame(data={
                'games': games_per_team,
                'ok': [1]*len(games_per_team)
            })
            games_per_team_df = games_per_team_df.groupby('games').size().reset_index(name='n')
            games_per_team_df = games_per_team_df[games_per_team_df['games'] == np.max(games_per_team_df['games'])]

            results_dict['mode'].append(np.max(games_per_team_df['games']))

    # Generate dataframe that will be saved
    df_balance = pd.DataFrame(data={'Day': results_dict['day']})

    for team in teams:
        df_balance.loc[:, team] = results_dict[team]

    # Calculate mode and difference with the maximum
    df_balance.loc[:, 'mode'] = results_dict['mode']
    df_balance.loc[:, 'max'] = df_balance[teams].max(axis=1)
    df_balance.loc[:, 'diff'] = df_balance['max'] - df_balance['mode']
    #df_balance.to_csv(f'./results/{league}_k_balance_{games}.csv', index=False, encoding='utf-8 sig')
    return df_balance

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
        for games in ['home', 'all']:
            calculate_k_balance(df, league, teams, tournament_days, games)