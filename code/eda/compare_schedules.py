import pandas as pd
import numpy as np
import os
import analysis_utils as au
from calculate_distance_and_breaks import *
from calculate_k_balance import *
import warnings
warnings.filterwarnings('ignore')


def load_schedules(league):
    """
    We laad our schedules, the planned schedule and the actual schedule

    Parameters
    ----------
    league: str
        String indicating the league whose schedule we want to load. Must be one of the following:
            - 'nba'
            - 'nhl'

    Returns
    -------
    df_planned: pd.DataFrame
        Dataframe containing the planned schedule, without reschedules
    df_actual: pd.DataFrame
        Dataframe containing the current schedule, the one that was actually played
    """
    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\eda', f'data\\schedules\\{league}')
    df_planned = pd.read_csv(f'{file_dir}\\{league}_original_schedule.csv')
    df_planned.loc[:, 'game_date'] = pd.to_datetime(df_planned['game_date'])
    df_actual = au.load_data(league)
    return df_planned, df_actual


def analyze_days_between_matches(df):
    """
    Function that calculates number of days between each match

    Parameters
    ----------
    df: pd.DataFrame
        Schedule of games

    Returns
    -------
    df_difference: pd.DataFrame
        Dataframe that has information by team, game_date, previous game_date and the difference
    """
    teams = list(df['home'].unique())
    df_difference = pd.DataFrame()

    for team in teams:
        df_team = df[(df['home'] == team) | (df['visitor'] == team)].sort_values(by='game_date')
        if len(df_team) > 0:
            df_team.loc[:, 'previous_game'] = df_team['game_date'].shift(1)
            df_team['Day Difference'] = (df_team['game_date'] - df_team['previous_game']).dt.days
            df_team.loc[:, 'Team'] = team
            df_difference = pd.concat([df_difference, df_team[['Team', 'game_date', 'previous_game', 'Day Difference']].dropna()],
                                      ignore_index=True)
    return df_difference


if __name__ == '__main__':
    distance_full = pd.DataFrame()
    breaks_full = pd.DataFrame()
    balance_full = pd.DataFrame()
    teams_with_re_full = pd.DataFrame()
    number_of_re_full = pd.DataFrame()
    difference_full = pd.DataFrame()
    schedule_full = pd.DataFrame()

    for league in ['nba', 'nhl']:
        df_planned, df_actual = load_schedules(league)
        dist_matrix_df = load_distance_matrix(league)
        teams = list(df_actual['home'].unique())
        dist_matrix = turn_dist_matrix_into_dict(dist_matrix_df, teams)

        # Get date range of the days of the tournament
        tournament_days = list(pd.date_range(
            start=np.min(df_actual['game_date']),
            end=np.max(df_actual['game_date']),
        ))

        # Calculate distance and breaks per schedule
        dict_df = {'Planned': df_planned, 'Actual': df_actual}

        distance_df = pd.DataFrame()
        breaks_df = pd.DataFrame()
        balance_df = pd.DataFrame()
        diff_df = pd.DataFrame()
        schedule_df = pd.DataFrame()

        for ix in dict_df:
            sched = dict_df[ix]
            sched = sched[['home', 'visitor', 'game_date']]
            sched.loc[:, 'Schedule Type'] = ix
            sched.loc[:, 'League'] = league.upper()
            schedule_df = pd.concat([schedule_df, sched], ignore_index=True)

            df_distance = calculate_distance(dict_df[ix], dist_matrix, teams)
            df_distance.loc[:, 'Schedule Type'] = ix
            df_distance.loc[:, 'League'] = league.upper()

            df_breaks = calculate_breaks(dict_df[ix], teams)
            df_breaks.loc[:, 'Schedule Type'] = ix
            df_breaks.loc[:, 'League'] = league.upper()

            df_balance = calculate_k_balance(dict_df[ix], league, teams, tournament_days, games='all')
            df_balance.loc[:, 'Schedule Type'] = ix
            df_balance.loc[:, 'League'] = league.upper()
            df_balance.loc[:, 'Balance 7-day rolling mean'] = df_balance['diff'].rolling(7, min_periods=1).mean()

            df_diff = analyze_days_between_matches(dict_df[ix])
            df_diff.loc[:, 'Schedule Type'] = ix
            df_diff.loc[:, 'League'] = league.upper()

            distance_df = pd.concat([distance_df, df_distance], ignore_index=True)
            breaks_df = pd.concat([breaks_df, df_breaks], ignore_index=True)
            balance_df = pd.concat([balance_df, df_balance], ignore_index=True)
            diff_df = pd.concat([diff_df, df_diff], ignore_index=True)

        teams_with_re = pd.read_csv(f'./results/{league}_teams_with_reschedules.csv')
        teams_with_re.loc[:, 'League'] = league.upper()
        numbers_of_re = pd.read_csv(f'./results/{league}_reschedules_by_team.csv')
        numbers_of_re.loc[:, 'League'] = league.upper()

        distance_full = pd.concat([distance_full, distance_df], ignore_index=True)
        breaks_full = pd.concat([breaks_full, breaks_df], ignore_index=True)
        balance_full = pd.concat([balance_full, balance_df], ignore_index=True)
        teams_with_re_full = pd.concat([teams_with_re_full, teams_with_re], ignore_index=True)
        number_of_re_full = pd.concat([number_of_re_full, numbers_of_re], ignore_index=True)
        difference_full = pd.concat([difference_full, diff_df], ignore_index=True)
        schedule_full = pd.concat([schedule_full, schedule_df], ignore_index=True)

        print(league, np.mean(df_planned['day_difference']))

    distance_full.to_csv('./output_for_tableau/distance_analysis.csv', index=False, encoding='utf-8 sig')
    breaks_full.to_csv('./output_for_tableau/breaks_analysis.csv', index=False, encoding='utf-8 sig')
    balance_full.to_csv('./output_for_tableau/balance_analysis.csv', index=False, encoding='utf-8 sig')
    difference_full.to_csv('./output_for_tableau/day_difference_analysis.csv', index=False, encoding='utf-8 sig')
    teams_with_re_full.to_csv('./output_for_tableau/teams_with_reschedules_analysis.csv', index=False, encoding='utf-8 sig')
    schedule_full.to_csv('./output_for_tableau/games_by_date_analysis.csv', index=False, encoding='utf-8 sig')

    number_of_re_full.to_csv('./output_for_tableau/number_of_reschedules_analysis.csv', index=False,
                             encoding='utf-8 sig')
