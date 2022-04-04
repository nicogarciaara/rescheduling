import pandas as pd
import numpy as np
import os
import warnings
import sys
from model_utils import League
warnings.filterwarnings('ignore')
cwd = os.getcwd()
eda_wd = cwd.replace('models', 'eda')
sys.path.insert(0, eda_wd)
import analysis_utils as au
from calculate_distance_and_breaks import *
from calculate_k_balance import *
from compare_schedules import *
from scheduling_rules import *
import pandas as pd

if __name__ == '__main__':
    leagues = ['nba', 'nhl']
    objs = {'nba': ['basic', 'min_diff', 'balanced'], 'nhl': ['basic']}
    distance_mode = {'nba': ['low', 'mid', 'high'], 'nhl': []}

    # Read base CSVs that will be used for comparison (as they include the KPIs from the original schedules)
    distance_full = pd.read_csv(f'{eda_wd}/output_for_tableau/distance_analysis.csv')
    breaks_full = pd.read_csv(f'{eda_wd}/output_for_tableau/breaks_analysis.csv')
    balance_full = pd.read_csv(f'{eda_wd}/output_for_tableau/balance_analysis.csv')
    difference_full = pd.read_csv(f'{eda_wd}/output_for_tableau/day_difference_analysis.csv')
    schedule_full = pd.read_csv(f'{eda_wd}/output_for_tableau/games_by_date_analysis.csv')
    schedule_full['game_date'] = pd.to_datetime(schedule_full['game_date'])
    rules_full = pd.DataFrame()
    schedules_most_reschedules_teams = pd.DataFrame()

    for league in ['nba']:
        for obj in objs[league]:
            for distance_mode in ['low']:
                for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
                    #for reschedule_mode in ['monthly', 'post_all_star', 'ten_days']:
                    for reschedule_mode in ['monthly', 'post_all_star']:
                        check = 0
                        try:
                            df = pd.read_csv(f'./output/BasicModel_{league}_{obj}_{distance_mode}_{instance}_{reschedule_mode}.csv')
                        except:
                            df = 1

                        if type(df) != int:
                            if 'final_date' in df.columns:
                                df.rename(columns={'game_date': 'aux_date'}, inplace=True)
                                df.rename(columns={'final_date': 'game_date'}, inplace=True)
                                df['aux_date'] = pd.to_datetime(df['aux_date'])

                            df['game_date'] = pd.to_datetime(df['game_date'])
                            df['original_date'] = pd.to_datetime(df['original_date'])
                            df.rename(columns={'original_date': 'new_date'}, inplace=True)

                            # Read the original schedule
                            L = League(league)
                            schedule = L.load_schedule()

                            schedule['Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode}"
                            schedule['Schedule Owner'] = 'NBAs'
                            schedule['League'] = league.upper()

                            # Merge both dataframes and rename columns
                            df = pd.merge(df, schedule[['home', 'visitor', 'game_date', 'original_date']], how='left',
                                          on=['home', 'visitor', 'game_date'])
                            df.rename(columns={'game_date': 'final_date', 'new_date': 'game_date'}, inplace=True)
                            df['day_difference'] = (df['game_date'] - df['original_date']).dt.days
                            if instance == 'basic':
                                df['final_day_difference'] = (df['final_date'] - df['original_date']).dt.days
                            else:
                                df['final_day_difference'] = (df['aux_date'] - df['original_date']).dt.days

                            df_different = df[df['game_date'] != df['original_date']]
                            df_reschedule = df[(df['reschedule'] == 1) & (df['final_day_difference'] > 6)]

                            home_reschedules = df_different.groupby('home').size().reset_index(
                                name='reschedules').rename(columns={'home': 'team'})
                            away_reschedules = df_different.groupby('visitor').size().reset_index(
                                name='reschedules').rename(columns={'visitor': 'team'})
                            total_reschedules = pd.concat([home_reschedules, away_reschedules], ignore_index=True)
                            reschedules_by_team = total_reschedules.groupby('team')['reschedules'].sum().reset_index()
                            reschedules_by_team = reschedules_by_team.sort_values(by='reschedules', ascending=False)
                            top_reschedules = reschedules_by_team[reschedules_by_team['reschedules'] ==
                                                                  np.max(reschedules_by_team['reschedules'])]
                            teams_with_most_reschedules = list(top_reschedules['team'])
                            teams_with_most_reschedules = [teams_with_most_reschedules[0]]
                            df.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode}"
                            df.loc[:, 'Schedule Owner'] = 'Us'
                            df.loc[:, 'League'] = league.upper()

                            df_top = df[(df['home'].isin(teams_with_most_reschedules)) | (
                                df['visitor'].isin(teams_with_most_reschedules))]
                            schedule_top = schedule[(schedule['home'].isin(teams_with_most_reschedules)) | (
                                df['visitor'].isin(teams_with_most_reschedules))]

                            schedules_most_reschedules_teams = pd.concat([
                                schedules_most_reschedules_teams, df_top, schedule_top], ignore_index=True)
    
                            # Calculate the different KPIs, first defining the necessity
                            teams = list(df['home'].unique())
                            dist_matrix = L.get_distance_matrix()
                            tournament_days = list(pd.date_range(np.min(df['game_date']), np.max(df['game_date'])))
    
                            df_distance = calculate_distance(df, dist_matrix, teams)
                            df_distance.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode}"
                            df_distance.loc[:, 'League'] = league.upper()
    
                            df_breaks = calculate_breaks(df, teams)
                            df_breaks.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode}"
                            df_breaks.loc[:, 'League'] = league.upper()
    
                            df_balance = calculate_k_balance(df, league, teams, tournament_days, games='all')
                            df_balance.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode}"
                            df_balance.loc[:, 'League'] = league.upper()
                            df_balance.loc[:, 'Balance 7-day rolling mean'] = df_balance['diff'].rolling(
                                7, min_periods=1).mean()

                            df_diff = analyze_days_between_matches(df)
                            df_diff.loc[:, 'Schedule Type'] = f"{obj} - {distance_mode} - {instance} - {reschedule_mode}"
                            df_diff.loc[:, 'League'] = league.upper()
    
                            # Concat with original measurements
                            distance_full = pd.concat([distance_full, df_distance], ignore_index=True)
                            breaks_full = pd.concat([breaks_full, df_breaks], ignore_index=True)
                            balance_full = pd.concat([balance_full, df_balance], ignore_index=True)
                            difference_full = pd.concat([difference_full, df_diff], ignore_index=True)
    
                            for col in schedule_full.columns:
                                if col in df.columns:
                                    pass
                                else:
                                    df[col] = ''
                            df = df[list(schedule_full.columns)]
                            schedule_full = pd.concat([schedule_full, df], ignore_index=True)

                            # Calculate scheduling rules so we can validate them we are not making any mistake
                            """
                            df_max_days = calculate_max_games_per_team(df, tournament_days, teams)
                            df_max_days
                            df_b2b = calculate_number_of_back_to_backs(df, tournament_days, teams)
                            df_stats = pd.merge(df_max_days, df_b2b, how='left', on='Team')
            
                            # Create a max for each column
                            df_rules = pd.DataFrame()
                            df_rules.loc[:, 'Schedule Type'] = [f"{obj} - {distance_mode}"]
                            df_rules.loc[:, 'League'] = [league.upper()]
                            stats_columns = [x for x in df_stats.columns if x not in ['Team']]
                            for col in stats_columns:
                                df_rules[col] = np.max(df_stats[col])
                            rules_full = pd.concat([rules_full, df_rules], ignore_index=True)
                            """

    distance_full.to_csv('./results_output/distance_analysis.csv', index=False, encoding='utf-8 sig')
    breaks_full.to_csv('./results_output/breaks_analysis.csv', index=False, encoding='utf-8 sig')
    balance_full.to_csv('./results_output/balance_analysis.csv', index=False, encoding='utf-8 sig')
    difference_full.to_csv('./results_output/day_difference_analysis.csv', index=False, encoding='utf-8 sig')
    schedule_full.to_csv('./results_output/all_schedules.csv', index=False, encoding='utf-8 sig')
    rules_full.to_csv('./results_output/schedule_rules.csv', index=False, encoding='utf-8 sig')
    schedules_most_reschedules_teams.to_csv('./results_output/teams_with_more_reschedules.csv', index=False,
                                            encoding='utf-8 sig')


