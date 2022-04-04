from basic_model import BasicModel
from model_utils import League
import warnings
import datetime
import cplex
import pandas as pd
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    leagues = ['nba']
    objectives = ['basic', 'min_diff', 'balanced']
    for distance_mode in ['low']:
        #for instance in ['basic', '15_more_games', '25_more_games', '15_games_in_march']:
        for instance in ['15_more_games', '25_more_games', '15_games_in_march']:
            #for reschedule_mode in ['monthly', 'post_all_star', 'ten_days']:
            for reschedule_mode in ['post_all_star']:
                for league in leagues:
                    for objective in objectives:
                        if league == 'nba':
                            if reschedule_mode == 'monthly':
                                starts = [datetime.datetime(2020, 12, 1),
                                          datetime.datetime(2021, 1, 1),
                                          datetime.datetime(2021, 2, 1),
                                          datetime.datetime(2021, 3, 1),
                                          datetime.datetime(2021, 4, 1),
                                          datetime.datetime(2021, 5, 1)]
                                ends = [datetime.datetime(2020, 12, 31),
                                        datetime.datetime(2021, 1, 31),
                                        datetime.datetime(2021, 2, 28),
                                        datetime.datetime(2021, 3, 31),
                                        datetime.datetime(2021, 4, 30),
                                        datetime.datetime(2021, 5, 31)]
                            elif reschedule_mode == 'post_all_star':
                                starts = [datetime.datetime(2020, 12, 1),
                                          datetime.datetime(2021, 3, 1),
                                          datetime.datetime(2021, 4, 1),
                                          datetime.datetime(2021, 5, 1)]
                                ends = [datetime.datetime(2021, 2, 28),
                                        datetime.datetime(2021, 3, 31),
                                        datetime.datetime(2021, 4, 30),
                                        datetime.datetime(2021, 5, 31)]
                            else:
                                starts = [datetime.datetime(2020, 12, 1),
                                          datetime.datetime(2021, 1, 1),
                                          datetime.datetime(2021, 1, 11),
                                          datetime.datetime(2021, 1, 21),
                                          datetime.datetime(2021, 2, 1),
                                          datetime.datetime(2021, 2, 11),
                                          datetime.datetime(2021, 2, 21),
                                          datetime.datetime(2021, 3, 1),
                                          datetime.datetime(2021, 3, 11),
                                          datetime.datetime(2021, 3, 21),
                                          datetime.datetime(2021, 4, 1),
                                          datetime.datetime(2021, 4, 11),
                                          datetime.datetime(2021, 4, 21),
                                          datetime.datetime(2021, 5, 1),
                                          datetime.datetime(2021, 5, 11),
                                          datetime.datetime(2021, 5, 21)
                                          ]
                                ends = [datetime.datetime(2020, 12, 31),
                                        datetime.datetime(2021, 1, 10),
                                        datetime.datetime(2021, 1, 20),
                                        datetime.datetime(2021, 1, 31),
                                        datetime.datetime(2021, 2, 10),
                                        datetime.datetime(2021, 2, 20),
                                        datetime.datetime(2021, 2, 28),
                                        datetime.datetime(2021, 3, 10),
                                        datetime.datetime(2021, 3, 20),
                                        datetime.datetime(2021, 3, 31),
                                        datetime.datetime(2021, 4, 10),
                                        datetime.datetime(2021, 4, 20),
                                        datetime.datetime(2021, 4, 30),
                                        datetime.datetime(2021, 5, 10),
                                        datetime.datetime(2021, 5, 20),
                                        datetime.datetime(2021, 5, 31)]
                        else:
                            starts = [datetime.datetime(2021, 1, 1), datetime.datetime(2021, 2, 1),
                                      datetime.datetime(2021, 3, 1), datetime.datetime(2021, 4, 1), datetime.datetime(2021, 5, 1)]
                            ends = [datetime.datetime(2021, 1, 31), datetime.datetime(2021, 2, 28),
                                    datetime.datetime(2021, 3, 31), datetime.datetime(2021, 4, 30), datetime.datetime(2021, 5, 31)]

                        L = League(league)
                        if instance == 'basic':
                            fixture = L.load_schedule()
                        else:
                            fixture = pd.read_csv(f'./other_instances/nba_schedule_{instance}.csv')
                            fixture['original_date'] = pd.to_datetime(fixture['original_date'])
                            fixture['game_date'] = pd.to_datetime(fixture['game_date'])
                            fixture['final_date'] = pd.to_datetime(fixture['final_date'])

                        all_needed_reschedules = []

                        # We make reschedules for the matches of that month

                        for s in range(len(starts)):
                            check = 0
                            start_date = starts[s]
                            end_date = ends[s]
                            print(f"Rescheduling matches for league {league} - objective {objective},  "
                                  f"between {start_date.date()} and {end_date.date()} - distance mode {distance_mode} "
                                  f"- instance {instance} - reschedule_mode {reschedule_mode}")
                            match_buffer = []
                            # Create the model and the lp problem
                            M = BasicModel(league, custom_fixture=fixture, start_date=start_date,
                                           distance_mode=distance_mode)
                            dis = M.disruptions
                            prob_lp = cplex.Cplex()

                            # Check if there are any additional reschedules needed
                            for match in all_needed_reschedules:
                                if start_date <= match['original_date'] <= end_date:
                                    match_buffer.append(match)
                                    all_needed_reschedules.remove(match)
                            match_buffer
                            # We create the variables that will go into the model
                            if start_date < datetime.datetime(2021, 5, 10):
                                x_var_dict, matches_to_be_scheduled, \
                                diff_games_dict, non_matched_matches = M.create_decision_variables_dict(
                                    start_date=start_date,
                                    end_date=end_date,
                                    objective=objective,
                                    match_buffer=match_buffer
                                )
                            else:
                                non_matched_matches = match_buffer
                            if len(non_matched_matches) > 0:
                                non_matched_output_df = M.assign_unassigned_matches(non_matched_matches, fixture)

                                fixture = pd.merge(fixture, non_matched_output_df[['home', 'visitor', 'game_date',
                                                                       'proposed_date', 'hard_reschedule']],
                                                   how='left',
                                                   on=['home', 'visitor', 'game_date'])
                                fixture
                                # Update date on schedule in order to consider new dates in the feasibility calculation
                                fixture.loc[fixture['hard_reschedule'] == 1, 'original_date'] = fixture[
                                    'proposed_date']
                                # Delete new columns in order to make future merges possible
                                fixture.drop(columns=['proposed_date', 'hard_reschedule'], inplace=True)

                            # Just to check, we reorder the x_var_dict indeces
                            idx = 0
                            for k in x_var_dict:
                                x_var_dict[k] = idx
                                idx += 1
                            if start_date < datetime.datetime(2021, 5, 10):
                                try:
                                    output_df = M.solve_lp(x_var_dict, diff_games_dict, prob_lp, objective)
                                except Exception as e:
                                    print("Couldn't generate schedule")
                                    break
                                fixture = pd.merge(fixture, output_df[['home', 'visitor', 'game_date',
                                                                       'proposed_date', 'model_reschedule']], how='left',
                                                   on=['home', 'visitor', 'game_date'])

                                # Update date on schedule in order to consider new dates in the feasibility calculation
                                fixture.loc[fixture['model_reschedule'] == 1, 'original_date'] = fixture['proposed_date']
                                fixture
                                # Delete new columns in order to make future merges possible
                                fixture.drop(columns=['proposed_date', 'model_reschedule'], inplace=True)

                                # We check the matches that will need a new reschedule and add it to our list
                                new_reschedules_list = M.calculate_needed_reschedules(output_df)
                                all_needed_reschedules = all_needed_reschedules + new_reschedules_list
                            check = 1
                        if check == 1:
                            fixture.to_csv(f'./output/BasicModel_{league}_{objective}_{distance_mode}_{instance}_{reschedule_mode}.csv', index=False, encoding='utf-8 sig')

