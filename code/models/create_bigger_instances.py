import pandas as pd
import numpy as np
from model_utils import League
import datetime

if __name__ == '__main__':
    for league in ['nba']:
        L = League(league)
        main_schedule = L.load_schedule()
        main_schedule['original_date'] = pd.to_datetime(main_schedule['original_date'])
        main_schedule_0 = main_schedule[main_schedule['reschedule'] == 0]
        main_schedule_1 = main_schedule[main_schedule['reschedule'] == 1]

        for mode in ['15_more_games', '25_more_games', '15_games_in_march']:
            schedule = main_schedule.copy()
            schedule['final_date'] = schedule['game_date'].copy()
            reschedules = main_schedule_0.copy()
            reschedules['Rand'] = np.random.rand(len(reschedules))

            if mode == '15_more_games':
                reschedules_select = reschedules[(reschedules['original_date'] >= datetime.datetime(2020, 12, 1)) & (
                        reschedules['original_date'] <= datetime.datetime(2021, 4, 10)
                )]
                reschedules_select = reschedules_select.sort_values(by='Rand').head(15)
                reschedules_select.loc[:, 'NewReschedule'] = 1
            elif mode == '25_more_games':
                reschedules_select = reschedules[(reschedules['original_date'] >= datetime.datetime(2020, 12, 1)) & (
                        reschedules['original_date'] <= datetime.datetime(2021, 4, 10)
                )]
                reschedules_select = reschedules_select.sort_values(by='Rand').head(25)
                reschedules_select.loc[:, 'NewReschedule'] = 1
            else:
                reschedules_select = reschedules[(reschedules['original_date'] >= datetime.datetime(2021, 3, 1)) & (
                        reschedules['original_date'] <= datetime.datetime(2021, 3, 31)
                )]
                reschedules_select = reschedules_select.sort_values(by='Rand').head(15)
                reschedules_select.loc[:, 'NewReschedule'] = 1

            schedule = pd.merge(schedule,
                                reschedules_select[['home', 'visitor', 'original_date', 'game_date', 'NewReschedule']],
                                how='left',
                                on=['home', 'visitor', 'original_date', 'game_date'])
            schedule.loc[schedule['NewReschedule'] == 1, 'game_date'] = np.max(schedule['game_date'])
            schedule.loc[schedule['NewReschedule'] == 1, 'reschedule'] = 1
            schedule.loc[schedule['NewReschedule'] == 1, 'day_difference'] = 100
            schedule.to_csv(f'./other_instances/nba_schedule_{mode}.csv', index=False, encoding='utf-8 sig')
