import pandas as pd
import os

if __name__ == '__main__':

    file_dir = os.getcwd()
    file_dir = file_dir.replace('code\\process_data', f'data')

    # Load dataframe and format date column
    df = pd.read_csv(f'{file_dir}\\schedules\\nhl\\2021_processed_schedule.csv')
    teams = list(df['home'].unique())
    teams.sort()

    # Distance matrix
    df_dist = pd.read_csv(f'{file_dir}\\teams\\nhl\\nhl_distances_matrix.csv')
    dist_teams = list(df_dist['Equipo'])
    dist_teams.sort()

    teams_not = [x for x in teams if x not in dist_teams]
    print(teams)
    print(dist_teams)

    print(teams_not)




