import pandas as pd
import os
import numpy as np
import datetime
import warnings
warnings.filterwarnings('ignore')


class League:
    def __init__(self, league, custom_schedule=pd.DataFrame()):
        """
        Initiate the League class

        Parameters
        ----------
        league: str
            String indicating the league whose schedule we want to load. Must be one of the following:
                - 'nba'
                - 'nhl'
        custom_schedule: pd.DataFrame
            Dataframe containing the custom schedule that we want to use for our model

        """
        self.league = league
        self.custom_schedule = custom_schedule

    def load_schedule(self):
        """
        Loads a dataframe that has the played and the planned schedule and another one


        Returns
        -------
        df_fixture: pd.DataFrame
            Planned and played schedule
        """
        if len(self.custom_schedule) > 0:
            df_fixture = self.custom_schedule
        else:
            # 2e load the schedule
            schedule_file_dir = os.getcwd()
            schedule_file_dir = schedule_file_dir.replace('code\\models', f'data\\schedules\\{self.league}')
            df_fixture = pd.read_csv(f'{schedule_file_dir}/{self.league}_original_and_true_schedule.csv')

        # Format date columns
        df_fixture['original_date'] = pd.to_datetime(df_fixture['original_date'])
        df_fixture['original_date'] = df_fixture['original_date'].fillna(df_fixture['game_date'])
        df_fixture['game_date'] = pd.to_datetime(df_fixture['game_date'])
        return df_fixture

    def load_rules(self):
        """
        Loads a dataframe that has the scheduling rules (how many games are played, at maximum, in a range of X days)

        Returns
        -------
        df_rules: pd.DataFrame
            Scheduling rules
        """

        # We load the schedule rules
        rules_file_dir = os.getcwd()
        rules_file_dir = rules_file_dir.replace('models', f'eda\\results')
        df_rules = pd.read_csv(f'{rules_file_dir}/{self.league}_schedule_rules.csv')

        return df_rules

    def get_disruptions(self):
        """
        Calculate for every rescheduled match, if it was a disruption (a game is a disruption if in the new date, we
        change the order of that game, compared to the old date)

        Returns
        -------
        disruption_games: list
            Array whose elements are dictionaries with information about the rescheduled games.
            The dictionary that has the following structure
            {game: (home_team, away_team),
            original_date: original datetime,
            new_date: played datetime}
        """
        # Load schedule
        df_schedule = self.load_schedule()

        teams = list(df_schedule['home'].unique())

        # Create output list
        disruption_games = []

        # The procedure will be the following:
        # - For each team, we check their games games
        # - For each rescheduled game we check the previous and next game, if it is the same it is not a disruption
        for team in teams:
            # First, we filter games of that particular team
            df_team = df_schedule[(df_schedule['home'] == team) | (df_schedule['visitor'] == team)].reset_index(drop=True)

            # We filter reschedules (only checking the ones that weren't rescheduled to a previous date)
            df_reschedules = df_team[((df_team['reschedule'] == 1) & (df_team['day_difference'] > 0))]

            for index, row in df_reschedules.iterrows():
                if row['home'] == team:
                    original_date = row['original_date']
                    new_date = row['game_date']

                    # We check the previous game
                    prev_games_old = df_team[df_team['original_date'] < original_date].sort_values(by='original_date',
                                                                                                   ascending=False).head(1)
                    prev_games_old = prev_games_old.reset_index(drop=True)
                    # We add a clause in case this is the first game of the season
                    if original_date != np.min(df_team['original_date']):
                        prev_date_old = prev_games_old['original_date'][0]
                    else:
                        prev_date_old = 'NAN'

                    # We check the previous game in the new schedule
                    prev_games_new = df_team[df_team['game_date'] < new_date].sort_values(by='game_date',
                                                                                          ascending=False).head(1)
                    prev_games_new = prev_games_new.reset_index(drop=True)
                    if new_date != np.min(df_team['game_date']):
                        prev_date_new = prev_games_new['game_date'][0]
                    else:
                        prev_date_new = 'NAN'

                    # We filter now the next game, first for the original schedule
                    next_games_old = df_team[df_team['original_date'] > original_date].sort_values(
                        by='original_date').head(1)
                    next_games_old = next_games_old.reset_index(drop=True)

                    if original_date != np.max(df_team['original_date']):
                        next_date_old = next_games_old['original_date'][0]
                    else:
                        next_date_old = 'NAN'

                    next_games_new = df_team[df_team['game_date'] > new_date].sort_values(by='game_date').head(1)
                    next_games_new = next_games_new.reset_index(drop=True)
                    if new_date != np.max(df_team['game_date']):
                        next_date_new = next_games_new['game_date'][0]
                    else:
                        next_date_new = 'NAN'

                    # We add a disruption if rivals change and dates change
                    if prev_date_old != prev_date_new and next_date_old != next_date_new:
                        disruption_games.append(
                            {'game': (team, row['visitor']),
                             'original_date': row['original_date'],
                             'game_date': row['game_date']}
                        )

        return disruption_games

    def get_available_dates(self):
        """
        We create a date range that has all dates between the first and last of the original schedule

        Returns
        -------
        league_dates: list
            Possible dates of the original schedule
        """
        df_schedule = self.load_schedule()
        league_dates = list(pd.date_range(np.min(df_schedule['original_date']), np.max(df_schedule['original_date'])))
        return league_dates

    def get_max_games_rules(self):
        """
        Creates a dictionary that saves how many games at max can be played in a span of time

        Returns
        -------
        max_games_dict: dict
            Information with maximum number of games in a span of time.
            Each item of a dictionary has the following structure:
            (home_away_condition, number_of_dates): number_of_games
        """
        # Load rules
        df_rules = self.load_rules()
        max_games_dict = {}

        for col_name in df_rules.columns:
            col_name_split = col_name.split('_')

            # As the name of the column is "Max_games_days_condition" (e.g. Max_games_1_home),
            # we use that to populate our dictionary
            if 'Max' in col_name_split:
                max_games_dict[(col_name_split[len(col_name_split) - 1],
                                int(col_name_split[len(col_name_split) - 2]))] = np.max(df_rules[col_name])

        return max_games_dict

    def get_back_to_back_rules(self):
        """
        Creates a dictionary that saves how many back to back games can be played with a particular home-away condition

        Returns
        -------
        back_to_backs_dict: dict
            Information with the maximum number of back to backs with a particular home away condition
            The dictionary has the following structure
            home_away_condition: number_of_back_to_backs
        """
        # Load rules
        df_rules = self.load_rules()
        back_to_backs_dict = {}

        for col_name in df_rules.columns:
            col_name_split = col_name.split('_')

            # As the column name is "Back2Backs_condition" (e.g. Back2Backs_home),
            # we use that to populate our dictionary
            if 'Back2Backs' in col_name_split:
                back_to_backs_dict[col_name_split[len(col_name_split) - 1]] = np.max(df_rules[col_name])
        return back_to_backs_dict

    def get_distance_matrix(self):
        """
        Generates a dictionary with the distance between two teams

        Returns
        -------
        dist_matrix: dict
            Distance matrix between teams. The dictionary has the following structure
            (team_a, team_b): distance_between_a_and_b
        """
        # As distances are saved in a dataframe, we load that first
        file_dir = os.getcwd()
        file_dir = file_dir.replace('code\\models', f'data\\teams\\{self.league}')
        dist_matrix_df = pd.read_csv(f'{file_dir}\\{self.league}_distances_matrix.csv')

        # List of teams
        teams = list(dist_matrix_df['Equipo'])

        dist_matrix = {}

        # We populate the dictionary
        for team_i in teams:
            for j in range(len(dist_matrix_df)):
                team_j = dist_matrix_df['Equipo'][j]
                dist_matrix[(team_i, team_j)] = dist_matrix_df[team_i][j]
        return dist_matrix


class Scheduler:
    def __init__(self, league, custom_fixture=None):
        """
        Initializes the Scheduler class

        Parameters
        ----------
        league: str
            String indicating the league whose schedule we want to load. Must be one of the following:
                - 'nba'
                - 'nhl'
        custom_fixture (optional): pd.DataFrame
            If specified, we use a custom fixture for the schedule. This will be useful when we are building
            models iteratively
        """
        self.league = league
        L = League(league, custom_schedule=custom_fixture)
        self.df_fixture = custom_fixture
        """
        # We import things from the League class
        try:
            if custom_fixture.shape[0] > 0:
                L = League(league, custom_schedule=custom_fixture)
                self.df_fixture = custom_fixture
            else:
                L = League(league)
                self.df_fixture = L.load_schedule()
        except:
            L = League(league)
            self.df_fixture = L.load_schedule()
        """
        self.disruptions = L.get_disruptions()
        self.league_dates = L.get_available_dates()
        self.max_games_rules = L.get_max_games_rules()
        self.back_to_back_rules = L.get_back_to_back_rules()
        self.dist_matrix = L.get_distance_matrix()
        self.teams = list(self.df_fixture['home'].unique())

    def obtain_available_dates_by_team(self):
        """
        Evaluate by team if we can play a game on a particular date, evaluating if
            - there's a game on that particular day
            - if we can play according to the rules that we have analyzed

        Returns
        -------
        available_games_dict: dict
            Available dates to play by team. The dictionary has the following structure
            team: [list_of_dates]
        """
        # We create a list that indicates the matches where there aren't any games
        if self.league == 'nba':
            dates_without_matches = [datetime.datetime(2020, 12, 24), datetime.datetime(2021, 3, 5),
                                     datetime.datetime(2021, 3, 6), datetime.datetime(2021, 3, 7),
                                     datetime.datetime(2021, 3, 8), datetime.datetime(2021, 3, 9)]
        else:
            dates_without_matches = [datetime.datetime(2021, 5, 9), datetime.datetime(2021, 5, 17)]

        available_games_dict = {}
        for team in self.teams:
            available_games_dict[team] = []
            # Filter games of a particular team
            home_games = self.df_fixture[self.df_fixture['home'] == team]
            away_games = self.df_fixture[self.df_fixture['visitor'] == team]
            all_games = self.df_fixture[((self.df_fixture['home'] == team) | (self.df_fixture['visitor'] == team))]

            for day in self.league_dates:
                # Variable that we will create to see if it is an available date
                available = 1

                # First, we check if there is a game on this date
                if day.date() in list(all_games['original_date']):
                    available = 0
                elif day in dates_without_matches or day.date() in dates_without_matches:
                    available = 0
                else:
                    # Now, we check if in the previous n days we already played the maximum number of games
                    for n in range(2, 5):
                        # We do a rolling window, checking games between x days
                        start = day - datetime.timedelta(days=n - 1)
                        end = day
                        # We select games in the window until we are done passing through our day
                        while start <= day:
                            df_aux_home = home_games[((home_games['original_date'] >= start) & (
                                    home_games['original_date'] <= end))]
                            df_aux_away = away_games[((away_games['original_date'] >= start) & (
                                    away_games['original_date'] <= end))]
                            df_aux_all = all_games[((all_games['original_date'] >= start) & (
                                    all_games['original_date'] <= end))]

                            # If the amount of games is equal to the maximum, then we can't put a game here
                            if len(df_aux_home) >= self.max_games_rules[('home', n)]:
                                available = 0
                            if len(df_aux_away) >= self.max_games_rules[('away', n)]:
                                available = 0
                            if len(df_aux_all) >= self.max_games_rules[('all', n)]:
                                available = 0
                            if day == datetime.datetime(2021, 1, 5):
                                rule_home = self.max_games_rules[('home', n)]
                                rule_away = self.max_games_rules[('away', n)]
                                rule_all = self.max_games_rules[('all', n)]

                            # We update start and end
                            start = start + datetime.timedelta(days=1)
                            end = end + datetime.timedelta(days=1)

                if available == 1:
                    available_games_dict[team].append(day)
        availables = []
        for team in available_games_dict:
            availables.append(len(available_games_dict[team]))
        return available_games_dict

    def get_tours_by_team(self):
        """
        Creates a dictionary that has, by team, a list of lists that has the dates of each tour
        A tour is a series of matches with the same home/away condition that have three days or less between games

        Returns
        -------
        tours_dict: dict
            Tours by team. The dictionary has the following structure
            team: [tour_1, tour_2, tour_3]
        """
        tours_dict = {}
        for team in self.teams:
            tours_dict[team] = []

            # We filter the games of this team
            team_games = self.df_fixture[((self.df_fixture['home'] == team) | (self.df_fixture['visitor'] == team))]
            team_games = team_games.sort_values(by='original_date').reset_index(drop=True)

            # We create a column that has the previous game date
            team_games['prev_date'] = team_games['original_date'].shift(1)
            team_games['diff'] = (team_games['original_date'] - team_games['prev_date']).dt.days
            team_games['diff'] = team_games['diff'].fillna(0)
            team_games = team_games.reset_index(drop=True)

            # Check the condition of the first game
            if team_games['home'][0] == team:
                prev_condition = 'H'
            else:
                prev_condition = 'A'

            tour_games = [team_games['original_date'][0]]

            for i in range(1, len(team_games)):
                # Check the condition of the game
                if team_games['home'][i] == team:
                    condition = 'H'
                else:
                    condition = 'A'
                # If it's the same condition, we add a new game to the tour
                if condition == prev_condition and team_games['diff'][i] < 4:
                    tour_games.append(team_games['original_date'][i])
                else:
                    # If not, we finish the tour and create a new one
                    if len(tour_games) > 1:
                        tours_dict[team].append(tour_games)
                    tour_games = [team_games['original_date'][i]]
                    prev_condition = condition
        return tours_dict

    def get_windows_by_team(self, available_games_dict):
        """
        Calculates the potential window of days in which a team has available days for games
        A window is defined by a set of consecutive games in which a team can have new days

        Parameters
        ----------
        available_games_dict: dict
            Available dates to play by team. The dictionary has the following structure
            team: [list_of_dates]

        Returns
        -------
        windows_dict: dict
            Windows by team. The dictionary structure is the following:
            team: [window_1, window_2, window_3]
        """
        windows_dict = {}
        for team in available_games_dict:
            windows_dict[team] = []
            available_dates = available_games_dict[team]

            # Initiate first window
            prev_date = available_dates[0]
            window = [available_dates[0]]

            # For each available date
            for i in range(1, len(available_dates)):
                now_date = available_dates[i]

                # We calculate the difference
                diff = abs((now_date - prev_date).days)

                # If difference is equal to 1, we add it to the window. If not,
                # we create a new window
                if diff == 1:
                    window.append(now_date)
                else:
                    if len(window) > 1:
                        windows_dict[team].append(window)
                    window = [now_date]
                prev_date = now_date

        return windows_dict

    def calculate_resched_windows(self):
        """
        Additionally to the hard rules a schedule has, we calculate sets of dates in which games will not be played due
        to reschedules. Therefore, if a rescheduled game is scheduled in a window of days where there was a COVID
        outbreak, a new reschedule should occur

        Returns
        -------
        resched_windows_dict: dict
            Information by team of the windows in which there were reschedules
        """
        resched_windows_dict = {}
        for team in self.teams:
            resched_windows_dict[team] = []

            team_games = self.df_fixture[((self.df_fixture['home'] == team) | (self.df_fixture['visitor'] == team))]

            # We filter reschedules (only checking the ones that weren't rescheduled to a previous date)
            df_reschedules = team_games[((team_games['reschedule'] == 1) & (team_games['day_difference'] > 0))]

            for index, row in df_reschedules.iterrows():
                new_date = row['original_date']

                # We check the previous game of the reschedule
                prev_game = team_games[(team_games['game_date'] < new_date) & (
                        team_games['reschedule'] == 0)].sort_values(by='game_date', ascending=False).head(1)
                prev_game = prev_game.reset_index(drop=True)

                # We check the next game of the reschedule
                next_game = team_games[(team_games['game_date'] > new_date) & (
                        team_games['reschedule'] == 0)].sort_values(by='game_date').head(1)
                next_game = next_game.reset_index(drop=True)

                # Create the date range between both dates and append it
                if len(prev_game) > 0 and len(next_game) > 0:
                    window = list(pd.date_range(prev_game['game_date'][0], next_game['game_date'][0]))
                elif len(prev_game) > 0 and len(next_game) == 0:
                    window = list(pd.date_range(prev_game['game_date'][0], np.max(self.df_fixture['original_date'])))
                elif len(prev_game) == 0 and len(next_game) > 0:
                    window = list(pd.date_range(np.min(self.df_fixture['original_date']), next_game['game_date'][0]))
                else:
                    window = []

                if window not in resched_windows_dict[team]:
                    resched_windows_dict[team].append(window)

        return resched_windows_dict

