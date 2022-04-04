import pandas as pd
from model_utils import Scheduler
import datetime
import pickle
import numpy as np


class BasicModel:
    def __init__(self, league, custom_fixture=None, start_date=datetime.datetime(2021, 1, 1), distance_mode='mid'):
        """
        Initiate basic model class

        Parameters
        ----------
        league: str
            String indicating the league whose schedule we want to load. Must be one of the following:
                - 'nba'
                - 'nhl'
        custom_fixture (optional): pd.DataFrame
            If specified, we use a custom fixture for the schedule. This will be useful when we are building
            models iteratively
        start_date: datetime.datetime
            Start datetime for the models that will be rescheduled
        distance_mode: str
            Indicates artificial modes that will be used to differentiate the distance tolerance.
            Must be one of the following:
                - low
                - mid
                - high
        """
        self.league = league

        # Create other classes
        S = Scheduler(league, custom_fixture=custom_fixture)
        self.df_fixture = S.df_fixture
        self.disruptions = S.disruptions
        self.league_dates = S.league_dates
        self.max_games_rules = S.max_games_rules
        self.back_to_back_rules = S.back_to_back_rules
        self.dist_matrix = S.dist_matrix
        self.teams = list(self.df_fixture['home'].unique())
        self.available_games_dict = S.obtain_available_dates_by_team()
        with open('av_dict.pickle', 'wb') as handle:
            pickle.dump(self.available_games_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        self.windows_dict = S.get_windows_by_team(self.available_games_dict)
        self.tours_dict = S.get_tours_by_team()
        self.resched_windows_dict = S.calculate_resched_windows()
        self.start_date = start_date
        self.distance_mode = distance_mode

    def check_distance_feasibility(self, games_to_chack, margin=0.2):
        """
        For each disruption and each possible day for each team, we see if it is a desirable day to put the match in.
        Basically, we calculate the distance each team would incur if we put the rescheduled game there and see if
        that's acceptable (the distance would be acceptable if it is lower than the original distance multiplied by
        1 + margin

        Parameters
        ----------
        games_to_chack: list
            Games that should be considered to evaluate its distance feasibility
        margin: float
            Maximum acceptable percentage level of difference between the original distance traveled and the new one in
            the new model

        Returns
        -------
        match_distance_feasibility: dict
            Has information per match and team of the days in which it is reasonable to have a match.
            The dictionary has the following structure
            match: (team_1: [list_of_feasible_days], team_2: [list_of_feasible_days]
        """
        match_distance_feasibility = {}
        # For every disruption game
        for match in games_to_chack:
            home_team = match['game'][0]
            away_team = match['game'][1]

            # Create a team dictionary of stats
            team_stats = {'home': {'team': home_team}, 'away': {'team': away_team}}

            # First, we calculate the distance traveled by each team. The distance will be equal to
            # Distance between home team of the previous game and home team of this game +
            # Distance between home team of this game and the home team of the next game
            for team in team_stats:
                team_games = self.df_fixture[((self.df_fixture['home'] == team_stats[team]['team']) | (
                        self.df_fixture['visitor'] == team_stats[team]['team']))]

                # We see check the previous and the next game
                prev_game = team_games[team_games['original_date'] < match['original_date']].sort_values(
                    by='original_date', ascending=False).head(1).reset_index(drop=True)
                next_game = team_games[team_games['original_date'] > match['original_date']].sort_values(
                    by='original_date').head(1).reset_index(drop=True)
                if len(prev_game) > 0:
                    prev_home = prev_game['home'][0]
                else:
                    prev_home = team_stats[team]['team']

                if len(next_game) > 0:
                    next_home = next_game['home'][0]
                else:
                    next_home = team_stats[team]['team']
                distance = self.dist_matrix[(prev_home, home_team)] + self.dist_matrix[(home_team, next_home)]
                team_stats[team]['distance'] = distance

                # In order to avoid restricting too much the space when we have to reschedule a home game, we calculate
                # the closest distance between this team and another
                closest_distance = 1e10
                for team_pair in self.dist_matrix:
                    if team_stats[team]['team'] in team_pair and self.dist_matrix[team_pair] > 0:
                        if self.dist_matrix[team_pair] < closest_distance:
                            closest_distance = self.dist_matrix[team_pair]

                # Create a list where we will add feasible days
                possible_days = []

                # For each potential day, we calculate the distance that we would have
                for potential_day in self.available_games_dict[team_stats[team]['team']]:

                    if potential_day > match['original_date']:

                        # Check potential previous and next game
                        pot_prev_game = team_games[team_games['original_date'] < potential_day].sort_values(
                            by='original_date', ascending=False).head(1).reset_index(drop=True)
                        pot_next_game = team_games[team_games['original_date'] > potential_day].sort_values(
                            by='original_date').head(1).reset_index(drop=True)
                        if len(pot_prev_game) > 0 and len(pot_next_game) > 0:

                            # Calculate distance in the same way
                            pot_prev_home = pot_prev_game['home'][0]
                            pot_next_home = pot_next_game['home'][0]
                            pot_distance = self.dist_matrix[(pot_prev_home, home_team)] + \
                                           self.dist_matrix[(home_team, pot_next_home)]
                            pot_distance_1 = np.min([self.dist_matrix[(pot_prev_home, home_team)],
                                                     self.dist_matrix[(home_team, pot_next_home)]])
                            pot_distance_2 = np.max([self.dist_matrix[(pot_prev_home, home_team)],
                                                     self.dist_matrix[(home_team, pot_next_home)]])

                            # If distance is reasonable, we add this to our list of potential dayss
                            if distance == 0:
                                reference = closest_distance
                            else:
                                reference = distance
                            if pot_distance <= reference * (1 + margin) or abs(pot_distance_2/pot_distance_1 - 1) <= margin:
                                possible_days.append(potential_day)

                if margin < 2500:
                    if len(possible_days) > 7:
                        match_distance_feasibility[(team_stats[team]['team'], match['original_date'],
                                                    match['game_date'])] = possible_days
                    else:
                        match_distance_feasibility[(team_stats[team]['team'], match['original_date'],
                                                    match['game_date'])] = []
                else:
                    match_distance_feasibility[(team_stats[team]['team'], match['original_date'],
                                                match['game_date'])] = possible_days
        return match_distance_feasibility

    def calculate_existing_back_to_backs(self):
        """
        Calculates the current number of back-to-backs, without considering the disruptions

        Returns
        -------
        b2b_dict: dict
            Dictionary containing information per team of the number of back to backs
            The dictionary has the following structure
            team: {'home': number_of_back_to_backs, 'away': n, 'all': m}
        """
        b2b_dict = {}

        for team in self.teams:
            team_disruptions = []

            # We calculate the disruptions of each team
            for dis in self.disruptions:
                if team == dis['game'][0] or team == dis['game'][1]:
                    team_disruptions.append(dis['original_date'])

            # We filter the team's matches and filter the matches we don't want
            team_games = self.df_fixture[((self.df_fixture['home'] == team) | (
                    self.df_fixture['visitor'] == team))]
            team_games = team_games[~team_games['original_date'].isin(team_disruptions)].sort_values(by='original_date')

            home_games = team_games[team_games['home'] == team]
            away_games = team_games[team_games['visitor'] == team]

            b2b_all = 0
            b2b_home = 0
            b2b_away = 0

            # For each day in the tournament
            for i in range(len(self.league_dates) - 1):
                # We check the range that we want to consider
                start = self.league_dates[i]
                end = self.league_dates[i + 1]

                # Filter dates
                df_filt_all = team_games[(team_games['original_date'] >= start) & (team_games['original_date'] <= end)]
                df_filt_home = home_games[(home_games['original_date'] >= start) & (home_games['original_date'] <= end)]
                df_filt_away = away_games[(away_games['original_date'] >= start) & (away_games['original_date'] <= end)]

                # Calculate number of games and check if it is the max
                n_games_all = df_filt_all.shape[0]
                n_games_home = df_filt_home.shape[0]
                n_games_away = df_filt_away.shape[0]

                if n_games_all == 2:
                    b2b_all += 1

                if n_games_home == 2:
                    b2b_home += 1

                if n_games_away == 2:
                    b2b_away += 1

            b2b_dict[team] = {
                'home': b2b_home,
                'away': b2b_away,
                'all': b2b_all
            }
        return b2b_dict

    def add_variables_dict_according_to_distance_threshold(self, matches_to_schedule, match_distance_feasibility,
                                                           idx, x_var_dict, x_var_dict_inv, end_date):
        """
        Creation of variables according to the selected dictionary of distance feasibility

        Parameters
        ----------
        matches_to_schedule: list
            Matches that are going to be rescheduled
        match_distance_feasibility: dict
            Has information per match and team of the days in which it is reasonable to have a match.
            The dictionary has the following structure
            match: (team_1: [list_of_feasible_days], team_2: [list_of_feasible_days]
        idx: int
            Maximum index of the variables that were created
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        x_var_dict_inv: dict
            Inverse dictionary of decision variables that will be included in the model. Each item in the
            dictionary will have the following structure
            index: (home_team, away_team, original_date, game_date, proposed_date)
        end_date: datetime.datetime
            End date of the window of games that we want to reschedule

        Returns
        -------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        x_var_dict_inv: dict
            Inverse dictionary of decision variables that will be included in the model. Each item in the
            dictionary will have the following structure
            index: (home_team, away_team, original_date, game_date, proposed_date)
        non_matched_matches: list
            Matches that under the current distance feasibility restrictions weren't assigned any potential date
        idx: int
            Maximum index of the variables that were created

        """
        # We create a list of matches that weren't matched to any date
        non_matched_macthes = []

        # For every match we will do the following:
        # Check every date and see if
        #   The date is in both team available's date
        #   The date is greater than end_date
        for match in matches_to_schedule:
            home_team = match['game'][0]
            away_team = match['game'][1]

            # Check which date we will evaluate
            date_to_check = 'original_date'

            # We check the conditions
            check = 0
            for pot_date in self.league_dates:
                if pot_date > end_date and \
                        pot_date in match_distance_feasibility[(home_team,
                                                                match[date_to_check], match['game_date'])] and \
                        pot_date in match_distance_feasibility[(away_team, match[date_to_check], match['game_date'])]:

                    # If all conditions apply, we add the match to the variables dict
                    x_var_dict[(home_team, away_team, match[date_to_check], match['game_date'], pot_date)] = idx
                    x_var_dict_inv[idx] = (home_team, away_team, match[date_to_check], match['game_date'], pot_date)
                    idx += 1
                    check = 1
            if check == 0:
                non_matched_macthes.append(match)

        return x_var_dict, x_var_dict_inv, non_matched_macthes, idx

    def create_decision_variables_dict(self, start_date, end_date, objective, match_buffer=None):
        """
        Creates a dictionary in which we save an index for each decision variable that we will include.
        As we will run this model monthly, we will try to imitate the real thing:
            - We will create variables for every suspended match between start_date and end_date

        This matches will include the ones whose original date was between start_date and end_date and any rescheduled
        match who was scheduled between a new COVID outbreak. In this sense, it would be as we are in end_date + 1

        Parameters
        ----------
        start_date: datetime.datetime
            Start date of the window of games that we want to reschedule
        end_date: datetime.datetime
            End date of the window of games that we want to reschedule
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'basic': Will try to maximize the number of scheduled games
                - 'min_diff': Will try to minimize the difference between the original date and the new date
                - 'balanced': We try to make the daily match assignment balanced: trying to minimize the number of games
                              is played by the team that has played the most games
        match_buffer: list
            List of games that we want to reschedule again because the reschedule date was during a new COVID outbreak

        Returns
        -------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index.
        matches_to_be_scheduled: list
            Matches that are rescheduled
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        non_matched_macthes5: list
            List with information of matches that we couldn't find any potential days
        """
        # First we will get the matches to be scheduled, checking if they are in the match buffer and
        # if their date was between the dates
        if match_buffer:
            matches_to_be_scheduled = match_buffer
        else:
            matches_to_be_scheduled = []
        for res in self.disruptions:
            if start_date <= res['original_date'] <= end_date:
                matches_to_be_scheduled.append(res)

        # We calculate the available dates per team, with different level of preference
        if self.distance_mode == 'low':
            match_distance_feasibility_1 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0)
            match_distance_feasibility_2 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.2)
            match_distance_feasibility_3 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.5)
        elif self.distance_mode == 'mid':
            match_distance_feasibility_1 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.2)
            match_distance_feasibility_2 = self.check_distance_feasibility(matches_to_be_scheduled, margin=0.7)
            match_distance_feasibility_3 = self.check_distance_feasibility(matches_to_be_scheduled, margin=1)
        else:
            match_distance_feasibility_1 = self.check_distance_feasibility(matches_to_be_scheduled, margin=1500)
            match_distance_feasibility_2 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2000)
            match_distance_feasibility_3 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2200)

        match_distance_feasibility_4 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2400)
        match_distance_feasibility_5 = self.check_distance_feasibility(matches_to_be_scheduled, margin=2500)

        # For every match we will do the following:
        # Check every date and see if
        #   The date is in both team available's date
        #   The date is greater than end_date
        x_var_dict = {}
        x_var_dict_inv = {}
        idx = 0

        # We populate our dictionaries according to the information that we have
        x_var_dict, x_var_dict_inv, non_matched_macthes1, idx = self.add_variables_dict_according_to_distance_threshold(
            matches_to_be_scheduled, match_distance_feasibility_1, idx, x_var_dict, x_var_dict_inv, end_date
        )
        x_var_dict, x_var_dict_inv, non_matched_macthes2, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes1, match_distance_feasibility_2, idx, x_var_dict, x_var_dict_inv, end_date
        )
        x_var_dict, x_var_dict_inv, non_matched_macthes3, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes2, match_distance_feasibility_3, idx, x_var_dict, x_var_dict_inv, end_date
        )
        x_var_dict, x_var_dict_inv, non_matched_macthes4, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes3, match_distance_feasibility_4, idx, x_var_dict, x_var_dict_inv, end_date
        )
        x_var_dict, x_var_dict_inv, non_matched_macthes5, idx = self.add_variables_dict_according_to_distance_threshold(
            non_matched_macthes4, match_distance_feasibility_5, idx, x_var_dict, x_var_dict_inv, end_date
        )

        # We add the indeces if we use the objective function in which we are trying to balance the number of games
        # per team
        diff_games_dict = {}
        if objective == 'balanced':
            for day in self.league_dates:
                diff_games_dict[day] = idx
                idx += 1

        return x_var_dict, matches_to_be_scheduled, diff_games_dict, non_matched_macthes5

    def add_schedule_rules_constraints(self, x_var_dict, prob_lp, home_away_status, n_days):
        """
        Adds a set of constraint that limits the number of games in a particular set of days. For example, for each set
        of consecutive days, we can't have more than two games. A constraint will be created per team, days and number
        of days. For example this constraint

        sum_{i} x_it + sum_{i} x_it+1 <= 2 \foreach t, i \in GamesOfTeamA

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        home_away_status: str
            Indicates the home/away status of the constraint. Must be one of the following:
                - 'home'
                - 'away'
                - 'all'
        n_days: int
            The number of days that will be considered for a particular constraint

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for team in self.teams:
            # We filter the games of this particular team
            if home_away_status == 'home':
                filt_games = self.df_fixture[self.df_fixture['home'] == team]
            elif home_away_status == 'away':
                filt_games = self.df_fixture[self.df_fixture['visitor'] == team]
            else:
                filt_games = self.df_fixture[((self.df_fixture['home'] == team) | (
                        self.df_fixture['visitor'] == team))]

            for i in range(len(self.league_dates) - n_days + 1):
                initial_day = self.league_dates[i]

                # We calculate the number of games that are already played on this window in order to substract them
                # from the right hand side. For example, if only two matches can be played in a span of three days and
                # already there is a fixed game, then from our options, we can only add one additional game, not two
                start = initial_day
                end = self.league_dates[i + n_days - 1]

                filt_days = filt_games[(filt_games['original_date'] >= start) & (filt_games['original_date'] <= end)]
                n_games = len(filt_days)

                # We create the list of variables and values that will be used for our constraint
                ind = []
                val = []
                # We check for any potential reschedule if this team is in that reschedule
                for var in x_var_dict:
                    if team == var[0] or team == var[1]:
                        # Additionally, we check if this particular date is on that reschedule
                        if initial_day == var[4]:
                            ind.append(x_var_dict[var])
                            val.append(1)

                # Now, we do the same for the next days
                for n in range(1, n_days):
                    new_day = self.league_dates[i + n]

                    # We check for any potential reschedule if this team is in that reschedule
                    for var in x_var_dict:
                        if team == var[0] or team == var[1]:
                            # Additionally, we check if this particular date is on that reschedule
                            if new_day == var[4]:
                                ind.append(x_var_dict[var])
                                val.append(1)

                # We check if we have variables in order to add our constraint
                if len(ind) > 0:
                    row = [ind, val]
                    if n_days == 7:
                        row

                    # We add the constraint, checking the number of played games and the maximum allowed
                    max_games_rule = self.max_games_rules[(home_away_status, n_days)]
                    if self.start_date == datetime.datetime(2021, 1, 1):
                        max_games_rule
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'],
                                                   rhs=[self.max_games_rules[(home_away_status, n_days)] - n_games])

        return prob_lp

    def one_game_per_day(self, x_var_dict, prob_lp):
        """
        Limits the number of games per day and team to a maximum of 1. This will be represented in the following way

        sum_{i} x_it <= 1 \foreach t, i \in GamesOfTeamA

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for team in self.teams:
            for i in range(len(self.league_dates)):
                initial_day = self.league_dates[i]

                # We check by day, if there are any potential reschedules on that particular date

                # We create the list of variables and values that will be used for our constraint
                ind = []
                val = []

                # We check for any potential reschedule if this team is in that reschedule
                for var in x_var_dict:
                    if team == var[0] or team == var[1]:
                        # Additionally, we check if this particular date is on that reschedule
                        if initial_day == var[4]:
                            ind.append(x_var_dict[var])
                            val.append(1)

                # We check if we have variables in order to add our constraint
                if len(ind) > 0:
                    row = [ind, val]

                    # We add the constraint, checking the number of played games and the maximum allowed
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[1])

        return prob_lp

    def how_many_times_each_games_is_played(self, x_var_dict, prob_lp, objective):
        """
        This constraint limits the number of times a single match is rescheduled to one time tops
        (if objective = 'basic') or equal to one if it is equal to 'min_diff'.
        As an example, we see this mathematically, this can be expressed in the following way (objective = 'basic'):

        sum_{t} x_it <= 1  \forall i

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'basic': Will try to maximize the number of scheduled games
                - 'min_diff': Will try to minimize the difference between the original date and the new date
                - 'balanced': We try to make the daily match assignment balanced: trying to minimize the number of games
                              is played by the team that has played the most games

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for dis in self.disruptions:
            ind = []
            val = []
            for var in x_var_dict:
                if dis['game'][0] == var[0] and dis['game'][1] == var[1] and dis['original_date'] == var[2] and \
                        dis['game_date'] == var[3]:
                    ind.append(x_var_dict[var])
                    val.append(1)
            if len(ind) > 0:
                row = [ind, val]
                prob_lp.linear_constraints.add(lin_expr=[row], senses=['E'], rhs=[1])

        return prob_lp

    def max_one_game_per_window(self, x_var_dict, prob_lp):
        """
        Creation of constraints that limit the number of games per window to 1. A window is defined as a set of
        consecutive available days. Mathematically this can be displayed in the following way if we think that a team
        has three consecutive available days with matches

        sum_{i} x_it + x_it+1 + x_it+2 <= 1 \foreach Window, i \in GamesOfTeamA

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # We create a constraint for each team and window
        for team in self.teams:
            for window in self.windows_dict[team]:
                ind = []
                val = []
                for var in x_var_dict:
                    # For each possible match we check if correct the team and date are in the variable
                    if team == var[0] or team == var[1]:
                        if var[4] in window:
                            ind.append(x_var_dict[var])
                            val.append(1)
                if len(ind) > 0:
                    row = [ind, val]
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[2])
        return prob_lp

    def limit_on_tour_games(self, x_var_dict, prob_lp):
        """
        Limits the number of games that can be scheduled into a tour to the maximum of games a team tour has.
        A tour is a series of matches with the same home/away condition that have three days or less between games.
        Mathematically, this can be expressed in the following way:

        sum_{t \\in tour} x_it <= Cmax - len(tour) \\forall tour

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        max_len = 0
        for team in self.teams:
            team_tours = self.tours_dict[team]
            for tour in team_tours:
                if len(tour) > max_len:
                    max_len = len(tour)
        for team in self.teams:
            for tour in self.tours_dict[team]:
                ind = []
                val = []

                # We now calculate the potential matches that could be scheduled during this tour
                for var in x_var_dict:
                    # If this is a tour of this team
                    if team == var[0] or team == var[1]:
                        # We check if the potential date is between the first and last date of the tour
                        if tour[0] < var[4] < tour[len(tour) - 1]:
                            ind.append(x_var_dict[var])
                            val.append(1)
                if len(ind) > 0:
                    row = [ind, val]
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'], rhs=[max_len - len(tour)])
        return prob_lp

    def limit_on_back_to_backs(self, x_var_dict, prob_lp, home_away_status):
        """
        We limit the number of back-to-backs that the final schedule would have. If we saved matches of team A that
        would generate a back-to-back in set B, this could be represented in the following way

        sum_{i \\in A} sum_{t} x_it <= B2BMax \\forall team

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        home_away_status: str
            Indicates the home/away status of the constraint. Must be one of the following:
                - 'home'
                - 'away'
                - 'all'

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for team in self.teams:
            # We filter the games of this particular team
            if home_away_status == 'home':
                filt_games = self.df_fixture[self.df_fixture['home'] == team]
            elif home_away_status == 'away':
                filt_games = self.df_fixture[self.df_fixture['visitor'] == team]
            else:
                filt_games = self.df_fixture[((self.df_fixture['home'] == team) | (
                        self.df_fixture['visitor'] == team))]

            ind = []
            val = []

            for var in x_var_dict:
                # We check if this variable includes this team
                if team == var[0] or team == var[1]:
                    proposed_date = var[4]
                    prev_date = proposed_date - datetime.timedelta(days=1)
                    next_date = proposed_date + datetime.timedelta(days=1)

                    # And also we check if there is a game in the previous/next day
                    if prev_date in filt_games['original_date'] or prev_date.date() in filt_games['original_date'] or \
                            next_date in filt_games['original_date'] or next_date.date() in filt_games['original_date']:
                        ind.append(x_var_dict[var])
                        val.append(1)

            if len(ind) > 0:
                row = [ind, val]
                prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'],
                                               rhs=[self.back_to_back_rules[home_away_status]])

        return prob_lp

    def add_balanced_objective_function_constraint(self, x_var_dict, prob_lp, diff_games_dict):
        """
        We create a constraint that relates the objective function that balances the number of games played by each
        team. If set A has the set of games of team te

        This constraint will be equal to
        Dt <= sum{t<t} xbar_it + x_it \for each te \in teams, te \in teams

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        prob_lp: cplex.Cplex
            Cplex problem
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        for day in self.league_dates:
            for team in self.teams:
                # Create lists of indices and values
                ind = []
                val = []
                team_games = self.df_fixture[((self.df_fixture['home'] == team) | (
                        self.df_fixture['visitor'] == team))]
                # I don't want to consider the games that we will reschedule, so we take them out

                # To see this, we check the actual game date of each match (as we are over-writing the proposed_date on
                # the dataframe)
                reschedule_dates = []
                for var in x_var_dict:
                    if var[3] in list(team_games['game_date']):
                        reschedule_dates.append(var[3].date())
                team_games_reschedules = team_games[team_games['game_date'].isin(reschedule_dates)]
                team_games_not_reschedules = team_games[~team_games['game_date'].isin(reschedule_dates)]

                # We count the games that have been played until today
                games_played = team_games_not_reschedules[team_games_not_reschedules['original_date'] <= day]
                n_games_played = len(games_played)

                # We add the matches that we can reschedule that are prior to the date that we are looking
                for var in x_var_dict:
                    # We consider this match if the propsed date is prior or equal to the date we are looking
                    if var[4] <= day.date():
                        if var[0] == team or var[1] == team:
                            ind.append(x_var_dict[var])
                            val.append(1)

                # Additionally, we add the variable corresponding to the day we are looking
                ind.append(diff_games_dict[day])
                val.append(-1)

                # If we have "x" variables, we add the constraint
                if len(ind) > 1:
                    row = [ind, val]
                    prob_lp.linear_constraints.add(lin_expr=[row], senses=['L'],
                                                   rhs=[-n_games_played])

        return prob_lp

    def add_constraint_matrix(self, x_var_dict, diff_games_dict, prob_lp, objective):
        """
        Adds constraint matrix to the problem, calling all the different methods

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, game_date, proposed_date): index
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        prob_lp: cplex.Cplex
            Cplex problem
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'basic': Will try to maximize the number of scheduled games
                - 'min_diff': Will try to minimize the difference between the original date and the new date
                - 'balanced': We try to make the daily match assignment balanced: trying to minimize the number of games
                              is played by the team that has played the most games

        Returns
        -------
        prob_lp: cplex.Cplex
            Cplex problem
        """
        # Add every created constraint
        for home_away_status in ['home', 'away', 'all']:
            prob_lp = self.limit_on_back_to_backs(x_var_dict, prob_lp, home_away_status)
            for n_days in list(range(2, 4)):
                prob_lp = self.add_schedule_rules_constraints(x_var_dict, prob_lp, home_away_status, n_days)

        prob_lp = self.one_game_per_day(x_var_dict, prob_lp)
        prob_lp = self.how_many_times_each_games_is_played(x_var_dict, prob_lp, objective)
        prob_lp = self.max_one_game_per_window(x_var_dict, prob_lp)
        prob_lp = self.limit_on_tour_games(x_var_dict, prob_lp)
        if objective == 'balanced':
            prob_lp = self.add_balanced_objective_function_constraint(x_var_dict, prob_lp, diff_games_dict)
        return prob_lp

    def populate_by_row(self, x_var_dict, diff_games_dict, prob_lp, objective):
        """
        Function that generates the model, creating the objective function and the needed constraints

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        prob_lp: cplex.Cplex
            Cplex problem
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'basic': Will try to maximize the number of scheduled games
                - 'min_diff': Will try to minimize the difference between the original date and the new date
                - 'balanced': We try to make the daily match assignment balanced: trying to minimize the number of games
                              is played by the team that has played the most games

        Returns
        -------

        """
        if objective == 'basic':
            # Here the objective funcion will be equal to
            # max sum_{x_it}.
            # Considering that x_it is a binary variable that indicates if match i (defined by home team, away team,
            # original date and played date) is rescheduled for day t, lower bound for this variable will be 0 and
            # upper bounds will be equal to 1
            coef = []
            lower_bounds = []
            upper_bounds = []
            types = []
            names = []
            for var in x_var_dict:
                coef.append(1)
                lower_bounds.append(0)
                upper_bounds.append(1)
                types.append('B')
                names.append(f'{var[0]}_{var[1]}_{var[2]}_{var[3]}_{var[4]}')

            # Add the variables to the problem and set the problem sense
            prob_lp.variables.add(obj=coef, lb=lower_bounds, ub=upper_bounds, types=types, names=names)
            prob_lp.objective.set_sense(prob_lp.objective.sense.maximize)

        elif objective == 'min_diff':
            # Here the objective funcion will be equal to
            # min sum_{diff_it*x_it}.
            # The only thing that changes here is the objective function coefficient diff_it, that is equal to the
            # difference between the potential date and the original date
            coef = []
            lower_bounds = []
            upper_bounds = []
            types = []
            names = []
            for var in x_var_dict:
                # As the key of each item of x_var_dict is equal to a tuple with the following information
                # (home_team, away_team, original_date, played_date, proposed_date), this coefficient will be equal to
                # proposed_date - original_date, i.e. the fifth element of the tuple minus the third
                coef.append(1/abs((var[4]-var[2]).days))
                lower_bounds.append(0)
                upper_bounds.append(1)
                types.append('B')
                names.append(f'{var[0]}_{var[1]}_{var[2]}_{var[3]}_{var[4]}')

            # Add the variables to the problem and set the problem sense
            prob_lp.variables.add(obj=coef, lb=lower_bounds, ub=upper_bounds, types=types, names=names)
            prob_lp.objective.set_sense(prob_lp.objective.sense.maximize)

        elif objective == 'balanced':
            # Here the objective will be equal to
            # min sum{Dt}
            coef = []
            lower_bounds = []
            upper_bounds = []
            types = []
            names = []
            for var in x_var_dict:
                # As the key of each item of x_var_dict is equal to a tuple with the following information
                # (home_team, away_team, original_date, played_date, proposed_date), this coefficient will be equal to
                # proposed_date - original_date, i.e. the fifth element of the tuple minus the third
                coef.append(0)
                lower_bounds.append(0)
                upper_bounds.append(1)
                types.append('B')
                names.append(f'{var[0]}_{var[1]}_{var[2]}_{var[3]}_{var[4]}')

            # Now, we add the variables corresponding to each day
            for day in self.league_dates:
                coef.append(1)
                lower_bounds.append(0)
                upper_bounds.append(83)
                types.append('I')
                names.append(f'{day}_games')

            # Add the variables to the problem and set the problem sense
            prob_lp.variables.add(obj=coef, lb=lower_bounds, ub=upper_bounds, types=types, names=names)
            prob_lp.objective.set_sense(prob_lp.objective.sense.minimize)

        prob_lp = self.add_constraint_matrix(x_var_dict, diff_games_dict, prob_lp, objective)
        prob_lp.write(f"RescheduleFixture_{objective}.lp")

        return prob_lp

    def solve_lp(self, x_var_dict, diff_games_dict, prob_lp, objective):
        """
        Creates and solves the linear programming problem

        Parameters
        ----------
        x_var_dict: dict
            Dictionary of decision variables that will be included in the model. Each item in the dictionary will have
            the following structure
            (home_team, away_team, original_date, proposed_date): index
        diff_games_dict: dict
            In the case we are having the 'balanced' objective, this dictionary will be populated and have the following
            structure
            day: index.
            With day being each day of the tournament
        prob_lp: cplex.Cplex
            Cplex problem
        objective: str
            Objective function used for the model. Must be one of the following:
                - 'basic': Will try to maximize the number of scheduled games
                - 'min_diff': Will try to minimize the difference between the original date and the new date
                - 'balanced': We try to make the daily match assignment balanced: trying to minimize the number of games
                              is played by the team that has played the most games

        Returns
        -------
        output_df: pd.DataFrame
            Information of the new dates of each match
        """
        # Create the problem
        prob_lp = self.populate_by_row(x_var_dict, diff_games_dict, prob_lp, objective)

        # Solve the problem
        prob_lp.solve()

        # Get the solution variables
        x_variables = prob_lp.solution.get_values()

        # We'll create an output dataframe that has the proposed dates. We create lists where will we save each
        # proposed dates
        homes = []
        visitors = []
        original_dates = []
        game_dates = []
        proposed_dates = []
        reschedule = []

        # We check each variable to see its results
        for var in x_var_dict:
            if round(x_variables[x_var_dict[var]]) != 0:
                homes.append(var[0])
                visitors.append(var[1])
                original_dates.append(var[2])
                game_dates.append(var[3])
                proposed_dates.append(var[4])
                reschedule.append(1)

        # Create output dataframe
        output_df = pd.DataFrame({
            'home': homes,
            'visitor': visitors,
            'original_date': original_dates,
            'game_date': game_dates,
            'proposed_date': proposed_dates,
            'model_reschedule': reschedule
        })
        return output_df

    def calculate_needed_reschedules(self, output_df):
        """
        Calculates the matches that need to be rescheduled in the future because during the proposed reschedule date,
        some team suffers a COVID outbreak

        Parameters
        ----------
        output_df: pd.DataFrame
            Information of the new dates of each match

        Returns
        -------
        new_reschedules_list: list
            List with information of the matches that will need to be rescheduled again
        """
        new_reschedules_list = []

        for index, row in output_df.iterrows():
            # Create a variable that will indicate if we need a new reschedule
            check = 1
            home = row['home']
            visitor = row['visitor']
            # For both teams
            for team in [home, visitor]:
                # We check the "COVID" windows
                resched_windows = self.resched_windows_dict[team]

                # If the proposed date is in any of the windows, we add this match to the list of games that need to be
                # rescheduled
                for window in resched_windows:
                    if row['proposed_date'] in window:
                        check = 0

            if check == 0:
                match_info = {
                    'game': (row['home'], row['visitor']),
                    'original_date': row['proposed_date'],
                    'game_date': row['game_date'],
                }
                new_reschedules_list.append(match_info)
        return new_reschedules_list

    def assign_unassigned_matches(self, non_matched_matches, fixture):
        """
        Method that checks the matches that couldn't be assigned to any date and put them in the end
        of the tournament

        Parameters
        ----------
        non_matched_matches: list
            Information of these matches, including the teams, that are playing and the original and actual date the
            game was played
        fixture: pd.DataFrame
            Schedule that is going to be played

        Returns
        -------
        non_matched_output_df: pd.DataFrame
            Dataframe with new dates for the matches that are being rescheduled later
        """
        non_matched_output_df = pd.DataFrame()
        # For each match, we check the first date in which we can put a game
        for match in non_matched_matches:
            home_team = match['game'][0]
            away_team = match['game'][1]

            # We check the last game of each team
            home_team_games = fixture[(fixture['home'] == home_team) | (fixture['visitor'] == home_team)]
            away_team_games = fixture[(fixture['home'] == away_team) | (fixture['visitor'] == away_team)]

            last_game_home_team = np.max(home_team_games['original_date'])
            last_game_away_team = np.max(away_team_games['original_date'])
            last_game = np.max([last_game_home_team, last_game_away_team])
            previous_date_last_game = last_game - datetime.timedelta(days=1)

            check = 0
            # We find the first date in which we can play the game.
            # We just check to not have three games in a row with games
            while check == 0:
                home_team_games_filt = home_team_games[(home_team_games['original_date'] >= previous_date_last_game) & (
                        home_team_games['original_date'] <= last_game
                )]
                away_team_games_filt = away_team_games[(away_team_games['original_date'] >= previous_date_last_game) & (
                        away_team_games['original_date'] <= last_game
                )]
                # If both teams haven't played...
                if len(home_team_games_filt) < 2 and len(away_team_games_filt) < 2:
                    # We create our output dataframe of this particular dataframe
                    game_date = last_game + datetime.timedelta(days=1)
                    game_output_df = pd.DataFrame(data={
                        'home': [home_team],
                        'visitor': [away_team],
                        'game_date': [match['game_date']],
                        'proposed_date': [game_date],
                        'hard_reschedule': [1]
                    })
                    # To consider future reschedules, we merge this information to our reschedule
                    fixture = pd.merge(fixture, game_output_df[['home', 'visitor', 'game_date',
                                                                'proposed_date', 'hard_reschedule']], how='left',
                                       on=['home', 'visitor', 'game_date'])
                    fixture.loc[fixture['hard_reschedule'] == 1, 'original_date'] = fixture['proposed_date']
                    fixture

                    # Delete new columns in order to make future merges possible
                    fixture.drop(columns=['proposed_date', 'hard_reschedule'], inplace=True)

                    # We concat our result to our output dataframe
                    non_matched_output_df = pd.concat([non_matched_output_df, game_output_df], ignore_index=True)
                    check = 1
                else:
                    # If we can't put a match here, we check a new date
                    check
                    last_game = last_game + datetime.timedelta(days=1)
                    previous_date_last_game = previous_date_last_game + datetime.timedelta(days=1)

        return non_matched_output_df
