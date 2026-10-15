import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards
# from features.master_data_feature_additiion import master_df
# from features.batter_profiles.batter_profile import batter_profile_df
# from features.bowler_profile.bowler_profile import bowler_profile_df

from utils.utils import *
from models.tactical_model import *

st.set_page_config(
    "Match ups",
    page_icon = "🏏",
    initial_sidebar_state= 'expanded',
    layout='wide'
)

master_df = st.session_state.master_df.copy()
pitch_coord_df = st.session_state.pitch_coord_df.copy()
ground_angles_df = st.session_state.ground_angles_df.copy()
selected_team = st.session_state.get("selected_team")
bowling_type_map = st.session_state.bowling_type_map
model_params = st.session_state.model_params
model_params = st.session_state.model_params
team_color_df = st.session_state.get('team_color_df')

if 'selected_batsman' not in st.session_state:
    st.session_state.selected_batsman = ''

if 'selected_bowler' not in st.session_state:
    st.session_state.selected_bowler = ''

if 'rival_team' not in st.session_state:
    st.session_state.rival_team = None

if 'own_team_role' not in st.session_state:
    st.session_state.own_team_role = 'Batting'

if 'rival_team_role' not in st.session_state:
    st.session_state.rival_team_role = 'Bowling'

if 'params_changed' not in st.session_state:
    st.session_state.params_changed = False

if 'close_dialog' not in st.session_state:
    st.session_state.close_dialog = False

if 'show_predictive_analysis' not in st.session_state:
    st.session_state.show_predictive_analysis = False

# batter_bowler_matchups_df = pd.DataFrame({
#     'bowler' : df['bowler'].dropna().unique(),
#     'batsman' : df['batsman_on_strike'].dropna().unique()
# })

# batter_bowl_style_matchup_df = df.groupby(['batsman_on_strike', 'bowl_style']).reset_index(drop = True)







@st.cache_data(show_spinner="Running runs & wickets simulation...")
def cached_run_wickets_runs_tactical_simulations(
    input_df,
    params,
    context_features=[],
    action_features=[],
    batsman_in_question = None,
    bowler_in_question= None,
    bowling_type_in_question = None
):
    return run_wickets_runs_tactical_simulations(
        input_df=input_df,
        params=params,
        context_features=context_features,
        action_features=action_features,
        batsman_in_question=batsman_in_question
    )


@st.cache_data(show_spinner="Running shot predictor...")
def cached_run_shot_predictor_simulations(
    input_df,
    context_features=[],
    params = {},
    action_features=[],
    bowler_in_question=None,
    batsman_in_question = None,
    bowling_type_in_question =None
):
    return run_shot_predictor_simulations(
        input_df=input_df,
        context_features=context_features,
        action_features=action_features,
        params=params,
        batsman_in_question=batsman_in_question
    )


@st.cache_data(show_spinner="Calculating shot outcomes...")
def cached_run_shot_weighted_outcomes(
    input_df,
    shot_scenarios_df,     
    _runs_model,            
    _wickets_model,         
    _label_encoders,        
    context_features=[],
    action_features=[]
):
    return run_shot_weighted_outcomes(
        input_df=input_df,
        shot_scenarios_df=shot_scenarios_df,
        runs_model=_runs_model,
        wickets_model=_wickets_model,
        label_encoders=_label_encoders,
        context_features=context_features,
        action_features=action_features
    )



def get_batter_bowler_selection_components(df: pd.DataFrame, selected_team: str, rival_team: str, own_team_role: str):
    """
    Renders player/bowling type selectors based on the team's selected role.
    """
    analysis_target = "Specific Bowler"
    selected_bowling_type = None

    if own_team_role == 'Batting':
        # Let user choose whether to target a specific bowler or a bowling action/type
        analysis_target = st.pills(
            "Analyze Against",
            options=["Specific Bowler", "Bowling Type"],
            default="Specific Bowler",
            key="analysis_target_pill"
        )

        selected_batsman = st.selectbox(
            "Select Batsman from Your Team",
            options=df.loc[df['team_bat'] == selected_team, 'p_bat'].dropna().unique(),
            key='batsman_selector'
        )

        if analysis_target == "Specific Bowler":
            selected_bowler = st.selectbox(
                "Select Bowler from Opposition",
                options=df.loc[
                    (df['team_bowl'] == rival_team) &
                    (df['p_bat'] == selected_batsman),
                    'p_bowl'
                ].dropna().unique(),
                key='bowler_selector'
            )
        else:
            selected_bowler = None
            selected_bowling_type = st.selectbox(
                "Select Opposition Bowling Type",
                options=df.loc[
                    (df['team_bowl'] == rival_team) &
                    (df['p_bat'] == selected_batsman),
                    'bowl_style'
                ].dropna().unique(),
                key='bowling_type_selector'
            )

    else:
        # Bowling Role logic remains direct bowler-vs-batsman
        selected_bowler = st.selectbox(
            "Select Bowler from Your Team",
            options=df.loc[df['team_bowl'] == selected_team, 'p_bowl'].dropna().unique(),
            key='bowler_selector'
        )

        selected_batsman = st.selectbox(
            "Select Batsman from Opposition",
            options=df.loc[
                (df['team_bat'] == rival_team) &
                (df['p_bowl'] == selected_bowler),
                'p_bat'
            ].dropna().unique(),
            key='batsman_selector'
        )

    return selected_batsman, selected_bowler, selected_bowling_type, analysis_target

def render_batting_role_inputs(df: pd.DataFrame, selected_team: str, rival_team: str):
    """Renders options when the user's team is Batting."""
    analysis_target = st.pills(
        "Analyze Against",
        options=["Specific Bowler", "Bowling Type"],
        default="Specific Bowler"
    )

    batsman_options = get_batsmen_by_team(df, selected_team)
    selected_batsman = st.selectbox("Select Batsman", options=batsman_options)

    selected_bowler = None
    selected_bowling_type = None

    if analysis_target == "Specific Bowler":
        bowler_options = get_bowlers_by_team(df, rival_team)
        selected_bowler = st.selectbox("Select Bowler", options=bowler_options)
    else:
        style_options = get_bowling_styles_by_team(df, rival_team)
        selected_bowling_type = st.selectbox("Select Opposition Bowling Type", options=style_options)

    return selected_batsman, selected_bowler, selected_bowling_type, analysis_target


def render_bowling_role_inputs(df: pd.DataFrame, selected_team: str, rival_team: str):
    """Renders options when the user's team is Bowling."""
    batsman_options = get_batsmen_by_team(df, rival_team)
    selected_batsman = st.selectbox("Select Batsman", options=batsman_options)

    bowler_options = get_bowlers_by_team(df, selected_team)
    selected_bowler = st.selectbox("Select Bowler", options=bowler_options)

    return selected_batsman, selected_bowler, None, "Specific Bowler"


def save_matchup_session_state(
    selected_batsman: str,
    selected_bowler: str,
    selected_bowling_type: str,
    analysis_target: str,
    rival_team: str,
    own_team_role: str,
    rival_team_role: str
):
    """Encapsulates all session state mutations upon form submission."""
    st.session_state.selected_batsman = selected_batsman
    st.session_state.selected_bowler = selected_bowler
    st.session_state.selected_bowling_type = selected_bowling_type
    st.session_state.analysis_target = analysis_target
    st.session_state.rival_team = rival_team
    st.session_state.own_team_role = own_team_role
    st.session_state.rival_team_role = rival_team_role
    st.session_state.params_changed = True
    st.session_state.close_dialog = True



@st.dialog(" ")
def select_matchup_params(master_df: pd.DataFrame, selected_team: str):
    df = master_df.copy()

    with st.form("matchup_params_form"):
        st.write("### Select Matchup Parameters")

        own_team_role = st.selectbox(
            "Select What Role You Want to Analyze for Your Team",
            options=["Batting", "Bowling"],
            index=0
        )
        rival_team_role = "Bowling" if own_team_role == "Batting" else "Batting"

        rival_team_options = get_unique_teams(df, selected_team)
        rival_team = st.selectbox("Select Opposition", options=rival_team_options)

        # Delegate sub-component rendering based on role
        if own_team_role == "Batting":
            selected_batsman, selected_bowler, selected_bowling_type, analysis_target = render_batting_role_inputs(
                df, selected_team, rival_team
            )
        else:
            selected_batsman, selected_bowler, selected_bowling_type, analysis_target = render_bowling_role_inputs(
                df, selected_team, rival_team
            )

        submitted = st.form_submit_button("Proceed with Matchup Analysis")

    if submitted:
        save_matchup_session_state(
            selected_batsman,
            selected_bowler, # type: ignore
            selected_bowling_type, # type: ignore
            analysis_target,
            rival_team,
            own_team_role,
            rival_team_role
        )
        st.rerun()

        return (
            selected_batsman,
            selected_bowler,
            selected_bowling_type,
            analysis_target,
            rival_team,
            own_team_role,
            rival_team_role,
        )

    return None, None, None, None, None, None, None

if not st.session_state.close_dialog:
    select_matchup_params(
        master_df,
        selected_team
    )

if st.session_state.params_changed:

    with st.sidebar:

        with st.expander("Change Matchup Parameters", expanded=False):

            selected_team = st.selectbox(
                'Select Your Team',
                options=master_df['team_bat'].dropna().unique(),
                key='team_change_if_needed'
            )

            rival_team = st.selectbox(
                'Select Opposition',
                options=[
                    team for team in master_df['team_bat'].dropna().unique() if team != selected_team
                    ],
                key='rival_team_change_if_needed'
            )

            own_team_role = st.selectbox(
                "Select What Role You Want to Analyze for Your Team",
                options=['Batting', 'Bowling'],
                key='role_selector'
            )

            rival_team_role = 'Bowling' if own_team_role == 'Batting' else 'Batting'

            # Fetch selections based on selected parameters
            (
                selected_batsman,
                selected_bowler,
                selected_bowling_type,
                analysis_target
            ) = get_batter_bowler_selection_components(
                df=master_df,
                selected_team=selected_team,
                rival_team=rival_team,
                own_team_role=own_team_role
            )

            # Sync selection state
            st.session_state.selected_team = selected_team
            st.session_state.rival_team = rival_team
            st.session_state.own_team_role = own_team_role
            st.session_state.rival_team_role = rival_team_role
            st.session_state.selected_batsman = selected_batsman
            st.session_state.selected_bowler = selected_bowler
            st.session_state.selected_bowling_type = selected_bowling_type
            st.session_state.analysis_target = analysis_target

        # -------------------------------------------------------------
        # Dynamic Dataframe Filtering (Bowler vs. Bowling Type)
        # -------------------------------------------------------------
        if st.session_state.get('analysis_target') == "Bowling Type":
            matchup_bbb_df = master_df[
                (master_df['team_bat'] == selected_team) &
                (master_df['team_bowl'] == rival_team) &
                (master_df['p_bat'] == selected_batsman) &
                (master_df['bowl_style'] == selected_bowling_type) 
            ].reset_index(drop=True)
        else:
            matchup_bbb_df = master_df[
                (master_df['team_bat'] == selected_team) &
                (master_df['team_bowl'] == rival_team) &
                (master_df['p_bat'] == selected_batsman) &
                (master_df['p_bowl'] == selected_bowler)
            ].reset_index(drop=True)

        # Construct unique match ID string
        matchup_bbb_df['unq_match_id'] = (
            "At " + matchup_bbb_df['ground'].astype(str) + " v/s " + matchup_bbb_df['team_bowl'].astype(str)
        )

        # Filters
        match_choice = st.multiselect(
            "Choose the matches you want Analyze",
            options=list(matchup_bbb_df['unq_match_id'].unique()),
            default=list(matchup_bbb_df['unq_match_id'].unique()),
            key='match_choice'
        )

        phase_selector = st.multiselect(
            "Select Phases to Analyze",
            options=matchup_bbb_df['phase'].dropna().unique(),
            default=list(matchup_bbb_df['phase'].dropna().unique()),
            key='phase_selector_matchups'
        )

        innings_filter = st.multiselect(
            "Choose Innings (Either one or Both)",
            options=[1, 2],
            default=[1, 2],  # Changed from single int to list to prevent Streamlit Multiselect TypeError
            key='innings_choice'
        )




    selected_batsman = st.session_state.selected_batsman


    selected_bowler = st.session_state.selected_bowler

    matchup_bbb_df = matchup_bbb_df[
        (matchup_bbb_df['unq_match_id'].isin(match_choice)) &
        (matchup_bbb_df['phase'].isin(phase_selector)) &
        (matchup_bbb_df['inns'].isin(innings_filter))
    ].reset_index(drop = True)

    matchup_bbb_df['bowling_type']  = np.where(
        matchup_bbb_df['bowl_style'].isin(bowling_type_map['pace']),
        'pace',
        'spin'
    ) 


    batter_profile_df = get_batting_profile(
        matchup_bbb_df,
        group_cols = ['p_bat']
    )

    bowler_profile_df = get_bowling_profile(
        matchup_bbb_df,
        group_cols = ['p_bowl', 'bowl_style']
    )
    st.subheader(f"{selected_batsman} Batting Profile against {selected_bowler}")
    # st.dataframe(batter_profile)
    kpis_df = pd.DataFrame({
        "kpis": [
            "runs", "balls", "fours", "sixes",
            "dots", "dismissals", "bat_overall_sr",
            'bowl_overall_avg', 'bowl_overall_sr',
            "balls_per_boundary", "bowled_dismissals", "lbw_dismissals",
            "caught_dismissals", "stumped_dismissals",
        ],
        "alias": [
            "Runs Scored", "Balls Faced", "Fours Hit", "Sixes Hit",
            "Dot Balls ", "Dismissals", "Batsman Strike Rate",
            'Bowling Average', 'Bowler Strike Rate',
            "Balls per Boundary", "Bowled Dismissals", "LBW Dismissals",
            "Caught Dismissals", "Stumped Dismissals",
        ]
    })

    n_cols = int(np.floor(len(kpis_df)//2))

    for start in range(0, len(kpis_df), n_cols):

        cols = st.columns(n_cols)

        for col, (_, row) in zip(cols, kpis_df.iloc[start:start+n_cols].iterrows()):
            with col:

                if row['kpis'] in batter_profile_df.columns:
                    st.metric(
                        label=row["alias"],
                        value=batter_profile_df.loc[
                            (batter_profile_df['p_bat'] == selected_batsman)
                            , row["kpis"]].iloc[0] # type: ignore
                    )
                else:
                    st.metric(
                        label=row["alias"],
                        value=bowler_profile_df.loc[
                            (bowler_profile_df['p_bowl'] == selected_bowler) |
                            (bowler_profile_df['bowl_style'] == selected_bowling_type)
                            , row["kpis"]].iloc[0] # type: ignore
                    )

    style_metric_cards(
        background_color = "#2d2d2d",
        border_radius_px = 10,
        border_size_px= 3,
        border_left_color = team_color_df.loc[
            team_color_df['team_name'] == selected_team, 'primary_color'
        ].reset_index(drop = True).iloc[0],
        box_shadow = True
    )


    shot_count_df = matchup_bbb_df.groupby('shot').agg(
                    number_of_shots = ('shot' , 'count')
                        ).reset_index().sort_values(by = 'number_of_shots', ascending = False)

    shot_count_df['percent'] = (shot_count_df['number_of_shots']/sum(shot_count_df['number_of_shots'])) *100

    shot_count_df['display_text'] = shot_count_df.apply(
            lambda row: f"{int(row['number_of_shots'])} Shots - {row['percent']:.1f}%", 
            axis=1
        )
    # 2. Build Plotly Horizontal Bar Chart
    fig = go.Figure(
        go.Bar(
            x=shot_count_df['percent'],          # Pass Series, not column string
            y=shot_count_df['shot'],             # Pass Series, not column string
            orientation='h',                      # Horizontal bar orientation
            text = shot_count_df['display_text'], 
            textposition='auto',
            marker=dict(color='#2980b9')
        )
    )

    # 3. Format layout
    fig.update_layout(
        title=dict(text="Shot Selection Breakdown (%)", x=0.5),
        # xaxis=dict(title="Percentage of Total Shots (%)", suffix="%"),
        # yaxis=dict(title="Shot Type"),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=20, r=20, t=50, b=20)
    )

    st.plotly_chart(fig)




###### Expander for Pitch Heatmaps
    with st.expander("Metrics Heatmap on Line and Length", expanded=True):

        # Toggle switch: True = Batter View, False = Bowler View
        metric_choice = st.toggle(
            "Toggle View: Batter Map vs. Bowler / Bowler Map vs. Batter",
            key='batting_bowling_toggle'
        )

        mode = 'batting' if metric_choice else 'bowling'

        # Master Config Dictionary mapping both modes
        heatmap_config = {
            'batting': {
                'title': f'{matchup_bbb_df["p_bat"][0]} vs {matchup_bbb_df["p_bowl"][0]} (Pitch Heatmaps)',
                'plot_func': plot_batsman_pitch_heatmap,
                'radio_label': "Bowling Type",
                'radio_opts': ['Pace', 'Spin'],
                'df': pd.DataFrame({
                    'metrics': ['strike_rate', 'average', 'dot_ball_pct', 'boundary_pct'],
                    'alias': [
                        'Batting Strike Rate', 
                        'Batting Average', 
                        'Batting Dot Ball %', 
                        'Batting Boundary %'],
                    'reverse_colors': [False, False, True, False],
                    "max_threshold" : [
                                    250,
                                    'max',
                                    'max',
                                    'max'
                                ]
                })
            },
            'bowling': {
                'title': f'{matchup_bbb_df["p_bowl"][0]} vs {matchup_bbb_df["p_bat"][0]} (Pitch Heatmaps)',
                'plot_func': plot_bowler_pitch_heatmap,
                'radio_label': "Batter Hand",
                'radio_opts': list(matchup_bbb_df['bat_hand'].dropna().unique()),
                'df': pd.DataFrame({
                    'metrics': ['strike_rate', 'average', 'dot_ball_pct', 'boundary_pct'],
                    'alias': ['Bowling Strike Rate', 'Bowling Average', 'Bowling Dot Ball %', 'Bowling Boundary Conceded %'],
                    'reverse_colors': [True, True, False, True]
                })
            }
        }

        current_cfg = heatmap_config[mode]
        metrics_df = current_cfg['df']

        st.write(f"### {current_cfg['title']}")

        # Render columns dynamically
        heatmap_cols = st.columns(len(metrics_df))

        for i, row in metrics_df.iterrows():
            with heatmap_cols[i]:
                st.markdown(f"**{row['alias']}**")

                # # Radio Selector (Pace/Spin or Righty/Lefty based on mode)
                # selected_filter = st.radio(
                #     label=current_cfg['radio_label'],
                #     options=current_cfg['radio_opts'],
                #     label_visibility='collapsed',
                #     key=f"{mode}_filter_radio_{row['metrics']}_{i}"
                # )

                # Build plot parameters dynamically based on selected mode
                if mode == 'batting':
                    # Safe fallback for batting style formatting
                    raw_style = matchup_bbb_df['bat_hand'].iloc[0] if not matchup_bbb_df.empty else "right_hand_bat"
                    formatted_style = str(raw_style).lower().strip().replace(" ", "_")

                    plot_kwargs = {
                        'bat_df': matchup_bbb_df,
                        'pitch_coords': pitch_coord_df,
                        'batsman_name': selected_batsman,
                        'bowling_type': matchup_bbb_df['bowling_type'][0].lower(),
                        'batting_style': formatted_style,
                        'metric': row['metrics'],
                        'pitch_color': '#c9a777'
                    }
                else:
                    # Bowler Mode parameters
                    bowling_type = matchup_bbb_df['bowling_type'].iloc[0] if not matchup_bbb_df.empty else "pace"

                    plot_kwargs = {
                        'bowl_df': matchup_bbb_df,
                        'pitch_coords': pitch_coord_df,
                        'bowler_name': selected_bowler,
                        'bowling_type': bowling_type,
                        'batting_style': matchup_bbb_df['bat_hand'][0],
                        'metric': row['metrics'],
                        'pitch_color': '#c9a777'
                    }

                    # Include reverse_colors if your plot function accepts it
                    if 'reverse_colors' in row:
                        plot_kwargs['reverse_colors'] = row['reverse_colors']

                # Call function directly from config mapping
                heatmap_fig = current_cfg['plot_func'](**plot_kwargs)

                # Display Heatmap
                st.plotly_chart(
                    heatmap_fig,
                    width = 'stretch',
                    key=f"{mode}_hm_{row['metrics']}_fig"
                )


###### Expander for Field direction - Wagon Wheel

    with st.expander("Wagon Wheel showing Areas targeted", expanded = False):

        # Toggle switch: True = Batter View, False = Bowler View
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        with filter_col1:
            available_lines = list(matchup_bbb_df['line'].dropna().unique())
            selected_lines = st.multiselect(
                "Filter Line",
                options=available_lines,
                default=available_lines,
                key=f"{mode}_wagon_wheel_line_filter"
            )
 
        with filter_col2:
            available_lengths = list(matchup_bbb_df['length'].dropna().unique())
            selected_lengths = st.multiselect(
                "Filter Length",
                options=available_lengths,
                default=available_lengths,
                key=f"{mode}_wagon_wheel_length_filter"
            )

        with filter_col3:
            available_shots = list(matchup_bbb_df['shot'].dropna().unique())
            selected_shots = st.multiselect(
                "Filter Shots",
                options = available_shots,
                default = available_shots,
                key = f"{mode}_shots_wagon_wheel"
            )


        # 3. Apply Line & Length Filters to Dataframe
        filtered_df = matchup_bbb_df.copy()

        if selected_lines and 'line' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['line'].isin(selected_lines)]

        if selected_lengths and 'length' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['length'].isin(selected_lengths)]

        if selected_shots and 'shot' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['shot'].isin(selected_shots)]


        fig = plot_player_field_partitions_with_runs(
            # ground_fig=fig,
            ground_angles_df=ground_angles_df,
            master_df=matchup_bbb_df,
            player_name=selected_batsman,
            player_in_question='batsman',
            batting_style =  matchup_bbb_df['bat_hand'].iloc[0]
        )


        st.plotly_chart(
            fig,
            key = "bowler_field_partitions",
        )



########### New section with predictive analysis

    st.markdown("---")
    st.subheader("Simulated Bowler Action Strategies")

    action_features = st.multiselect(
        'Select Tactical Actions to Model',
        options=['line', 'length', 'bowl_kind', 'bowl_release_point', 'bat_hand'],
        default=['line', 'length'],
        key='select_bowling_action_feat_matchups'
    )


    if st.button(
        "Run Predictive Simulation",
        key = 'matchup_simulator_btn',
        type = 'primary'
    ):

        st.session_state.show_predictive_analysis = True
        st.rerun()


        combined_context_features = [

            'inns', 'p_bat', 'bowler', 'phase','bowl_style',
            'bat_hand', 'bowl_kind', 'bowl_speed_category',
            'target','inns_runs_rem','inns_rr', 'inns_runs','inns_wkts', 
            'batsman_nic_runs_counter', 'event_occuring','inns_rrr','inns_balls_rem',
            'last_10_balls_runs_vs_bow_team','last_10_balls_boundaries_vs_bow_team','last_10_balls_wickets_vs_bow_team',
            'last_10_balls_dots_vs_bow_team', 'last_30_balls_runs_vs_bow_team', 'last_30_balls_boundaries_vs_bow_team',
            'last_30_balls_wickets_vs_bow_team','last_30_balls_dots_vs_bow_team', 'last_6_balls_runs_vs_bowler',
            'prev_1_balls_line_faced_by_batsman', 'last_6_balls_boundaries_vs_bowler',
            'prev_2_balls_line_faced_by_batsman','prev_3_balls_line_faced_by_batsman',
            'prev_4_balls_line_faced_by_batsman','prev_5_balls_line_faced_by_batsman',
            'prev_1_balls_length_faced_by_batsman','prev_2_balls_length_faced_by_batsman',
            'prev_3_balls_length_faced_by_batsman','prev_4_balls_length_faced_by_batsman',
            'prev_5_balls_length_faced_by_batsman','prev_6_balls_length_faced_by_batsman',
            'prev_1_balls_shot_by_batsman','prev_2_balls_shot_by_batsman',
            'prev_3_balls_shot_by_batsman','prev_4_balls_shot_by_batsman', 'prev_6_balls_line_faced_by_batsman',
            'prev_5_balls_shot_by_batsman','prev_6_balls_shot_by_batsman',
            'prev_length_by_bowler','prev_line_by_bowler','prev_shot_vs_bowler','prev_1_balls_line_by_bowler',
            'prev_2_balls_line_by_bowler','prev_3_balls_line_by_bowler','prev_4_balls_line_by_bowler',
            'prev_5_balls_line_by_bowler','prev_6_balls_line_by_bowler','prev_1_balls_length_by_bowler',
            'prev_2_balls_length_by_bowler','prev_3_balls_length_by_bowler','prev_4_balls_length_by_bowler',
            'prev_5_balls_length_by_bowler','prev_6_balls_length_by_bowler','prev_1_balls_shot_vs_bowler',
            'prev_2_balls_shot_vs_bowler','prev_3_balls_shot_vs_bowler','prev_4_balls_shot_vs_bowler',
            'prev_5_balls_shot_vs_bowler','prev_6_balls_shot_vs_bowler',
        ]


        shot_context_features = [
            'inns', 'bowler', 'phase','bowl_style', 'bowl_speed_category', 'event_occuring',
            'target','inns_runs_rem','inns_rr', 'inns_runs','inns_wkts', 
            'inns_rrr','inns_balls_rem',
            'last_10_balls_runs_vs_bow_team','last_10_balls_boundaries_vs_bow_team','last_10_balls_wickets_vs_bow_team',
            'last_10_balls_dots_vs_bow_team', 'last_30_balls_runs_vs_bow_team', 'last_30_balls_boundaries_vs_bow_team',
            'last_30_balls_wickets_vs_bow_team','last_30_balls_dots_vs_bow_team', 'last_6_balls_runs_vs_bowler','last_6_balls_boundaries_vs_bowler',
            'prev_length_by_bowler','prev_line_by_bowler','prev_shot_vs_bowler','prev_1_balls_line_by_bowler',
            'prev_2_balls_line_by_bowler','prev_3_balls_line_by_bowler','prev_4_balls_line_by_bowler',
            'prev_5_balls_line_by_bowler','prev_6_balls_line_by_bowler','prev_1_balls_length_by_bowler',
            'prev_2_balls_length_by_bowler','prev_3_balls_length_by_bowler','prev_4_balls_length_by_bowler',
            'prev_5_balls_length_by_bowler','prev_6_balls_length_by_bowler','prev_1_balls_shot_vs_bowler',
            'prev_2_balls_shot_vs_bowler','prev_3_balls_shot_vs_bowler','prev_4_balls_shot_vs_bowler',
            'prev_5_balls_shot_vs_bowler','prev_6_balls_shot_vs_bowler',
        ]



        if st.session_state.get('analysis_target') == "Bowling Type":
            (
                # combined_scenario_df,
                label_encoders,
                runs_model,
                wickets_model
            ) = cached_run_wickets_runs_tactical_simulations(
                    input_df = master_df,
                    params = model_params,
                    context_features= combined_context_features,
                    action_features = action_features,
                    batsman_in_question = selected_batsman,
                    bowling_type_in_question = selected_bowling_type
                )

            (
                shot_scenario_df,
                shot_label_encoders,
                shot_model
            ) = cached_run_shot_predictor_simulations(
                    input_df= master_df,
                    context_features=shot_context_features,
                    action_features=action_features,
                    bowling_type_in_question = selected_bowling_type 
                )

            
        else:

            (
                # combined_scenario_df,
                label_encoders,
                runs_model,
                wickets_model
            ) = cached_run_wickets_runs_tactical_simulations(
                    input_df = master_df,
                    params = model_params,
                    context_features= combined_context_features,
                    action_features = action_features,
                    batsman_in_question = selected_batsman,
                    bowler_in_question = selected_bowler # type: ignore
                )
                        
            (
                shot_scenario_df,
                shot_label_encoders,
                shot_model
            ) = cached_run_shot_predictor_simulations(
                    input_df= master_df,
                    context_features=shot_context_features,
                    action_features=action_features,
                    bowler_in_question = selected_bowler  # type: ignore
                )



        (shot_breakdown_df, delivery_expectation, final_scenario_df) = cached_run_shot_weighted_outcomes( # type: ignore
            input_df = matchup_bbb_df,
            shot_scenarios_df = shot_scenario_df,
            _runs_model=runs_model,
            _wickets_model= wickets_model,
            _label_encoders=shot_label_encoders,
            context_features = combined_context_features,
            action_features = action_features,
        )
        

        _,_,colp = st.columns(3)

        with colp:

            phase_chooser = st.segmented_control(
                "Select phase for Results",
                options = matchup_bbb_df['phase'].unique(),
                width = 'stretch',
                selection_mode = 'multi',
                key = 'matchup_phase_selector_heatmap'
            )


        # st.write("Step B: Shot result probability")
        # st.dataframe(shot_breakdown_df)
        
        ##### Authenticity sorted. Now build batter and bowler runs wickets heatmaps
        
        if delivery_expectation is not None:

            st.dataframe(
                delivery_expectation[[
                    'phase',
                    'line',
                    'length',
                    'highest_prob_shot',
                    'highest_shot_probability'
                ]].round(2),
                width = 'stretch',
                hide_index = True,
                key = 'delivery_expectations_df'
            )
        
        
            delivery_expectation_by_phase = delivery_expectation[
                delivery_expectation['phase'].isin(phase_chooser)
            ].reset_index(drop = True)
        
            predictive_heatmap_df_2 = pd.DataFrame({
                'labels' : [
                    'Expected Runs', 
                    '',
                    'Wicket Probabilities'
                    # 'High Pressure Runs', 
                    # 'Typical Runs', 
                    # "Maximum Runs of a Line Length", 
                    # 'Total Wickets in 100 balls'
                ], 
                'metrics' : [
                    'batter_exp_runs', 
                    '', 
                    'batter_wicket_prob',                    
                    # 'max_damage_ceiling', 
                    # 'exp_wkts_per_100_balls'
                ],
                'agg_func' : [
                    'sum', 
                    '', 
                    'sum', 
                    # 'sum', 
                    # 'sum'
                ],
                "max_threshold" : [
                    'max',
                    '',
                    'max',
                ]
            })
        
            # heatmap_metrics2 = st.multiselect(
            #     'Select Metrics to Visualize',
            #     options= predictive_heatmap_df['labels'].unique(),
            #     max_selections= 5,
            #     # default= list(predictive_heatmap_df['labels'].unique())[0:3],
            #     key = 'select_metrics_to_visualize_heatmap3'
            # )
        
            n_cols = 3
            colz = st.columns(n_cols)
        
        
            for i, rows in predictive_heatmap_df_2.iterrows():
        
                with colz[i]:
                    if i % 2 == 0:
                        # st.text(rows['labels'])
                        st.plotly_chart(
                            plot_batsman_pitch_heatmap(
                                delivery_expectation_by_phase,
                                pitch_coord_df,
                                batsman_name= selected_batsman,
                                # batting_style= matchup_bbb_df[
                                #         'bat_hand'
                                #     ][0].lower().strip().replace(" ", "_"),
                                metric = rows['metrics'],
                                agg_func = rows['agg_func'],
                                pitch_color = '#c9a777'
                            ),
                            key = f'exp_runs_map_{i}'
                        )
                    else:
        
                        st.markdown("** Most Probable Shots by Line Length and Phase **")
                        st.dataframe(
                            delivery_expectation[[
                                'phase',
                                'line',
                                'length',
                                'highest_prob_shot',
                                'highest_shot_probability'
                            ]].round(2),
                            width = 'stretch',
                            key = 'delivery_expectations_df'
                        )
        
        


