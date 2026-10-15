import streamlit as st
import pandas as pd
import numpy as np
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.great_tables import  great_tables


from great_tables import GT


from model.tactical_model import *

from utils.utils import *

st.set_page_config(
    "Batsman Profile",
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
team_color_df = st.session_state.get('team_color_df')
pace_cat_dict = st.session_state.get('pace_cat_dict')
release_ht_dict = st.session_state.get('release_ht_dict')

if 'show_predictive_analysis' not in st.session_state:
    st.session_state.show_predictive_analysis = False


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





with st.sidebar:

    season = st.segmented_control(
        "Select Season",
        options = list(master_df['year'].unique()),
        selection_mode = 'single',
        width = 'stretch',
        default = list(master_df['year'].unique())[0],
        key = 'season_selectbox',
    )

    st.session_state.season = season


    chosen_team = st.selectbox(
        "Your Team",
        options = master_df['team_bat'].unique(),
        index = list(master_df['team_bat'].unique()).index(selected_team),
        key = 'chosen_team',
    )

    team_batting_df_bbb = master_df.query('team_bat == @chosen_team').reset_index().copy()

    team_batting_df_bbb['unq_match_id'] =(
        "At "
        + team_batting_df_bbb['ground']
        + " v/s "
        + team_batting_df_bbb['team_bowl']
    )

    match_choice = st.multiselect(
        "Choose the matches you want Analyze",
        options = list(team_batting_df_bbb['unq_match_id'].unique()),
        default = list(team_batting_df_bbb['unq_match_id'].unique()),
        key = 'match_choice'
    )

    team_batting_df_bbb = team_batting_df_bbb.loc[
        (team_batting_df_bbb['unq_match_id'].isin(match_choice)) &
        (team_batting_df_bbb['year'] == season)
    ].reset_index(drop = True)

    batsman_choice = st.selectbox(
        'Select Batsman for Analysis',
        options = team_batting_df_bbb['bat'].unique(),
        index = 0,
        key = 'selected_batsman'
    )

    innings_filter = st.multiselect(
        "Choose Innings (Either one or Both)",
        options = [1,2],
        default = 1,
        key = 'innings_choice'
    )



cols = st.columns([1,1])

with cols[0]:

    phase_select = st.multiselect(
        'Select Phase',
        options = team_batting_df_bbb['phase'].unique(),
        default = team_batting_df_bbb['phase'].unique()[0],
        key = 'chosen_phase'
    )


team_batting_df_bbb = team_batting_df_bbb.loc[
    (team_batting_df_bbb['inns'].isin(innings_filter)) &
    (team_batting_df_bbb['phase'].isin(phase_select)) &
    (team_batting_df_bbb['bat'].eq(batsman_choice))
].reset_index(drop = True)



with cols[1]:

    bowling_action_selector = st.multiselect(
        "Choose Bowling Style",
        options = team_batting_df_bbb['bowl_style'].unique(),
        default= team_batting_df_bbb['bowl_style'].unique()[0],
        key = 'bowling_action_selected'
    )

    team_batting_df_bbb = team_batting_df_bbb.loc[
        (team_batting_df_bbb['bowl_style'].isin(bowling_action_selector))
    ].reset_index(drop = True)



batter_profile_df = get_batting_profile(
    team_batting_df_bbb,
    ['bat']
)

kpis_df = pd.DataFrame({
    "kpis": [
        "runs", "balls", "fours", "sixes",
        "dots", 
        "dismissals", "bat_overall_sr",
        "balls_per_boundary", "bowled_dismissals", "lbw_dismissals",
        "caught_dismissals", "stumped_dismissals",
    ],
    "alias": [
        "Runs Scored", "Balls Faced", "Fours Hit", "Sixes Hit",
        "Dots Faced", 
        "Dismissals", "Strike Rate",
        "Balls per Boundary", "Bowled Dismissals", "LBW Dismissals",
        "Caught Dismissals", "Stumped Dismissals",
    ]
})


n_cols = int(np.floor(len(kpis_df)//2))
for start in range(0, len(kpis_df), n_cols):

    cols = st.columns(n_cols)

    for col, (_, row) in zip(cols, kpis_df.iloc[start:start+n_cols].iterrows()):
        with col:
            st.metric(
                label=row["alias"],
                value=batter_profile_df.loc[
                    (batter_profile_df['bat'] == batsman_choice) 
                    , row["kpis"]].iloc[0] # type: ignore
            )
style_metric_cards(
    background_color = "#2d2d2d",
    border_radius_px = 10,
    border_size_px= 3,
    border_left_color = team_color_df.loc[
        team_color_df['team_name'] == selected_team, 'primary_color' # type: ignore
    ].reset_index(drop = True).iloc[0],
    box_shadow = True
)


team_batting_df_bbb['Bowler Release Height'] = team_batting_df_bbb['bowl_release_point'].map(release_ht_dict)
team_batting_df_bbb['Ball Speed Category'] = team_batting_df_bbb['bowl_speed_category'].map(pace_cat_dict)



bowling_type_body_w_dict = {
    bowl_type : list(team_batting_df_bbb.loc[
        team_batting_df_bbb['bowl_style'].isin([bowl_type]),'Ball Speed Category'
    ].unique()) for bowl_type in bowling_action_selector
}

ball_speed_df = get_body_weight_metrics(
    team_batting_df_bbb,
    ['bowl_style', 'Ball Speed Category']
)

release_height_df = get_body_weight_metrics(
    team_batting_df_bbb,
    ['bowl_style', 'Bowler Release Height']
)

metric_cols = [
    "Number of Deliveries",
    "Runs Scored",
    "Wickets Lost",
    "False Shot %"
]

with st.expander("Ball Speed and Release Height", expanded=False):

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        table1 = (
            GT(ball_speed_df)
            .tab_header(
                title="Batting Metrics by Ball Speed",
                subtitle=f"{batsman_choice}"
            )
            .data_color(
                domain=[
                    ball_speed_df[metric_cols].apply(pd.to_numeric).max().max(),
                    0, 
                ],
                palette=["#30204D", "#9474C5", "#FF8A65"],
                na_color="white",
            )
            .tab_stub(
                rowname_col="Ball Speed Category",
                groupname_col="bowl_style"
            )
        )

        great_tables(table1)

    with col2:
        table2 = (
            GT(release_height_df)
            .tab_header(
                title="Batting Metrics by Release Height",
                subtitle=f"{batsman_choice}"
            )
            .data_color(
                domain=[
                    release_height_df[metric_cols].apply(pd.to_numeric).max().max(),
                    0, 
                ],
                palette=["#30204D", "#9474C5", "#FF8A65"],
                na_color="white",
            )            
            .tab_stub(
                rowname_col="Bowler Release Height",
                groupname_col="bowl_style"
            )
        )

        great_tables(table2)




### ------ On-Pitch Heatmaps -------- ###

heatmap_metrics_df = pd.DataFrame({
    'metrics' : [
        'bat_overall_sr', 
        'bat_overall_avg',
        'dot_ball_pct',
        'false_shot_%'
    ],
    'alias' : [
        'Srike Rate', 
        "Average",
        'Dot Ball %',
        'Shots not in Control'
    ],
    'agg_func' : [
        '',
        '',
        '',
        'sum'
    ],
    'max_threshold' : [
        250,
        None,
        None,
        None
    ]
})


cols_ht = st.columns([4,1])
with cols_ht[1]:

    metric_selector = st.radio(
        "Select metric to visualize",
        options=heatmap_metrics_df['alias'],
        index=0,
        key='map_metrics_selector',
    )


with cols_ht[0]:
    st.markdown(
        f"""
        <div style="text-align: center;
        font-size : 28px;">
            <strong> Batter Metrics Heatmaps: {metric_selector}  </strong>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.plotly_chart(
            plot_single_metric_heatmap(
            team_batting_df_bbb,
            heatmap_metrics_df[heatmap_metrics_df['alias'].eq(metric_selector)]['metrics'],
            MAX_THRESHOLD= heatmap_metrics_df[heatmap_metrics_df['alias'].eq(metric_selector)]['max_threshold'].iloc[0],
            line_order = ['wide_outside_offstump', 'outside_offstump', 'on_the_stumps', 'down_leg', 'wide_down_leg'],
            length_order = ['full_toss', 'yorker', 'full', 'good_length', 'short_of_a_good_length', 'short'],
            metric_title = heatmap_metrics_df[heatmap_metrics_df['alias'].eq(metric_selector)]['alias'].iloc[0]
        ),
        width = 'content',
        key = f'heatmap_{heatmap_metrics_df["metrics"][0]}_fig'
    )


#### ----- Field wise analysis ---- ####


st.markdown(
    f"""
    <div style="text-align: center;
    font-size : 28px;">
        <strong> Wagon Wheel By Outcomes </strong>
    </div>
    """,
    unsafe_allow_html=True
)
        
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)

with filter_col1:
    available_lines = list(team_batting_df_bbb['line'].dropna().unique())
    selected_lines = st.multiselect(
        "Filter Line",
        options=available_lines,
        default=available_lines,
        key="bowling_wagon_wheel_line_filter"
    )

with filter_col2:
    available_lengths = list(team_batting_df_bbb['length'].dropna().unique())
    selected_lengths = st.multiselect(
        "Filter Length",
        options=available_lengths,
        default=available_lengths,
        key="bowling_wagon_wheel_length_filter"
    )

with filter_col3:
    available_shots = list(team_batting_df_bbb['shot'].dropna().unique())
    selected_shots = st.multiselect(
        "Filter Shots",
        options = available_shots,
        default = available_shots,
        key = "bowling_shots_wagon_wheel"
    )

with filter_col4:
    available_outcomes = list(team_batting_df_bbb['outcomes'].dropna().unique())
    runs_selected =  st.multiselect(
        "Filter Outcome",
        options = available_outcomes,
        default = available_outcomes,
        key = "bowling_outcomes_wagon_wheel"
    )


filtered_df = team_batting_df_bbb.copy()

if selected_lines and 'line' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['line'].isin(selected_lines)].reset_index(drop = True)

if selected_lengths and 'length' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['length'].isin(selected_lengths)].reset_index(drop = True)

if selected_shots and 'shot' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['shot'].isin(selected_shots)].reset_index(drop = True)

if 'batruns' in filtered_df.columns:
    filtered_df = filtered_df[filtered_df['outcomes'].isin(runs_selected)].reset_index(drop = True)

ground_angles_df = master_df[[
    'wagonx', 'wagony', 'wagonzone'
]].reset_index(drop = True)


fig = plot_player_wagon_wheel(
    filtered_df,
    player_name=batsman_choice,
    player_in_question='batsman',
    max_coord=filtered_df['wagonx'].max(),
    min_coord= filtered_df['wagonx'].min(),
    image_size=600
    
)




st.plotly_chart(fig)





