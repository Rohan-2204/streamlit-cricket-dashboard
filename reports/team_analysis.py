import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_extras.resizable_columns import resizable_columns
from streamlit_extras.metric_cards import style_metric_cards

from utils.utils import *


st.set_page_config(
    "Team Analysis",
    page_icon = "🏏",
    initial_sidebar_state= 'expanded',
    layout='wide'
)


master_df = st.session_state.master_df.copy()
selected_team = st.session_state.get("selected_team")
team_color_df = st.session_state.get('team_color_df')




def load_css(css_file: str = "main.css"):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css('./assets/main.css')

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

    st.session_state.selected_team = chosen_team
    
    phase_selected = st.multiselect(
        "Game phase",
        options = list(master_df['phase'].unique()),
        default = [master_df['phase'].unique()[0]],
        key = 'selected_phase',
    )

    # opposition = master_df['team_bat'].where(
    #     master_df['team_bowl'].eq(chosen_team),
    #     master_df['team_bowl']
    # )


    master_df['unq_match_id'] =(
         # 'No ' 
        # + team_batting_df_bbb['p_match'].str.partition("_")[0]
        chosen_team  # type: ignore
        + " v/s "
        + master_df['team_bat'].where(
            master_df['team_bowl'].eq(chosen_team),
            master_df['team_bowl']
        )
        + " At "
        + master_df['ground']
    )

    team_batting_df_bbb = master_df.query('team_bat == @chosen_team').reset_index(drop = True).copy()
    team_bowling_df_bbb = master_df.query('team_bowl == @chosen_team').reset_index(drop = True).copy()

    match_choice = st.multiselect(
        "Choose the matches you want Analyze",
        options = list(team_batting_df_bbb['unq_match_id'].unique()),
        default = list(team_batting_df_bbb['unq_match_id'].unique()),
        key = 'match_choice'
    )

    team_batting_df_bbb = team_batting_df_bbb.loc[
        # (team_batting_df_bbb['phase'].isin(phase_selected)) &
        (team_batting_df_bbb['year'] == season) &
        (team_batting_df_bbb['unq_match_id'].isin(match_choice))
    ].reset_index()

    team_bowling_df_bbb = team_bowling_df_bbb.loc[
        # (team_bowling_df_bbb['phase'].isin(phase_selected)) &
        (team_bowling_df_bbb['year'] == season) &
        (team_bowling_df_bbb['unq_match_id'].isin(match_choice))
    ].reset_index()


    st.info(f"{chosen_team} Bat 1st Innings: {len(team_batting_df_bbb.loc[team_batting_df_bbb['inns'] == 1].groupby(['p_match']))} out of {len(team_batting_df_bbb.groupby('p_match'))} Matches")



overall_batsman_stats_df = get_batting_profile(
    master_df.loc[
        (master_df['year'].eq(season))
    ],
    ['team_bat', 'bat']
)


overall_bowler_stats_df = get_bowling_profile(
    master_df.loc[
        (master_df['year'].eq(season))
    ],
    ['team_bowl', 'bowl']
)


all_team_speed_cat_batting_df = get_batting_profile(
    master_df,
    ['year', 'phase', 'bowl_kind', 'bowl_speed_category', 'bowl_release_point']
)

all_team_speed_cat_bowling_df = get_bowling_profile(
    master_df,
    ['year', 'phase', 'bowl_kind', 'bowl_speed_category', 'bowl_release_point']
)

### KPIs

# Batting KPIs : balls per boundary, dismissal rate, dot %, run rate, batsman avg strike rate


tab1, tab2 = st.tabs(['Team Batting Stats', "Team Bowling Stats"])

if chosen_team is not None and chosen_team in master_df['team_bat'].unique():
    with tab1:

        # st.dataframe(overall_batsman_stats_df)
        ### Overall batsman stats kpis : runs, fours, sixes, overall strike rate
        batsman_stats_kpis = pd.DataFrame({
            'kpis' : ['runs', 'fours', 'sixes', 'bat_overall_sr'],
            'alias' : ['Most Runs', 'Most 4s', ' Most 6s', 'Highest Strike Rate'],
            'agg_func' : ['max', 'max', 'max', 'max'],
            # 'delta_value' : ['normal', 'normal', 'normal', 'normal']
        })

        assert team_color_df is not None
        primary_color = team_color_df.loc[
                team_color_df['team_name'] == chosen_team, 'primary_color' # type: ignore
            ].reset_index(drop = True).iloc[0]

        secondary_color = team_color_df.loc[
                team_color_df['team_name'] == chosen_team, 'secondary_color' # type: ignore
            ].reset_index(drop = True).iloc[0]
        
        cols0 = resizable_columns(len(batsman_stats_kpis))


        temp_overall_df = overall_batsman_stats_df.loc[
            (overall_batsman_stats_df['team_bat'].eq(chosen_team)) &
            (overall_batsman_stats_df['balls'] >= 50)
        ].reset_index(drop = True)
        

        for i, row in batsman_stats_kpis.iterrows():


            with cols0[i]: # type: ignore
                # st.markdown(f"**Season - {row['alias']}**")
                st.markdown(
                        f"""
                        <div style="text-align: center;
                        font-size : 20px;">
                            <strong>Season - {row['alias']}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                
                st.metric(
            
                    label = temp_overall_df[
                        temp_overall_df[
                                row['kpis']] == temp_overall_df[row['kpis']
                            ].agg(row['agg_func'])
                        ]['bat'].iloc[0],# type: ignore
                    value = temp_overall_df[row['kpis']].max(), # type: ignore
                )

                st.markdown(
                        f"""
                        <div style="
                        text-align: center;
                        font-size : 20px;">
                            <strong>Phase - {row['alias']}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                phase_wise_batsman_stats_df = get_batting_profile(
                    team_batting_df_bbb[
                        team_batting_df_bbb['phase'].isin(phase_selected)
                    ],
                    ['team_bat', 'bat']
                )
                st.dataframe(
                    phase_wise_batsman_stats_df[['bat', row['kpis']]]
                    # groupby('bat').agg(
                    #     row['kpis'] = (row['kpis'], 'sum')
                    # )
                    .sort_values(by = row['kpis'], ascending = False)\
                    .rename(columns = {
                        'bat' : "Batsman Name",
                        row['kpis'] : row['alias']
                    }).head(5),
                    hide_index=True
                ) # type: ignore

        style_metric_cards(
            background_color = "#2d2d2d",
            # background_color = primary_color,
            border_radius_px = 10,
            border_size_px=3,
            border_left_color = primary_color,
            box_shadow = True
        )


        batting_kpis_delta = pd.DataFrame({
            'kpis' : [
                'balls_per_boundary', 
                'dismissal_rate', 
                'dot_ball_pct', 
                'run_rate'
            ],

            'alias' : [
                'Balls per Boundary', 
                'Dismissal Rate (Balls/Wicket)', 
                'Dot Ball %', 
                'Run Rate'
            ],
            'delta_value' : [
                'inverse', 
                'normal', 
                'inverse', 
                'normal'
            ]
        })


        cols = resizable_columns(
            len(batting_kpis_delta), 
            # gap = 'xxlarge'
        )
        
        team_batting_df = get_batting_profile(
            team_batting_df_bbb.loc[
                    (team_batting_df_bbb["phase"].isin(phase_selected))
            ],
            ['team_bat']
        )

        for i, row in batting_kpis_delta.iterrows():
            kpi = row['kpis']
            label = row['alias']

            with cols[i]: # type: ignore

                team_value = team_batting_df[kpi].iloc[0] # type: ignore

                all_teams_batting_df = get_batting_profile(
                        master_df.loc[
                        (master_df["phase"].isin(phase_selected)) &
                        (master_df['year'].isin([season])),
                        ],
                        ['phase']
                    )
                
                phase_avg = all_teams_batting_df[kpi].iloc[0]

                # ].iloc[0]

                # st.write(team_value)

                st.metric(
                    label=label,
                    value=round(team_value, 2),
                    delta=round(team_value - phase_avg, 2),
                    delta_color = row['delta_value'],
                    width = 'stretch',
                    # border = True
                )

        run_rate_df = master_df[master_df['year'].eq(season)].groupby(
                    ['inns','team_bat', 'over_num']).agg(
                        crr = ('inns_rr', 'mean'),
                ).reset_index()


        # """
        # try making the backgroundColor change acc to team from config.toml and set metric card 
        #     background color as black and text as primary logo color of the team.

        #     Take screenshots
        # """

        phase_color_map = {
            "Powerplay phase":lighten_color(primary_color, 0.85),
            "Middle phase": lighten_color(primary_color, 0.55),
            "Death phase":  primary_color,
        }



        with st.container(
            # border = True,
            key = 'partnerships'
        ):
            col1, col2 = resizable_columns(2)
            with col1:
                st.plotly_chart(
                    plot_partnership_runs(
                        team_batting_df_bbb,
                        selected_team=chosen_team,
                        color_map=phase_color_map,
                        innings=[1],
                        secondary_color=secondary_color,
                        title = 'Partnership Runs by wickets over season in 1st Innings',
                        bar_color = primary_color
                    ),
                    width = 'stretch',
                    key = 'partnership_plot_1'
                )

            with col2:
                st.plotly_chart(
                    plot_partnership_runs(
                        team_batting_df_bbb,
                        selected_team=chosen_team,
                        color_map=phase_color_map,
                        secondary_color=secondary_color,
                        innings=[2],
                        bar_color = primary_color,
                        title = 'Partnership Runs by wickets over season in 2nd Innings'
                    ),
                    width = 'stretch',
                    key = 'partnership_plot_2'
                )


        with st.container(
            # border = True,
            key = 'run-rates'
        ):
            coll1,coll2 = resizable_columns(2)
            
            with coll1:
                st.plotly_chart(
                    plot_team_run_rate_comparison(
                        run_rate_df,
                        selected_team=chosen_team,
                        innings=[1],
                        highlight_color= primary_color,
                        # grayscale_color=secondary_color,
                        secondary_color=secondary_color,
                        color_map_df=team_color_df, # type: ignore
                        title= f"Comparison of Average Run Rate maintained at end of each over for 1st Innings"
                    ),
                    width = 'stretch',
                    key = 'run_rate1'
                )

            with coll2:
                st.plotly_chart(
                    plot_team_run_rate_comparison(
                        run_rate_df,
                        selected_team=chosen_team,
                        innings=[2],
                        color_map_df=team_color_df, # type: ignore
                        secondary_color=secondary_color,
                        # grayscale_color=secondary_color,
                        highlight_color= primary_color,
                        title= f"Comparison of Average Run Rate maintained at end of each over for 2nd Innings"
                    ),
                    width = 'stretch',
                    key = 'run_rate2'
                )    



        # RAA vs dismissal Rate scatter chart graph 
        
        impact_score_df = get_batter_pca_impact(
            # phase_wise_batsman_stats_df.loc[
            #     # (overall_batsman_stats_df['team_bat'].eq(chosen_team)) &
            #     (phase_wise_batsman_stats_df['balls'] >= 50)
            # ].reset_index(drop = True),
            get_batting_profile(
                master_df.loc[
                    (master_df['year'].eq(season)&
                     (master_df['phase'].isin(phase_selected))),
                ],
                ['team_bat', 'bat']
            ),
            feature_cols = [
                'bat_overall_sr',
                'runs_team_runs_ratio',
                'team_run_rate_diff',
                'bat_overall_avg',
                'dot_ball_pct',
                'boundary_pct'
            ]
        )

        st.plotly_chart(
            plot_raa_vs_dismissal_rate(
                impact_score_df.loc[
                    # (overall_batsman_stats_df['team_bat'].eq(chosen_team)) &
                    (impact_score_df['balls'] >= 50)
                ].reset_index(drop = True),
                chosen_team=chosen_team,
                highlight_color=primary_color,

                title = f"Impact Score vs RAA per 100 Balls"
            ),
            width = 'stretch',
            key = 'raa_vs_dismissal'
        )




    with tab2:

        bowling_stat_kpis_df = pd.DataFrame({
            'kpis' : [
                'wickets', 
                'dot_balls', 
                'overall_eco', 
                'bowl_overall_sr', 
                # 'bowl_overall_avg'
            ],
            'alias' : [
                'Most Wickets', 
                'Highest Dot Balls', 
                'Best Economy', 
                'Best Strike Rate', 
                # 'Best Average'
            ],

            'agg_func' : [
                'max', 
                'max', 
                'min', 
                'min', 
                # 'min'
            ],
            'sort_val' : [
                False, 
                False, 
                True, 
                True, 
                # True
            ]
        })

        temp_overall_bowl_df = overall_bowler_stats_df.loc[
            (overall_bowler_stats_df['team_bowl'].eq(chosen_team)) &
            (overall_bowler_stats_df['balls'] >= 100)
        ].reset_index(drop = True)

        cols1 = resizable_columns(len(bowling_stat_kpis_df))
        for i, row in bowling_stat_kpis_df.iterrows():

            with cols1[i]: # type: ignore
                st.markdown(
                        f"""
                        <div style="text-align: center;
                        font-size : 20px;">
                            <strong>Season - {row['alias']}</strong>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )            
                st.metric(
                    # label = row['alias'],
                    label = temp_overall_bowl_df.loc[
                        temp_overall_bowl_df[row['kpis']] == temp_overall_bowl_df[row['kpis']].agg(row['agg_func']),
                        'bowl'
                    ].iloc[0],# type: ignore
                    value = temp_overall_bowl_df[row['kpis']].agg(row['agg_func']), # type: ignore
                    # delta =  - 0, # type: ignore

                    # delta_color = 'off',
                    # delta_arrow = 'off'
                )
        
        # for i,row in bowling_stat_kpis_df.iterrows():
                st.markdown(
                    f"""
                    <div style="
                    text-align: center;
                    font-size : 20px;">
                        <strong>Phase - {row['alias']}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                phase_wise_bowler_stats_df = get_bowling_profile(
                    team_bowling_df_bbb[
                        team_bowling_df_bbb['phase'].isin(phase_selected)
                    ],
                    ['team_bowl', 'bowl']
                )

                st.dataframe(
                    phase_wise_bowler_stats_df[
                            ['bowl', row['kpis']]
                        ].sort_values(
                            by = row['kpis'], 
                            ascending = row['sort_val']
                        )\
                        .rename(
                            columns = {
                                'bowl' : "Bowler Name",
                                row['kpis'] : row['alias']
                            }
                        ).head(5),
                    hide_index=True
                ) 

        
        # Bowling KPIs : boundary conc rate, dots %, economy rate, strike rate
        bowling_kpis_delta = pd.DataFrame({
            'kpis' : ['balls_per_boundary_conc', 'dot_ball_pct', 'overall_eco', 'bowl_overall_sr'],
            'alias' : ['Balls per boundary conceded', 'Dot Ball %', 'Economy Rate', 'Strike Rate'],
            'delta_value' : ['normal', 'normal', 'inverse', 'inverse']
        })

        cols_bow = resizable_columns(len(bowling_kpis_delta))

        
        team_bowling_df = get_bowling_profile(
            team_bowling_df_bbb.loc[
                (team_bowling_df_bbb["phase"].isin(phase_selected))
            ],
            ['team_bowl']
        )

        for i, row in bowling_kpis_delta.iterrows():
            kpi = row['kpis']
            label = row['alias']

            with cols_bow[i]: # type: ignore


        
                team_value = team_bowling_df[kpi].iloc[0] # type: ignore

                all_teams_bowling_df = get_bowling_profile(
                        master_df.loc[
                        (master_df["phase"].isin(phase_selected)) &
                        (master_df['year'].isin([season])),
                       ],
                        ['phase']
                    )
                
                phase_avg = all_teams_bowling_df[kpi].iloc[0]

                st.metric(
                    label=label,
                    value=round(team_value, 2),
                    delta=round(team_value - phase_avg, 2),
                    delta_color = row['delta_value'],
                    width = 'stretch',
                )




        st.subheader(f'Overwise False Shot % Induced by {chosen_team} (Number to Right of bars - Wickets Taken)')

        col1, col2 = resizable_columns(2)
        with col1:

            st.plotly_chart(
                plot_false_shots_by_over(
                    team_bowling_df_bbb,
                    bar_color=primary_color,
                    color_map=phase_color_map,
                    secondary_color=secondary_color,
                    innings=[1],
                    # title = f'False Shots Induced by {chosen_team} overwise through 1st Innings (Showing runs conceded)'
                    title = f'1st Innings'
                ),
                width = 'stretch',
                key = 'false_shot_plot_1'
            )


        with col2:
            st.plotly_chart(
                plot_false_shots_by_over(
                    team_bowling_df_bbb,
                    innings=[2],
                    bar_color=primary_color,
                    secondary_color=secondary_color,
                    color_map=phase_color_map,
                    # title = f'False Shots Induced by {chosen_team} overwise through 2nd Innings (Showing runs conceded)'
                    title = '2nd Innings'
                ),
                width = 'stretch',
                key = 'false_shot_plot_2'
            )


    ############# 2 separate functions for false shot induced & Bowling metrics per over. Start from here
        _,_,_col3 = resizable_columns([2,2,1])
        with _col3:
            metric = st.selectbox(
                'Select Bowling Metric',
                options = [
                    'Bowling Average',
                    'Economy Rate',
                    'Bowling Strike-Rate',
                    'Dot Ball %'
                ],

                index = 0,
                key = 'bowling_metric_selectbox'
            )

        col1, col2 = resizable_columns(2)
        with col1:

            st.plotly_chart(
                plot_economy_rate_comparison(
                    master_df,
                    selected_team=chosen_team, # type: ignore
                    season = [season],
                    highlight_color=primary_color,
                    secondary_color=secondary_color,
                    innings=[1],
                    metric=metric,
                    title = f'Overwise {metric} by {chosen_team} through 1st Innings'
                ),
                width = 'stretch',
                key = 'eco_rate_1'
            )

        with col2:

            st.plotly_chart(
                plot_economy_rate_comparison(
                    master_df,
                    selected_team=chosen_team, # type: ignore
                    season = [season],
                    highlight_color=primary_color,
                    secondary_color=secondary_color,
                    metric=metric,
                    innings=[2],
                    title = f'Overwise {metric} by {chosen_team} through 2nd Innings'
                ),
                width = 'stretch',
                key = 'eco_rate2'
            )    

