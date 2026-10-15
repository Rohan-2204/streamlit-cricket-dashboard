import os
import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from utils.utils import *
from streamlit_image_select import image_select 
from streamlit_extras.card_selector import card_selector
from features.feature_addition import generate_features

st.set_page_config(
    "Home",
    page_icon = "🏏",
    initial_sidebar_state= 'expanded',
    layout='wide'
)


def load_css(css_file: str = "main.css"):
    with open(css_file, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


load_css('./assets/main.css')

ground_angles_df = pd.read_csv('data/raw/ground_angles.csv')
master_df = generate_features(
    pd.read_csv('data/raw/master_df.csv')
)

if 'initialized' not in st.session_state:

    st.session_state.update({
        'initialized' : True,
        "master_df" : master_df.copy(),
        "home_page" : True,
        'selected_team' : None,
        'season' : None,
        'ground_angles_df' : ground_angles_df.copy(),
        'bowling_type_map' : {
                'spin' : [
                        'Left Arm Wrist Spin',
                        'Right Arm Leg Spin',
                        'Left Arm Orthodox',
                        'Right Arm Off Spin',
                    ],
                'pace' : [
                        'Right Arm Fast Medium',
                        'Left Arm Medium',
                        'Left Arm Fast',
                        'Left Arm Fast Medium',
                        'Right Arm Medium',
                        'Right Arm Fast',
                    ]
            },
        'team_color_df' : pd.DataFrame({
                    'team_name' : [
                        'Chennai Super Kings', 
                        'Royal Challengers Bengaluru', 
                        'Sunrisers Hyderabad', 
                        'Kolkata Knight Riders', 
                        'Mumbai Indians', 
                        'Punjab Kings', 
                        "Delhi Capitals", 
                        'Lucknow Super Giants', 
                        'Gujarat Titans', 
                        'Rajasthan Royals'
                    ],
                    'primary_color' : [
                        '#FCCA06',  # CSK - Yellow
                        '#7c110b',  # RCB - Red
                        '#F26522',  # SRH - Orange
                        '#1b0b2f',  # KKR - Purple
                        '#004BA0',  # MI  - Blue
                        '#DD1D25',  # PBKS - Red
                        '#123674',  # DC  - Dark Blue/Navy
                        '#b3171a',  # LSG - Maroon
                        '#1B2133',  # GT  - Dark Navy
                        '#EA1A85'   # RR  - Pink
                    ],
                    'secondary_color': [
                        '#005CA8',  # CSK - Blue
                        '#212845',  # RCB - Dark navy Blue
                        '#000000',  # SRH - Black
                        '#d3ad5a',  # KKR - Gold
                        '#D4AF37',  # MI  - Gold
                        '#33619d',  # PBKS - Silver
                        '#ba110c',  # DC  - Red
                        '#281859',  # LSG - Red/Orange
                        '#E3B34C',  # GT  - Gold
                        '#29284a'   # RR  - Navy Blue
                    ],
                    'font_color': [
                        '#000000',  # CSK - Black text (best readability on yellow)
                        '#FFFFFF',  # RCB - White text
                        '#FFFFFF',  # SRH - White text
                        '#FFFFFF',  # KKR - White text
                        '#FFFFFF',  # MI  - White text
                        '#FFFFFF',  # PBKS - White text
                        '#FFFFFF',  # DC  - White text
                        '#FFFFFF',  # LSG - White text
                        '#FFFFFF',  # GT  - White text
                        '#000000'   # RR  - Black text
                    ]
                }
            ),
            'pace_cat_dict' : {
                1 : '> 140 kph',
                2 : '130 - 140 kph',
                3 : '120 - 130 kph',
                4 : '110 - 120 kph',
                5 : '100 - 110 kph',
                6 : '85 - 100 kph',
                7 : '70 - 85 kph',
                8 : '< 70kph'
            },
            'release_ht_dict' : {
                1 : '> 2m',
                2 : '1.8 - 2m',
                3 : '1.6 - 1.8m',
                4 : '< 1.6m'
            }
        }
    )


team_names = [t.lower() for t in list(master_df['team_bat'].dropna().unique())]


if st.session_state.home_page:


    st.title("Tactics-helper Dashboard")
    with st.container(border = True):

        st.success("Select your team to make the decisions for. \n The dashboard will choose this as your team and provide analysis accordingly. \n You can change your team in the Team analysis Page.")
        mapping_name = auto_map_and_rename_logos()

        assert mapping_name is not None

        selected_team = image_select( 
            label=" ", 
            images= [f"assets/logos/{team}.png" for k, team in mapping_name.items()], 
            captions = [t.title() for k, t in mapping_name.items()], 
            return_value = 'original', 
            use_container_width = False 
        )

        if st.button(
            'Confirm Team',
            type = 'primary',
            key = 'team_confirmation_btn'
        ):
            st.session_state.selected_team = os.path.splitext(os.path.basename(selected_team))[0].title()
            st.session_state.home_page = False
            st.switch_page("reports/team_analysis.py")



else:

    st.info("Please refresh the page. If it doesn't work, please restart the app from the terminal")



