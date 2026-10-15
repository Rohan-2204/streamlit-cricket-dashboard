import streamlit as st

st.set_page_config(page_title="IPL BBB Dataset Analysis", layout="wide")


page_names = [
    st.Page('reports/home.py', title = "Home"),
    st.Page('reports/team_analysis.py', title = "Team Analysis"),
    st.Page('reports/batter_profile.py', title = "Batsman Profile"),
    st.Page('reports/bowler_profile.py', title =  "Bowler Profile"),
    # st.Page('reports/match_ups.py', title = "Match-ups")
]


pg = st.navigation(page_names)
pg.run()


