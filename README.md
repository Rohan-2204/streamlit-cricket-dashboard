IPL Ball-by-Ball Cricket Analytics Dashboard
An advanced, interactive sports analytics application built with Streamlit to analyze Indian Premier League (IPL) ball-by-ball datasets, evaluate team and player performances, and provide tactical insights.

Features and Pages
Home: Interactive team selection interface with customized color branding and session state initialization.

Team Analysis: Comprehensive view of team batting and bowling statistics, points table tracking, phase-wise breakdowns, partnership progressions, run-rate comparisons, and impact analysis.

Batsman Profile: Detailed performance metrics, ball-speed and release-height vulnerability tables, on-pitch heatmaps (strike rate, average, dot ball percentage, false shots), and interactive wagon wheels filtered by line, length, shot type, and outcome.

Bowler Profile: Comprehensive bowler evaluation including economy, strike rate, dot ball percentage, false shots induced by over, release height and ball speed metrics, and delivery distribution heatmaps.

Project Structure
Plaintext
├── app.py                  # Main entry point and Streamlit page navigation
├── reports/
│   ├── home.py             # Landing page and session state setup
│   ├── team_analysis.py    # Team-level performance and tactical reports
│   ├── batter_profile.py   # Individual batsman profiling and heatmaps
│   └── bowler_profile.py   # Individual bowler profiling and delivery analysis
├── features/
│   └── feature_addition.py # Feature engineering module for raw match data
├── assets/
│   ├── main.css            # Custom styling definitions
│   └── logos/              # Team logo image assets
├── data/
│   └── raw/                # Raw datasets (master_df.csv, ground_angles.csv)
└── requirements.txt        # Project dependencies and pinned versions
Installation and Setup
Clone the repository and navigate to the project directory.

Install the required dependencies listed in requirements.txt:

Bash
pip install -r requirements.txt
Run the Streamlit application:

Bash
streamlit run app.py
Dependencies
Key Python libraries and packages utilized in this project include:

Streamlit and Streamlit Extras

Pandas and NumPy

Plotly and Matplotlib / Seaborn

Great Tables

Scikit-Learn and LightGBM
