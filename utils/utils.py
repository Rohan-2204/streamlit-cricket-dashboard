import difflib
from pathlib import Path
import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb, to_hex
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plotly.graph_objects as go
import plotly.express as px


def slugify(text: str) -> str:
    """Normalizes string by removing spaces and special characters for comparison."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


def auto_map_and_rename_logos(
    logos_dir_path: str = "assets/logos", 
    data_path: str = "data/raw/master_df.csv"
):
    """Automatically maps and renames logo files to match team names in the dataset."""
    logos_dir = Path(logos_dir_path)

    if not logos_dir.exists():
        print(f"Error: Directory '{logos_dir}' does not exist.")
        return

    if Path(data_path).exists():
        df = pd.read_csv(data_path)
        teams = sorted(
            list(
                set(
                    df["team_bat"].dropna().unique().tolist()
                    + df["team_bowl"].dropna().unique().tolist()
                )
            )
        )
    else:
        print(
            f"Warning: Dataset not found at {data_path}. Using standard IPL team"
            " names list."
        )
        teams = [
            "Chennai Super Kings",
            "Royal Challengers Bengaluru",
            "Sunrisers Hyderabad",
            "Kolkata Knight Riders",
            "Mumbai Indians",
            "Punjab Kings",
            "Delhi Capitals",
            "Lucknow Super Giants",
            "Gujarat Titans",
            "Rajasthan Royals",
        ]

    print(f"Found {len(teams)} unique teams in dataset.")

    logo_files = list(logos_dir.glob("*.[pP][nN][gG]")) + list(
        logos_dir.glob("*.[jJ][pP][gG]")
    )

    if not logo_files:
        print(f"No image files found in {logos_dir}.")
        return

    mapping = {}

    for file_path in logo_files:
        original_name = file_path.name
        stem = file_path.stem
        clean_stem = slugify(stem)

        best_match = None
        highest_score = 0.0

        for team in teams:
            team_slug = slugify(team)

            ratio = difflib.SequenceMatcher(None, clean_stem, team_slug).ratio()

            if clean_stem in team_slug or team_slug in clean_stem:
                ratio += 0.5

            if ratio > highest_score:
                highest_score = ratio
                best_match = team

        if best_match:
            safe_team_name = re.sub(r'[\\/*?:"<>|]', "", best_match)
            new_filename = f"{safe_team_name}{file_path.suffix}"
            new_path = logos_dir / new_filename

            mapping[original_name] = safe_team_name

            if file_path.resolve() != new_path.resolve():
                file_path.rename(new_path)

    return mapping


def lighten_color(hex_color, amount):
    """Lightens a given hex color by a specified amount."""
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = round(r * (1 - amount))
    g = round(g * (1 - amount))
    b = round(b * (1 - amount))
    return f"#{r:02X}{g:02X}{b:02X}"


def get_unique_teams(df: pd.DataFrame, exclude_team: str) -> list:
    """Returns a list of rival teams excluding the selected team."""
    return [
        team for team in df["team_bat"].dropna().unique()
        if team != exclude_team
    ]


def get_batsmen_by_team(df: pd.DataFrame, team_name: str) -> list:
    """Retrieves unique batsmen for a given team."""
    return df.loc[
        df["team_bat"] == team_name, "striker"
    ].dropna().unique().tolist()


def get_bowlers_by_team(df: pd.DataFrame, team_name: str) -> list:
    """Retrieves unique bowlers for a given team."""
    return df.loc[
        df["team_bowl"] == team_name, "bowler"
    ].dropna().unique().tolist()


def get_bowling_styles_by_team(df: pd.DataFrame, team_name: str) -> list:
    """Retrieves unique bowling styles for a given team."""
    return df.loc[
        df["team_bowl"] == team_name, "bowling_style"
    ].dropna().unique().tolist()


def get_points_table(master_df):
    """Calculates and returns the points table for each season from match data."""
    df = master_df.copy()

    innings = (
        df.groupby(['year', 'p_match', 'inns', 'team_bat', 'team_bowl'])
        .agg(
            runs=('score', 'sum'),
            balls=('inns_balls', 'max'),
            wickets=('out', 'sum'),
            max_balls=('max_balls', 'max')
        )
        .reset_index()
    )

    innings['balls'] = innings['balls'].where(
        innings['wickets'] < 10, innings['max_balls']
    )

    matches = (
        df.groupby(['year', 'p_match'])
        .agg(
            team1=('team_bat', 'first'),
            team2=('team_bowl', 'first'),
            winner=('winner', 'first')
        )
        .reset_index()
    )

    teams = pd.concat([
        matches.rename(columns={'team1': 'team'})[['year', 'p_match', 'team', 'winner']],
        matches.rename(columns={'team2': 'team'})[['year', 'p_match', 'team', 'winner']]
    ], ignore_index=True)

    teams['wins'] = (teams['team'] == teams['winner']).astype(int)
    teams['NR'] = teams['winner'].isna() | teams['winner'].eq('-')
    teams['losses'] = (
        (teams['wins'] == 0) & ~teams['NR']
    ).astype(int)

    table = (
        teams.groupby(['year', 'team'])
        .agg(
            wins=('wins', 'sum'),
            losses=('losses', 'sum'),
            NR=('NR', 'sum')
        )
        .reset_index()
    )

    batting = innings.groupby(['year', 'team_bat']).agg(
        runs_for=('runs', 'sum'),
        balls_faced=('balls', 'sum')
    )

    bowling = innings.groupby(['year', 'team_bowl']).agg(
        runs_against=('runs', 'sum'),
        balls_bowled=('balls', 'sum')
    )

    table = (
        table
        .merge(batting, left_on=['year', 'team'], right_index=True)
        .merge(bowling, left_on=['year', 'team'], right_index=True)
    )

    table['points'] = 2 * table['wins'] + table['NR']

    table['NRR'] = (
        6 * table['runs_for'] / table['balls_faced']
        - 6 * table['runs_against'] / table['balls_bowled']
    ).round(3)

    table = table.sort_values(
        ['year', 'points', 'NRR'],
        ascending=[True, False, False]
    )

    table['position'] = table.groupby('year').cumcount() + 1

    return (
        table[['year', 'position', 'team',  'wins', 'losses', 'NR', 'points', 'NRR']].rename(
            columns={
                'year': 'Season',
                'position' : "Position",
                'team' : 'Team',
                'wins' : 'Wins',
                'losses' : "Losses",
                'NR' : "No-Result",
                'points' : 'Points',
                'NRR' : 'Net-Run-Rate'
            }
        )
        .reset_index(drop=True)
    )


def get_body_weight_metrics(df, group_cols):
    """Computes aggregated delivery metrics grouped by specified columns."""
    group_df = (
        df
        .groupby(group_cols)
        .agg(
            number_of_balls=('ball', 'count'),
            total_runs=('batruns', 'sum'),
            wickets=('out', 'sum'),
            false_shots=('control', 'sum')
        )
        .reset_index()
    )

    group_df['False Shot %'] = round(group_df['false_shots'] * 100 / group_df['number_of_balls'], 2)
    group_df = group_df.rename(columns={
        'number_of_balls': 'Number of Deliveries',
        'total_runs': 'Runs Scored',
        'wickets': 'Wickets Lost',
    }).drop(columns = ['false_shots'])

    return group_df


def get_plain_pitch_stumps_markings_figure(
    pitch_coords,
    x_coord = 'x_coord',
    y_coord = 'y_coord',
    fig_height = 700,
    fig_width = 100,
    pitch_color = "rgba(240, 230, 140, 0.2)"
):
    """Generates a plain figure representing pitch markings and stumps layout."""
    df = pitch_coords.copy()

    pitch_edges = df[df['point_description'] == 'edge'].reset_index(drop = True)
    stumps = df[df['point_description'].isin(['stumps'])].reset_index(drop = True)
    marking_lines = df[df['point_description'].isin(['markings'])].reset_index(drop = True)

    pitch_x_max = pitch_edges[x_coord].max()
    pitch_y_max = pitch_edges[y_coord].max()

    fig = go.Figure()

    fig.add_shape(
        type="rect",
        x0=0, 
        y0=0, 
        x1= pitch_x_max, 
        y1= pitch_y_max,
        line=dict(color="#d5ab6f", width=0.5),
        fillcolor = pitch_color,
        layer = 'below'
    )

    fig.add_trace(
        go.Scatter(
            x = stumps[x_coord],
            y = stumps[y_coord],
            mode='markers',
            marker = dict(color = 'black', size = 7),
            zorder=10
        )
    )

    fig.add_trace(
        go.Scatter(
            x = marking_lines[x_coord],
            y = marking_lines[y_coord],
            mode='markers',
            marker = dict(color = '#1c1c84', size = 8),
            zorder=10
        )
    )

    fig.update_layout(
        xaxis=dict(range=[-20, pitch_x_max + 20], showgrid=False, zeroline=False),
        yaxis=dict(range=[-50, pitch_y_max + 50], showgrid=False, zeroline=False),
        width = fig_width,
        height = fig_height,
        showlegend=False,
        plot_bgcolor='white'
    )

    return fig


def get_stumps_line_on_pitch(
    fig,
    pitch_coords,
    stumps_lines_needed = ['off stump', 'leg stump'],
):
    """Adds stumps lines to the pitch figure."""
    off_leg_line = pitch_coords[
        (pitch_coords['point_name'].isin(['off stump', 'leg stump']))
        &
        (pitch_coords['point_description'] == 'stumps')
    ].groupby(['point_description', 'point_name'])

    for (_, _), group in off_leg_line:
        fig.add_trace(
            go.Scatter(
                x = group['x_coord'],
                y = group['y_coord'],
                mode = 'lines',
                line = dict(color = 'black', width = 1.5, dash = 'dot')
            )
        )

    return fig


def get_lines_lengths_on_pitch(
    pitch_fig,
    pitch_coords,
    grouping_cols = ['point_name', 'end', 'bowling_type'],
    x_coord = 'x_coord',
    y_coord = 'y_coord',
    color = ''
):
    """Adds pitch lines and lengths markers to the figure."""
    df = pitch_coords.copy()
    grouped = df.groupby(grouping_cols)

    for (_, name, end), group in grouped:
        group = group.sort_values(by=[x_coord, y_coord])
        if len(group) >= 2:
            pitch_fig.add_trace(
                go.Scatter(
                    x=group[x_coord],
                    y=group[y_coord],
                    mode='lines',
                    name=f"{name}",
                    line=dict(color = color, width=1.5, dash='dot'),
                    hoverinfo='name'
                )
            )

    return pitch_fig


def add_pitch_length_annotations(fig, pitch_coords, bowling_type='pace'):
    """Adds pitch length text annotations to the figure."""
    length_coords = pitch_coords[
        (pitch_coords['point_description'] == 'length') & 
        ((pitch_coords['bowling_type'] == bowling_type) | (pitch_coords['bowling_type'].isna()))
    ]

    length_midpoints = length_coords.groupby('point_name')['y_coord'].mean().to_dict()
    min_x = pitch_coords['x_coord'].min() if 'x_coord' in pitch_coords.columns else 0

    for length_name, y_pos in length_midpoints.items():
        label_text = length_name.replace('_', ' ').title()
        
        fig.add_annotation(
            x=min_x + 180,
            y=y_pos,
            text=f"<b>{label_text}</b>",
            showarrow=False,
            xanchor="center",
            yanchor="middle",
            font=dict(size=13, color="#333333"),
            bgcolor="rgba(255, 255, 255, 0.2)",
            align="right"
        )

    return fig


def add_pitch_side_annotations(fig, pitch_coords, batting_style='right_handed'):
    """Adds off-side and leg-side side annotations based on batting style."""
    min_x = pitch_coords['x_coord'].min() if 'x_coord' in pitch_coords.columns else -2
    max_x = pitch_coords['x_coord'].max() if 'x_coord' in pitch_coords.columns else 2
    max_y = pitch_coords['y_coord'].max() if 'y_coord' in pitch_coords.columns else 22

    is_right_handed = 'right' in batting_style.lower()
    
    left_side_label = "<b>OFF SIDE</b>" if is_right_handed else "<b>LEG SIDE</b>"
    right_side_label = "<b>LEG SIDE</b>" if is_right_handed else "<b>OFF SIDE</b>"

    fig.add_annotation(
        x=min_x + (max_x - min_x) * 0.25,
        y=max_y + 0.5,
        text=left_side_label,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=12, color="#1F77B4")
    )

    fig.add_annotation(
        x=min_x + (max_x - min_x) * 0.75,
        y=max_y + 0.5,
        text=right_side_label,
        showarrow=False,
        xanchor="center",
        yanchor="bottom",
        font=dict(size=12, color="#1F77B4")
    )

    return fig


def calculate_bowler_runs_by_direction(
        master_df, 
        ground_angles_df, 
        player_in_question = 'bowler',
        player_name = 'Shami'
    ):
    """Calculates runs conceded or scored grouped by shot direction sectors."""
    if player_in_question == 'bowler':
        player_df = master_df[master_df['bowler'] == player_name].copy()
    else:
        player_df = master_df[master_df['striker'] == player_name].copy()

    if player_df.empty:
        raise ValueError(f"No delivery data found for {player_in_question}: {player_name}")

    agg_dict = {
        'runs_conceded': ('batruns', 'sum'),
        'balls_bowled': ('batruns', 'count'),
        '4s_conceded': ('batruns', lambda x: (x == 4).sum()),
        '6s_conceded': ('batruns', lambda x: (x == 6).sum()),
        'wickets': ('out', 'sum')
    }
    direction_stats = player_df.groupby('direction').agg(**agg_dict).reset_index()

    merged_df = pd.merge(
        ground_angles_df,
        direction_stats,
        left_on='shot_direction',
        right_on='direction',
        how='left'
    )

    merged_df['runs_conceded'] = merged_df['runs_conceded'].fillna(0).astype(int)
    merged_df['balls_bowled'] = merged_df['balls_bowled'].fillna(0).astype(int)
    merged_df['4s_conceded'] = merged_df['4s_conceded'].fillna(0).astype(int)
    merged_df['6s_conceded'] = merged_df['6s_conceded'].fillna(0).astype(int)
    merged_df['wickets'] = merged_df['wickets'].fillna(0).astype(int)

    sector_summary = merged_df.groupby(['general_direction', 'start_angle', 'end_angle']).agg(
        total_runs=('runs_conceded', 'sum'),
        total_balls=('balls_bowled', 'sum'),
        total_4s=('4s_conceded', 'sum'),
        total_6s=('6s_conceded', 'sum'),
        total_wickets=('wickets', 'sum')
    ).reset_index().sort_values(by = 'start_angle', ascending = False)

    return merged_df, sector_summary


def get_sector_color(pct):
    """Returns color coding based on run share percentage."""
    if pct < 5:
        return "#D9F99D"
    elif pct < 10:
        return "#84CC16"
    elif pct < 20:
        return "#22C55E"
    else:
        return "#166534"


def plot_player_field_partitions_with_runs(
    ground_angles_df,
    master_df,
    player_name,
    player_in_question='bowler',
    batting_style='',
    radius = 66.0,
    striker_stumps_y = 10.06,
    half_pitch_width = 1.83,
    fig_height = 700,
    fig_width = 700,
    line_width=1.5,
    outfield_color="#000000",
    pitch_color="#c9a777"
):
    """Generates a field partition chart displaying run distribution across sectors."""
    if isinstance(batting_style, list) and len(batting_style) > 1:
        filter_value = 'right_handed'
    else:
        style_str = batting_style[0] if isinstance(batting_style, list) else batting_style
        filter_value = style_str.lower().strip().replace(" ", "_")

    if batting_style is not None and batting_style != '':
        ground_angles_df = ground_angles_df[
            ground_angles_df['batting_style'] == filter_value
        ].reset_index(drop=True)

    _, sector_summary = calculate_bowler_runs_by_direction(
        master_df, 
        ground_angles_df, 
        player_in_question=player_in_question,
        player_name=player_name
    )

    ground_fig = get_plain_ground_and_pitch_figure(
            radius = radius,
            pitch_x_half = half_pitch_width,
            pitch_y_half = striker_stumps_y,
            fig_height = fig_height,
            fig_width = fig_width,
            outfield_color=outfield_color,
            pitch_color = pitch_color
        )

    df = ground_angles_df.copy()
    all_angles = pd.concat([df['start_angle'], df['end_angle']]).unique()

    for angle in all_angles:
        angle_deg = float(angle)
        rad = np.deg2rad(angle_deg + 90)
        
        boundary_x = radius * np.cos(rad)
        boundary_y = radius * np.sin(rad)
        
        ground_fig.add_trace(go.Scatter(
            x=[0, boundary_x],
            y=[striker_stumps_y, boundary_y],
            mode='lines',
            line=dict(color='black', width=line_width, dash='dashdot'),
            hoverinfo='none',
            zorder = 100
        ))

    total_runs_all_sectors = sector_summary['total_runs'].sum()
    sector_summary['runs_pct'] = sector_summary['total_runs'] * 100 / total_runs_all_sectors if total_runs_all_sectors > 0 else 0.0
    sector_summary['badge_color'] = sector_summary['runs_pct'].apply(get_sector_color)

    for _, row in sector_summary.iterrows():
        start = float(row['start_angle'])
        end = float(row['end_angle'])

        if start > end:
            angles = np.linspace(start, end + 360, 200)
        else:
            angles = np.linspace(start, end, 200)
        angles = angles % 360
        rads = np.deg2rad(angles + 90)

        arc_x = radius * np.cos(rads)
        arc_y = (radius * np.sin(rads))
        
        wedge_x = [0] + list(arc_x) + [0]
        wedge_y = [striker_stumps_y] + list(arc_y) + [striker_stumps_y]

        ground_fig.add_trace(go.Scatter(
            x=wedge_x,
            y=wedge_y,
            fill='toself',
            fillcolor=row['badge_color'],
            line=dict(color='rgba(0,0,0,0)'), 
            hoverinfo='text',
            hovertext=f"{row['general_direction'].upper()}<br>Runs: {row['total_runs']} ({row['runs_pct']:.1f}%)",
            zorder=0
        ))

        if start > end:
            mid_angle = ((start + (end + 360)) / 2) % 360
        else:
            mid_angle = (start + end) / 2

        label_radius = radius * 1.2
        rad = np.deg2rad(mid_angle + 90)
        label_x = label_radius * np.cos(rad)
        label_y = (label_radius * np.sin(rad))

        sector_runs = row['total_runs']

        disp_text = f'''<b>{sector_runs} ({row['total_balls']})</b><br>4s-{row['total_4s']}<br>6s-{row['total_6s']}'''

        ground_fig.add_annotation(
            x=label_x,
            y=label_y,
            text=disp_text,
            showarrow=False,
            font=dict(size=15, color="white"),
            borderpad=0.1,
        )

    legend_items = [
        ("< 5%", "#D9F99D", "Run Share"),
        ("5-10%", "#84CC16", "Run Share"),
        ("10-20%", "#22C55E", "Run Share"),
        ("≥ 20%", "#166534", "Run Share"),
    ]

    legend_x = 0.82
    legend_y = 0.98
    line_spacing = 0.055

    for i, (pct_range, color, description) in enumerate(legend_items):
        y = legend_y - i * line_spacing
        ground_fig.add_shape(
            type="rect",
            xref="paper", yref="paper",
            x0=legend_x, x1=legend_x + 0.025,
            y0=y - 0.018, y1=y + 0.012,
            fillcolor=color,
            line=dict(width=0),
        )

        ground_fig.add_annotation(
            xref="paper", yref="paper",
            x=legend_x + 0.035, y=y - 0.003,
            text=f"<b>{pct_range}</b> — {description}",
            showarrow=False,
            xanchor="left", yanchor="middle",
            font=dict(size=15, color="white"),
        )

    annotations_df = pd.DataFrame({
        'batting_style': ['right_handed', 'left_handed', 'right_handed', 'left_handed'],
        'label' : ['Off Side', "Off Side", 'Leg Side', 'Leg Side'],
        'x' : [0.15, 0.85, 0.85, 0.15],
        'y' : [0.55, 0.55, 0.55, 0.55],
    })

    for _, rows in annotations_df[annotations_df['batting_style'] == filter_value].iterrows():
        ground_fig.add_annotation(
            xref="paper", yref="paper",
            x=rows['x'], y=rows['y'],
            text=rows['label'],
            showarrow=False,
            align="center",
            font=dict(size=14, color='black'),
            bgcolor="rgba(240, 240, 240, 0.8)",
            bordercolor="#cccccc",
            borderwidth=1,
            borderpad=6
        )

    return ground_fig


def get_plain_ground_and_pitch_figure(
    image_size=200,
    max_coord =363,
    min_coord = 0,
    pitch_x_half=2.75,
    pitch_y_half=15.1,
    display_size=700,
    outfield_color="#2ca02c",
    pitch_color="#c9a777"
):
    """Creates a base figure representing a cricket ground and pitch layout."""
    fig = go.Figure()

    center = (max_coord - min_coord)/ 2

    radius = 2 * center

    theta = np.linspace(0, 2 * np.pi, 300)

    ground_x = center + radius * np.cos(theta)
    ground_y = center + radius * np.sin(theta)

    fig.add_trace(go.Scatter(
        x=ground_x,
        y=ground_y,
        mode='lines',
        line=dict(
            color=outfield_color,
            width=3
        ),
        fill='toself',
        hoverinfo='none',
        name='Boundary Rope'
    ))

    fig.add_shape(
        type="rect",
        x0=center - pitch_x_half,
        y0=center - pitch_y_half,
        x1=center + pitch_x_half,
        y1=center + pitch_y_half,
        line=dict(
            color=pitch_color,
            width=10
        ),
        fillcolor=pitch_color
    )

    fig.update_layout(
        width=display_size,
        height=display_size,
        showlegend=False,

        xaxis=dict(
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True
        ),

        yaxis=dict(
            range=[image_size, -200],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
            scaleratio=1,
            fixedrange=True
        ),

        margin=dict(
            l=100,
            r=0,
            t=100,
            b=0
        )
    )

    return fig


def plot_player_wagon_wheel(
    master_df,
    player_name,
    max_coord =363,
    min_coord = 0,
    player_in_question='bowler',
    image_size=1000,
    display_size=700,
    outfield_color="#2ca02c",
    pitch_color="#c9a777"
):
    """Generates a wagon wheel scatter plot for a selected batsman or bowler."""
    if player_in_question == 'bowler':
        player_df = master_df[
            master_df['bowl'] == player_name
        ].copy()
    else:
        player_df = master_df[
            master_df['bat'] == player_name
        ].copy()

    if player_df.empty:
        raise ValueError(
            f"No delivery data found for {player_in_question}: {player_name}"
        )

    player_df['wagonx'] = pd.to_numeric(
        player_df['wagonx'],
        errors='coerce'
    )

    player_df['wagony'] = pd.to_numeric(
        player_df['wagony'],
        errors='coerce'
    )

    player_df = player_df.dropna(
        subset=['wagonx', 'wagony']
    )

    fig = get_plain_ground_and_pitch_figure(
        image_size=image_size,
        display_size=display_size,
        outfield_color=outfield_color,
        max_coord =max_coord,
        min_coord = min_coord,
        pitch_color=pitch_color
    )

    fig.add_trace(go.Scatter(
        x=player_df['wagonx'],
        y=player_df['wagony'],
        mode='markers',

        marker=dict(
            size=7,
            opacity=0.75,
            line=dict(
                width=1,
                color='black'
            )
        ),

        hovertemplate=(
            "Runs: %{customdata[0]}<br>"
            "Shot: %{customdata[1]}<br>"
            "Direction: %{customdata[2]}<br>"
            "X: %{x}<br>"
            "Y: %{y}"
            "<extra></extra>"
        ),

        customdata=player_df[
            ['batruns', 'shot']
        ].values,

        name='Shots'
    ))

    return fig


def get_batting_profile(
    master_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Computes comprehensive batting statistics grouped by specified columns."""
    batting_df = (
        master_df.groupby(group_cols)
        .agg(
            runs=("batruns", "sum"),
            balls=("is_legal", "sum"),
            dismissals=("out", "sum"),
            fours=("4s", "sum"),
            sixes=("6s", "sum"),
            boundaries=("boundaries", "sum"),
            dots=("batruns", lambda x: (x == 0).sum()),
            singles=("batruns", lambda x: (x == 1).sum()),
            twos=("batruns", lambda x: (x == 2).sum()),
            threes=("batruns", lambda x: (x == 3).sum()),
            bowled_dismissals = ('batsman_bowled_counter', 'sum'),
            caught_dismissals = ('batsman_caught_counter', 'sum'),
            lbw_dismissals = ('batsman_leg_before_wicket_counter', 'sum'),
            stumped_dismissals = ('batsman_stumped_counter', 'sum'),
            false_shots_number = ('false_shot', 'sum'),
            raa = ('raa_per_ball', 'sum'),
            entry_runs = ('batter_entry_runs', 'sum'),
            exit_runs = ('batter_exit_runs', 'sum'),
            entry_rr = ('batter_entry_rr', 'mean'),
            exit_rr = ('batter_exit_rr', 'mean'),
        )
        .reset_index()
    )

    batting_df["dot_ball_pct"] = (batting_df["dots"] * 100 / batting_df["balls"]).round(2)
    batting_df["run_rate"] = (batting_df["runs"] * 6 / batting_df["balls"]).round(2)
    batting_df["bat_overall_sr"] = (batting_df["runs"] * 100 / batting_df["balls"]).round(2)
    batting_df["bat_overall_avg"] = (batting_df["runs"] / batting_df["dismissals"].replace(0, np.nan)).round(2)
    batting_df["boundary_pct"] = (batting_df["boundaries"] * 100 / batting_df["balls"]).round(2)
    batting_df["balls_per_boundary"] = (batting_df["balls"] / batting_df["boundaries"].replace(0, np.nan)).round(2)
    batting_df["balls_per_4s"] = (batting_df["balls"] / batting_df["fours"].replace(0, np.nan)).round(2)
    batting_df["balls_per_6s"] = (batting_df["balls"] / batting_df["sixes"].replace(0, np.nan)).round(2)
    batting_df["dismissal_rate"] = (batting_df["balls"] / batting_df["dismissals"]).round(2)
    batting_df["rotation_pct"] = ((batting_df["singles"] + batting_df["twos"] + batting_df["threes"]) * 100 / batting_df["balls"]).round(2)
    batting_df['false_shot_%'] = (batting_df['false_shots_number'] * 100 / batting_df['balls']).round(2)
    batting_df['raa_per_100_balls'] = (batting_df['raa'] / batting_df['balls'] * 100).round(2)
    batting_df['runs_team_runs_ratio'] = batting_df['runs'] / (batting_df['exit_runs'] - batting_df['entry_runs']).replace(0, np.nan)
    batting_df['team_run_rate_diff'] = batting_df['exit_rr'] - batting_df['entry_rr']
    
    return batting_df


def get_bowling_profile(
    master_df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """Computes comprehensive bowling statistics grouped by specified columns."""
    bowling_df = (
        master_df.groupby(group_cols)
        .agg(
            runs_conceded = ('bowlruns', 'sum'),
            balls = ('is_legal', 'sum'),
            wickets = ('wicket_counter', 'sum'),
            dot_balls = ('batruns', lambda x: (x == 0).sum()),
            bowled_dismissals = ('batsman_bowled_counter', 'sum'),
            caught_dismissals = ('batsman_caught_counter', 'sum'),
            lbw_dismissals = ('batsman_leg_before_wicket_counter', 'sum'),
            stumped_dismissals = ('batsman_stumped_counter', 'sum'),
            batsman_not_in_control_balls = ('false_shot', 'sum'),
            boundaries=('boundaries', 'sum'),
            fours = ('4s', 'sum'),
            sixes = ('6s', 'sum'),
            false_shots_number = ('false_shot', 'sum')
        )
        .reset_index()
    )

    bowling_df["overall_eco"] = (bowling_df["runs_conceded"] * 6 / bowling_df["balls"]).round(2)
    bowling_df["bowl_overall_avg"] = (bowling_df["runs_conceded"] / bowling_df["wickets"].replace(0, np.nan)).round(2)
    bowling_df["bowl_overall_sr"] = (bowling_df["balls"] / bowling_df["wickets"].replace(0, np.nan)).round(2)
    bowling_df["balls_per_boundary_conc"] = (bowling_df["balls"] / bowling_df["boundaries"].replace(0, np.nan)).round(2)
    bowling_df["balls_per_4s_conc"] = (bowling_df["balls"] / bowling_df["fours"].replace(0, np.nan)).round(2)
    bowling_df["balls_per_6s_conc"] = (bowling_df["balls"] / bowling_df["sixes"].replace(0, np.nan)).round(2)
    bowling_df["dot_ball_pct"] = (bowling_df["dot_balls"] * 100 / bowling_df["balls"]).round(2)
    bowling_df['false_shot_%'] = (bowling_df['false_shots_number'] * 100 / bowling_df['balls']).round(2)

    return bowling_df


def get_pairs(x):
    """Generates unique pairs of items from an input list or iterable."""
    players = sorted(set(x))
    if len(players) < 2:
        return [(players[0], None)]
    
    pairs = []
    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            pairs.append((players[i], players[j]))
    return pairs


def get_batter_pca_impact(
        df_batting,
        feature_cols
    ):
    """Computes a PCA-based Batter Impact Score from core performance features."""
    df = df_batting.copy()
    
    df = df[df['balls'] > 18].reset_index(drop = True)

    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    
    pca_features = feature_cols.copy()
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[pca_features])
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled_data)
    
    df['PC1'] = pca_result[:, 0]
    df['PC2'] = pca_result[:, 1]
    
    explained_variance = pca.explained_variance_ratio_
    df['impact_score'] = (
        (df['PC1'] * explained_variance[0]) + 
        (df['PC2'] * explained_variance[1])
    ) / sum(explained_variance)
    
    df['impact_score'] = (df['impact_score'] - df['impact_score'].mean()) / df['impact_score'].std()
    
    df = df.sort_values(by='impact_score', ascending=False).reset_index(drop=True)
    
    print(f"PCA Explained Variance Ratio: PC1 = {explained_variance[0]:.2f}, PC2 = {explained_variance[1]:.2f}")
    
    return df


def plot_single_metric_heatmap(
    df, 
    metric_name, 
    role_in_question='batsman',
    title_text='', 
    group_cols = ['line', 'length'],
    length_order = ['yorker', 'full', 'good', 'short of a length', 'short'],
    line_order = ['far outside off', 'outside off', 'stumps', 'on leg', 'outside leg'],
    colorscale='YlOrRd',
    MAX_THRESHOLD = 250,
    metric_title = '',
    balls_col_candidates = ['balls', 'ball', 'deliveries']
):
    """Generates an individual metric heatmap with pitch lines and lengths."""
    if role_in_question == 'batsman':
        metrics_df = get_batting_profile(
            df,
            group_cols=group_cols
        )
    else:
        metrics_df = get_bowling_profile(
            df,
            group_cols=group_cols
        )
        
    pivot_df = metrics_df.pivot(index='length', columns='line', values=metric_name)

    balls_pivot = None
    actual_balls_col = next((col for col in balls_col_candidates if col in metrics_df.columns), None)
    if actual_balls_col:
        balls_pivot = metrics_df.pivot(index='length', columns='line', values=actual_balls_col)

    if isinstance(pivot_df.columns, pd.MultiIndex):
        pivot_df.columns = pivot_df.columns.get_level_values(1)
        if balls_pivot is not None and isinstance(balls_pivot.columns, pd.MultiIndex):
            balls_pivot.columns = balls_pivot.columns.get_level_values(1)

    pivot_df.columns = [str(x).replace("_", " ").title() for x in pivot_df.columns]
    pivot_df.index = [str(x).replace("_", " ").title() for x in pivot_df.index]
    
    reindexed_lengths = [l.replace("_", " ").title() for l in length_order]
    reindexed_lines = [l.replace("_", " ").title() for l in line_order]

    pivot_df = pivot_df.reindex(index=reindexed_lengths, columns=reindexed_lines)
    
    if balls_pivot is not None:
        balls_pivot.columns = [str(x).replace("_", " ").title() for x in balls_pivot.columns]
        balls_pivot.index = [str(x).replace("_", " ").title() for x in balls_pivot.index]
        balls_pivot = balls_pivot.reindex(index=reindexed_lengths, columns=reindexed_lines)

    text_matrix = []
    for r_idx, row_label in enumerate(pivot_df.index):
        row_texts = []
        for c_idx, col_label in enumerate(pivot_df.columns):
            val = pivot_df.iloc[r_idx, c_idx]
            if pd.isna(val):
                row_texts.append("")
            else:
                formatted_val = f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
                if balls_pivot is not None:
                    b_val = balls_pivot.iloc[r_idx, c_idx]
                    if not pd.isna(b_val):
                        row_texts.append(f"{formatted_val}<br>({int(b_val)})")
                    else:
                        row_texts.append(formatted_val)
                else:
                    row_texts.append(formatted_val)
        text_matrix.append(row_texts)

    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            text=text_matrix,
            texttemplate="%{text}",
            textfont=dict(
                size=11,
                color="black"
            ),
            colorscale=colorscale,
            zmin=0,
            zmax=MAX_THRESHOLD,
            zsmooth='best',
            hovertemplate="Line: %{x}<br>Length: %{y}<br>" + f"{metric_name}: %{{z}}<extra></extra>"
        )
    )

    fig.update_layout(
        title=title_text or f"{metric_title} by Pitch Line & Length",
        xaxis_title="Pitch Line",
        yaxis_title="Pitch Length",
        template="plotly_white",
        margin=dict(
            l=0, 
            r=0, 
            t=25, 
            b=0
        ),
        xaxis=dict(
            type='category',
            showgrid=False,
        ),
        yaxis=dict(
            type='category', 
            showgrid=False,
            autorange="reversed"
        )
    )

    return fig


def plot_team_run_rate_comparison(
    df: pd.DataFrame,
    selected_team: str,
    highlight_color: str = "#FF1E27",
    color_map_df = pd.DataFrame(),
    grayscale_color: str = "#C9C7C7",
    secondary_color = '',
    innings: list[int] = None,
    title: str = "Mean Current Run Rate Maintained at End of Each Over",
):
    """Plots a line chart comparing team run rates across overs."""
    filtered_df = df.copy()

    if innings:
        filtered_df = filtered_df[filtered_df['inns'].isin(innings)]

    run_rate_graph_df = filtered_df.copy()
    all_teams = run_rate_graph_df['team_bat'].dropna().unique()

    color_map = {
        team: (highlight_color if team == selected_team else grayscale_color)
        for team in all_teams
    }

    fig = px.line(
        run_rate_graph_df,
        x='over_num',
        y='crr',
        color='team_bat',
        color_discrete_map=color_map,
        markers=True,
        title=title
    )

    fig.add_vrect(
        x0=0, x1=6,
        fillcolor="lightgrey",
        opacity=0.06,
        line_width=0,
        layer="below"
    )

    fig.add_vrect(
        x0=6, x1=15,
        fillcolor="lightgrey",
        opacity=0.03,
        line_width=0,
        layer="below"
    )

    fig.add_vrect(
        x0=15, x1=20,
        fillcolor="lightgrey",
        opacity=0,
        line_width=0,
        layer="below"
    )

    for x, label in [(3, "POWERPLAY"), (10.5, "MIDDLE"), (17.5, "DEATH")]:
        fig.add_annotation(
            x=x,
            y=0.98,
            yref="paper",
            text=label,
            showarrow=False,
            font=dict(size=11),
            xanchor="center"
        )

    for trace in fig.data:
        if trace.name == selected_team:
            trace.line.width = 4
            trace.marker.size = 8
            trace.opacity = 1.0
        else:
            trace.line.width = 1.5
            trace.marker.size = 3
            trace.opacity = 0.2

    fig.update_layout(
        xaxis_title="Overs",
        yaxis_title="Current Run Rate",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        legend_title_text="Batting Team",        
        xaxis=dict(
            showgrid=False
        ),
        yaxis=dict(
            showgrid=False,
        )
    )

    return fig


def plot_partnership_runs(
    df: pd.DataFrame,
    selected_team: str = None,
    innings: list[int] = None,
    color_map = {},
    bar_color: str = "#1F77B4",
    secondary_color = '',
    title: str = "Partnership Runs by Wicket"
):
    """Plots a bar chart showing partnership runs grouped by wicket order."""
    filtered_df = df.copy()

    if selected_team:
        filtered_df = filtered_df[filtered_df['team_bat'] == selected_team]

    if innings:
        filtered_df = filtered_df[filtered_df['inns'].isin(innings)]

    batting_partnerships_df = (
        filtered_df.groupby(['phase', 'team_bat', 'partnership_id'], as_index=False)
        .agg(
            partnership_runs=('batruns', 'sum'),
            partnership_balls=('batruns', 'size')
        )
        .sort_values(['phase', "partnership_id"], ascending = False)
    )

    batting_partnerships_df['partnership_id'] = batting_partnerships_df['partnership_id']

    fig = px.bar(
        batting_partnerships_df,
        x="partnership_id",
        y="partnership_runs",
        color="phase",
        color_discrete_map=color_map,
        text="partnership_runs",
        hover_data=['partnership_runs', 'partnership_balls'],
        title=title
    )

    fig.update_traces(
        textposition="outside",
        textfont=dict(size=12, color="white"),
        hovertemplate="<b>Wicket %{x}</b><br>Runs: %{y}<br>Balls: %{customdata[1]}<extra></extra>"
    )

    fig.update_layout(
        barmode = 'stack',
        xaxis_title="Partnership for Wicket Number",
        yaxis_title="Runs Added",
        xaxis=dict(tickmode='linear', dtick=1, showgrid=False),
        plot_bgcolor=secondary_color,
        paper_bgcolor=secondary_color,
        hovermode = False,
        margin=dict(l=10, r=10, t=40, b=10),
        yaxis=dict(showgrid=False)
    )
    return fig


def plot_false_shots_by_over(
    df: pd.DataFrame,
    innings: list[int] = [],
    secondary_color = '',
    color_map = {},
    bar_color: str = "#1F77B4",
    title: str = "False Shots Induced by Over (Runs Scored Included)"
):
    """Plots a horizontal bar chart of false shot percentages induced by over."""
    filtered_df = df.copy()

    if innings and 'inns' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['inns'].isin(innings)]

    group_cols = ['phase','over_num']

    if 'inns' in filtered_df.columns and len(filtered_df['inns'].unique()) > 1:
        group_cols.append('inns')

    false_shot_df = filtered_df.groupby(group_cols, as_index=False).agg(
        false_shot_num=('false_shot', 'sum'),
        balls = ('is_legal', 'sum'),
        wickets_num=('false_shot_wkts', 'sum'),
    ).sort_values(group_cols, ascending = False)

    false_shot_df['false_shot_pct'] = round((false_shot_df['false_shot_num'] * 100/false_shot_df['balls']),2)
    if 'inns' in false_shot_df.columns:
        false_shot_df['inns'] = false_shot_df['inns'].astype(str)

    fig = px.bar(
        false_shot_df,
        y="over_num",
        x="false_shot_pct",
        color= 'phase',
        color_discrete_map= color_map,
        orientation = 'h',
        text="wickets_num",
        title=title,
    )

    fig.update_traces(
        textposition="outside",
        texttemplate="%{text}",
        textfont=dict(size=15, color="white"),
        hoverinfo='skip',
        hovertemplate=None,
    )

    fig.update_layout(
        yaxis_title="Over Number",
        xaxis_title="False Shot % Induced",
        hovermode=False,
        plot_bgcolor=secondary_color,
        paper_bgcolor=secondary_color,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=False, tickmode='linear', dtick=4),
        yaxis=dict(showgrid=False, autorange="reversed"),
    )

    return fig


def plot_economy_rate_comparison(
    df: pd.DataFrame,
    selected_team: str,
    season = [2023],
    metric = '',
    highlight_color: str = "#FF1E27",
    grayscale_color: str = "#A0A0A0",
    secondary_color = '',
    innings: list[int] = [],
    title: str = "Over-wise Economy Rate Comparison Across Teams"
):
    """Plots a comparison line chart for bowling metrics across overs for teams."""
    filtered_df = df.copy()
    team_col = 'team_bowl' 

    if innings and 'inns' in filtered_df.columns:
        filtered_df = filtered_df[
            (filtered_df['inns'].isin(innings)) &
            (filtered_df['year'].isin(season))
        ]

    economy_rate_df = get_bowling_profile(
        filtered_df,
        [team_col, 'over_num']
    )

    label_to_col_dict = {
        'Bowling Average' : 'bowl_overall_avg',
        'Economy Rate' : 'overall_eco',
        'Bowling Strike-Rate' : 'bowl_overall_sr',
        'Dot Ball %' : 'dot_ball_pct'
    }

    all_teams = economy_rate_df[team_col].dropna().unique()

    color_map = {
        team: (highlight_color if team == selected_team else grayscale_color)
        for team in all_teams
    }

    fig = px.line(
        economy_rate_df,
        x='over_num',
        y=label_to_col_dict[metric],
        color=team_col,
        color_discrete_map=color_map,
        markers=True,
        title=title
    )

    for trace in fig.data:
        if trace.name == selected_team:
            trace.line.width = 4
            trace.marker.size = 8
            trace.opacity = 1.0
        else:
            trace.line.width = 1.8
            trace.marker.size = 5
            trace.opacity = 0.15

    fig.update_layout(
        xaxis_title="Over Number",
        yaxis_title=metric,
        showlegend=False,
        legend_title_text="Team",
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(showgrid=True, tickmode='linear', dtick=2),
        yaxis=dict(showgrid=False)
    )

    return fig


def plot_raa_vs_dismissal_rate(
    filtered_df,
    highlight_color: str = "#FF1E27",
    chosen_team='CSK',
    title = "RAA per 100 Balls vs Impact Score for Batsmen (Min. 50 Balls Faced)",
    text_size: int = 11,
    text_color: str = "#1A1A1A",
    text_bg_color: str = "rgba(255, 255, 255, 0.85)"
):
    """Generates a scatter chart comparing RAA per 100 balls vs Impact Score with smart non-overlapping text labels."""
    df = filtered_df.copy()

    df['Legend'] = df['team_bat'].apply(
        lambda team: team if team == chosen_team else 'Other Teams'
    )

    color_discrete_map = {
        chosen_team: highlight_color,
        'Other Teams': '#C0C0C0'
    }

    fig = px.scatter(
        df,
        x='raa_per_100_balls',
        y='impact_score',
        color='Legend',
        color_discrete_map=color_discrete_map,
        hover_name='bat',
        hover_data={
            'team_bat': True,
            'raa_per_100_balls': ':.2f',
            'impact_score': ':.2f'
        },
        labels={
            'raa_per_100_balls': 'RAA per 100 Balls',
            'impact_score': 'Impact Score',
            'team_bat': 'Team'
        }
    )

    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey'))
    )

    offset_options = [
        {'xanchor': 'left',  'yanchor': 'bottom', 'xshift': 12,  'yshift': 8},   
        {'xanchor': 'right', 'yanchor': 'bottom', 'xshift': -12, 'yshift': 8},   
        {'xanchor': 'left',  'yanchor': 'top',    'xshift': 12,  'yshift': -8},  
        {'xanchor': 'right', 'yanchor': 'top',    'xshift': -12, 'yshift': -8},  
        {'xanchor': 'right', 'yanchor': 'middle', 'xshift': -15, 'yshift': 0},   
        {'xanchor': 'left',  'yanchor': 'middle', 'xshift': 15,  'yshift': 0}    
    ]

    chosen_team_df = df[df['team_bat'] == chosen_team].reset_index(drop=True)
    
    placed_coords = []

    for idx, row in chosen_team_df.iterrows():
        x_val = row['raa_per_100_balls']
        y_val = row['impact_score']
        
        chosen_offset = offset_options[idx % len(offset_options)]

        for coord in placed_coords:
            dist = np.sqrt((x_val - coord['x'])**2 + (y_val - coord['y'])**2)
            if dist < 5:
                chosen_offset = offset_options[(idx + 2) % len(offset_options)]
                break

        placed_coords.append({'x': x_val, 'y': y_val})

        fig.add_annotation(
            x=x_val,
            y=y_val,
            text=row['bat'],
            showarrow=False,
            xshift=chosen_offset['xshift'],
            yshift=chosen_offset['yshift'],
            xanchor=chosen_offset['xanchor'],
            yanchor=chosen_offset['yanchor'],
            font=dict(
                size=text_size,
                color=text_color
            ),
            bgcolor=text_bg_color,
            bordercolor="rgba(200, 200, 200, 0.4)",
            borderwidth=1,
            borderpad=3,
            opacity=0.95
        )

    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor='center'
        ),
        xaxis_title="<b>RAA per 100 Balls</b>",
        yaxis_title="<b>Impact Score</b>",
        template="plotly_white",
        width=800,
        height=600,
        margin=dict(l=40, r=40, t=60, b=40)
    )

    fig.update_xaxes(showgrid=False, title="<b>RAA per 100 Balls</b>")
    fig.update_yaxes(showgrid=False, title="<b>Impact Score</b>")

    fig.add_vline(x=0, line_dash="dash", line_color="gray", line_width=1.5)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1.5)

    return fig