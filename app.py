import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Initialize Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = "🌡️ Global Temperature Change Dashboard"

# ========== DATA PREPARATION ==========
print("📂 Loading and preprocessing data...")

# Load the dataset
df = pd.read_csv('Environment_Temperature_change_E_All_Data_NOFLAG.csv', encoding='ISO-8859-1')

# Step 1: Renaming and preprocessing the data
df.rename(columns={'Area': 'Country Name'}, inplace=True)

# Set 'Months' as the index and rename seasons
df.set_index('Months', inplace=True)
df.rename({'Dec\x96Jan\x96Feb': 'Winter', 'Mar\x96Apr\x96May': 'Spring', 
           'Jun\x96Jul\x96Aug': 'Summer', 'Sep\x96Oct\x96Nov': 'Fall'}, axis='index', inplace=True)

# Reset the index to bring 'Months' back as a column
df.reset_index(inplace=True)

# Step 2: Filter out rows where 'Element' is not 'Temperature change'
df = df[df['Element'] == 'Temperature change']

# Step 3: Drop unnecessary columns from the main dataset
df.drop(['Area Code', 'Months Code', 'Element Code', 'Element', 'Unit'], axis=1, inplace=True)

# Step 4: Reshape the data using melt to make it suitable for clustering
df = df.melt(id_vars=["Country Name", "Months"], var_name="year", value_name="tem_change")

# Step 5: Extract the year part (removing the 'Y' from the year column)
df["year"] = df["year"].apply(lambda x: x.split("Y")[-1])

# Step 6: Handle missing values (drop rows with NaN in 'tem_change')
df_cleaned = df.dropna(subset=['tem_change'])

print(f"✅ Data loaded: {len(df_cleaned):,} rows, {df_cleaned['Country Name'].nunique()} countries")

# ========== APPLY CLUSTERING ==========
print("🤖 Applying K-Means clustering...")

# Standardize the data
temperature_data = df_cleaned[['tem_change']]
scaler = StandardScaler()
temperature_data_scaled = scaler.fit_transform(temperature_data)

# Apply K-Means clustering to categorize countries based on their temperature change
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_cleaned['Cluster'] = kmeans.fit_predict(temperature_data_scaled)

print(f"✅ Clustering complete: {df_cleaned['Cluster'].nunique()} clusters created")

# ========== PREPARE AGGREGATED DATA ==========
# Convert year to integer for calculations
df_cleaned['year_int'] = df_cleaned['year'].astype(int)

# Aggregating data for trend lines (average temperature change per cluster)
df_agg = df_cleaned.groupby(['Cluster', 'year_int'])['tem_change'].mean().reset_index()

# Seasonal analysis
df_seasonal = df_cleaned.groupby(['Cluster', 'Months'])['tem_change'].mean().reset_index()
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
df_seasonal['Months'] = pd.Categorical(df_seasonal['Months'], categories=season_order, ordered=True)

# Decadal analysis
df_cleaned['decade'] = (df_cleaned['year_int'] // 10) * 10
df_decadal = df_cleaned.groupby(['Cluster', 'decade'])['tem_change'].mean().reset_index()

# Top countries per cluster
df_country_avg = df_cleaned.groupby(['Cluster', 'Country Name'])['tem_change'].mean().reset_index()
top_countries_per_cluster = []
for cluster in sorted(df_country_avg['Cluster'].unique()):
    cluster_data = df_country_avg[df_country_avg['Cluster'] == cluster]
    top_5 = cluster_data.nlargest(5, 'tem_change')
    top_countries_per_cluster.append(top_5)

df_top_countries = pd.concat(top_countries_per_cluster)

# ========== IMPROVED COLOR SCHEME ==========
# Professional, accessible color scheme with good contrast
COLOR_SCHEME = {
    0: '#2E86AB',    # Rich Blue (cool cluster - lower temps)
    1: '#A23B72',    # Magenta (medium temp cluster)
    2: '#F18F01',    # Golden Orange (warm cluster)
    3: '#C73E1D',    # Brick Red (hot cluster - highest temps)
}

# Colors for highlighting during brushing
HIGHLIGHT_COLORS = {
    0: '#5BC0EB',    # Bright Blue for highlighting cluster 0
    1: '#FF6B6B',    # Bright Coral for highlighting cluster 1
    2: '#FFD166',    # Bright Yellow for highlighting cluster 2
    3: '#EF476F',    # Bright Pink for highlighting cluster 3
}

# ========== DASHBOARD LAYOUT ==========
app.layout = dbc.Container([
    # HEADER
    dbc.Row([
        dbc.Col([
            html.H1("🌡️ Global Temperature Change Analysis Dashboard", 
                   className="text-center mb-4",
                   style={'color': '#FFD700'}),
            html.P("Interactive visualization of temperature anomalies (1961-2019) with brushing & linking",
                  className="text-center text-muted mb-4"),
            html.P("🎯 Click or select in any chart to highlight across all visualizations",
                  className="text-center text-warning"),
        ], width=12)
    ]),
    
    # FILTER CONTROLS
    dbc.Row([
        dbc.Col([
            html.Label("Select Cluster(s):", style={'color': '#FFD700'}),
            dcc.Dropdown(
                id='cluster-filter',
                options=[{'label': f'Cluster {c}', 'value': c} 
                        for c in sorted(df_cleaned['Cluster'].unique())],
                value=list(sorted(df_cleaned['Cluster'].unique())),
                multi=True,
                style={'backgroundColor': '#222', 'color': 'white'}
            )
        ], width=6),
        dbc.Col([
            html.Label("Select Year Range:", style={'color': '#FFD700'}),
            dcc.RangeSlider(
                id='year-slider',
                min=int(df_cleaned['year_int'].min()),
                max=int(df_cleaned['year_int'].max()),
                step=5,
                marks={str(year): str(year) for year in range(1960, 2020, 10)},
                value=[1961, 2019],
                tooltip={"placement": "bottom", "always_visible": True}
            )
        ], width=6)
    ], className="mb-4"),
    
    # MAIN VISUALIZATIONS
    dbc.Row([
        # Chart 1: Scatter Plot
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Temperature Anomalies by Year", style={'color': '#FFD700'}),
                dbc.CardBody([
                    dcc.Graph(
                        id='scatter-plot', 
                        config={'displayModeBar': True},
                        style={'height': '400px'}
                    ),
                    html.Small("Click points or draw selection box to highlight across dashboard", 
                             className="text-muted mt-2")
                ])
            ], className="h-100")
        ], width=6),
        
        # Chart 2: Trend Lines
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Average Temperature Trends", style={'color': '#FFD700'}),
                dbc.CardBody([
                    dcc.Graph(
                        id='trend-plot', 
                        config={'displayModeBar': True},
                        style={'height': '400px'}
                    ),
                    html.Small("Click lines or markers to highlight clusters", 
                             className="text-muted mt-2")
                ])
            ], className="h-100")
        ], width=6)
    ], className="mb-4"),
    
    dbc.Row([
        # Chart 3: Box Plot
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📦 Temperature Distribution by Cluster", style={'color': '#FFD700'}),
                dbc.CardBody([
                    dcc.Graph(
                        id='box-plot', 
                        config={'displayModeBar': True},
                        style={'height': '400px'}
                    ),
                    html.Small("Click on any box to highlight that cluster", 
                             className="text-muted mt-2")
                ])
            ], className="h-100")
        ], width=6),
        
        # Chart 4: Tabs for other visualizations
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Additional Analysis", style={'color': '#FFD700'}),
                dbc.CardBody([
                    dcc.Tabs([
                        dcc.Tab(label='🏆 Top Countries', children=[
                            dcc.Graph(id='top-countries', config={'displayModeBar': True})
                        ]),
                        dcc.Tab(label='🌤️ Seasonal Patterns', children=[
                            dcc.Graph(id='seasonal-plot', config={'displayModeBar': True})
                        ]),
                        dcc.Tab(label='📅 Decadal Trends', children=[
                            dcc.Graph(id='decadal-plot', config={'displayModeBar': True})
                        ])
                    ], colors={
                        "border": "#FFD700",
                        "primary": "#A23B72",
                        "background": "#222"
                    })
                ])
            ], className="h-100")
        ], width=6)
    ]),
    
    # INFO PANEL
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Statistics & Interaction Panel", style={'color': '#FFD700'}),
                dbc.CardBody([
                    html.Div(id='stats-panel'),
                    html.Hr(),
                    html.Div([
                        html.H6("🎯 Brushing & Linking Instructions:", className="text-warning"),
                        html.Ul([
                            html.Li("Click on any point in Scatter Plot to highlight that cluster"),
                            html.Li("Draw a selection box in Scatter Plot to select multiple points"),
                            html.Li("Click on any Box Plot to highlight that cluster"),
                            html.Li("Click on Trend Lines to highlight clusters"),
                            html.Li("Double-click any chart to reset highlighting")
                        ])
                    ])
                ])
            ], color="dark", inverse=True)
        ], width=12)
    ], className="mt-4")
], fluid=True, style={'backgroundColor': '#121212'})

# ========== GLOBAL VARIABLES FOR BRUSHING ==========
highlighted_cluster = None  # Track which cluster is highlighted

# ========== MAIN CALLBACK FUNCTION ==========
@app.callback(
    [Output('scatter-plot', 'figure'),
     Output('trend-plot', 'figure'),
     Output('box-plot', 'figure'),
     Output('top-countries', 'figure'),
     Output('seasonal-plot', 'figure'),
     Output('decadal-plot', 'figure'),
     Output('stats-panel', 'children')],
    [Input('cluster-filter', 'value'),
     Input('year-slider', 'value'),
     Input('scatter-plot', 'selectedData'),
     Input('box-plot', 'clickData'),
     Input('trend-plot', 'clickData')]
)
def update_dashboard(selected_clusters, year_range, scatter_selected, box_clicked, trend_clicked):
    global highlighted_cluster
    
    ctx = callback_context
    
    # Determine which input triggered the callback
    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # Update highlighted cluster based on interaction
    if trigger_id == 'scatter-plot' and scatter_selected:
        points = scatter_selected.get('points', [])
        if points:
            # Get cluster from selected point
            point_cluster = points[0].get('curveNumber', None)
            if point_cluster is not None and point_cluster < len(selected_clusters):
                highlighted_cluster = selected_clusters[point_cluster]
    
    elif trigger_id == 'box-plot' and box_clicked:
        points = box_clicked.get('points', [])
        if points:
            # Get cluster from clicked box
            cluster_label = points[0].get('x', '')
            if 'Cluster' in cluster_label:
                highlighted_cluster = int(cluster_label.split('Cluster ')[1])
    
    elif trigger_id == 'trend-plot' and trend_clicked:
        points = trend_clicked.get('points', [])
        if points:
            # Get cluster from clicked trend line
            point_cluster = points[0].get('curveNumber', None)
            if point_cluster is not None and point_cluster < len(selected_clusters):
                highlighted_cluster = selected_clusters[point_cluster]
    
    # If no specific selection, reset highlighting
    elif not any([scatter_selected, box_clicked, trend_clicked]):
        highlighted_cluster = None
    
    # Filter data based on selections
    filtered_df = df_cleaned[
        (df_cleaned['Cluster'].isin(selected_clusters)) &
        (df_cleaned['year_int'] >= year_range[0]) &
        (df_cleaned['year_int'] <= year_range[1])
    ]
    
    filtered_agg = df_agg[
        (df_agg['Cluster'].isin(selected_clusters)) &
        (df_agg['year_int'] >= year_range[0]) &
        (df_agg['year_int'] <= year_range[1])
    ]
    
    filtered_top = df_top_countries[df_top_countries['Cluster'].isin(selected_clusters)]
    
    # ========== CREATE VISUALIZATIONS WITH HIGHLIGHTING ==========
    
    # 1. SCATTER PLOT
    scatter_fig = go.Figure()
    for i, cluster in enumerate(sorted(selected_clusters)):
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster]
        # Sample for better performance if too many points
        if len(cluster_data) > 1000:
            cluster_data = cluster_data.sample(1000, random_state=42)
        
        base_color = COLOR_SCHEME.get(cluster, '#2E86AB')
        highlight_color = HIGHLIGHT_COLORS.get(cluster, '#5BC0EB')
        
        # Determine if this cluster should be highlighted
        is_highlighted = (highlighted_cluster == cluster)
        
        scatter_fig.add_trace(go.Scatter(
            x=cluster_data['year'],
            y=cluster_data['tem_change'],
            mode='markers',
            name=f'Cluster {cluster}',
            marker=dict(
                size=8 if is_highlighted else 4,
                opacity=1.0 if is_highlighted else 0.6,
                color=highlight_color if is_highlighted else base_color,
                line=dict(
                    width=3 if is_highlighted else 0.5,
                    color='white' if is_highlighted else 'rgba(255,255,255,0.3)'
                ),
                symbol='circle' if is_highlighted else 'circle'
            ),
            customdata=np.column_stack([cluster_data['Country Name'], 
                                       [cluster] * len(cluster_data)]),
            hovertemplate=(
                "<b>Cluster %{customdata[1]}</b><br>"
                "Year: %{x}<br>"
                "Country: %{customdata[0]}<br>"
                "Temp Change: %{y:.2f}°C<br>"
                "<extra></extra>"
            ),
            selectedpoints=[] if is_highlighted else None,
            selected=dict(marker=dict(opacity=1.0, size=10)) if is_highlighted else None
        ))
    
    scatter_fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Year", gridcolor='#444', showgrid=True),
        yaxis=dict(title="Temperature Anomaly (°C)", gridcolor='#444', showgrid=True),
        hovermode='closest',
        dragmode='select',  # Enable selection mode
        showlegend=True
    )
    
    # Add selection instructions
    scatter_fig.add_annotation(
        text="Click points or draw box to select",
        xref="paper", yref="paper",
        x=0.02, y=0.02,
        showarrow=False,
        font=dict(size=10, color="#AAAAAA"),
        bgcolor="rgba(0,0,0,0.5)"
    )
    
    # 2. TREND LINES
    trend_fig = go.Figure()
    for i, cluster in enumerate(sorted(selected_clusters)):
        cluster_data = filtered_agg[filtered_agg['Cluster'] == cluster]
        base_color = COLOR_SCHEME.get(cluster, '#2E86AB')
        highlight_color = HIGHLIGHT_COLORS.get(cluster, '#5BC0EB')
        
        # Determine if this cluster should be highlighted
        is_highlighted = (highlighted_cluster == cluster)
        
        trend_fig.add_trace(go.Scatter(
            x=cluster_data['year_int'],
            y=cluster_data['tem_change'],
            mode='lines+markers',
            name=f'Cluster {cluster}',
            line=dict(
                width=4 if is_highlighted else 2,
                color=highlight_color if is_highlighted else base_color,
                dash='solid'
            ),
            marker=dict(
                size=10 if is_highlighted else 6,
                color=highlight_color if is_highlighted else base_color,
                line=dict(width=2 if is_highlighted else 1, color='white')
            ),
            hovertemplate=(
                "<b>Cluster %{fullData.name}</b><br>"
                "Year: %{x}<br>"
                "Avg Temp: %{y:.2f}°C<br>"
                "<extra></extra>"
            )
        ))
    
    trend_fig.update_layout(
        title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Year", gridcolor='#444', showgrid=True),
        yaxis=dict(title="Avg Temperature Anomaly (°C)", gridcolor='#444', showgrid=True),
        hovermode='closest'
    )
    
    # 3. BOX PLOT
    box_fig = go.Figure()
    for i, cluster in enumerate(sorted(selected_clusters)):
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster]
        base_color = COLOR_SCHEME.get(cluster, '#2E86AB')
        highlight_color = HIGHLIGHT_COLORS.get(cluster, '#5BC0EB')
        
        # Determine if this cluster should be highlighted
        is_highlighted = (highlighted_cluster == cluster)
        
        box_fig.add_trace(go.Box(
            y=cluster_data['tem_change'],
            name=f'Cluster {cluster}',
            marker_color=highlight_color if is_highlighted else base_color,
            line=dict(
                color='white' if is_highlighted else 'rgba(255,255,255,0.7)',
                width=3 if is_highlighted else 1.5
            ),
            fillcolor=f'rgba{tuple(int(highlight_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.3,)}' if is_highlighted else 
                     f'rgba{tuple(int(base_color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4)) + (0.2,)}',
            boxmean='sd',
            hovertemplate=(
                "<b>Cluster %{x}</b><br>"
                "Temp Change: %{y:.2f}°C<br>"
                "<extra></extra>"
            )
        ))
    
    box_fig.update_layout(
        title="",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Cluster", gridcolor='#444', showgrid=True),
        yaxis=dict(title="Temperature Anomaly (°C)", gridcolor='#444', showgrid=True),
        showlegend=False
    )
    
    # 4. TOP COUNTRIES
    top_fig = go.Figure()
    for i, cluster in enumerate(sorted(selected_clusters)):
        cluster_data = filtered_top[filtered_top['Cluster'] == cluster].head(5)
        base_color = COLOR_SCHEME.get(cluster, '#2E86AB')
        highlight_color = HIGHLIGHT_COLORS.get(cluster, '#5BC0EB')
        
        # Determine if this cluster should be highlighted
        is_highlighted = (highlighted_cluster == cluster)
        
        # Shorten long country names
        cluster_data['Country_Short'] = cluster_data['Country Name'].apply(
            lambda x: x[:20] + '...' if len(x) > 20 else x
        )
        
        top_fig.add_trace(go.Bar(
            x=cluster_data['Country_Short'],
            y=cluster_data['tem_change'],
            name=f'Cluster {cluster}',
            marker_color=highlight_color if is_highlighted else base_color,
            marker_line=dict(
                color='white' if is_highlighted else 'rgba(255,255,255,0.5)',
                width=2 if is_highlighted else 1
            ),
            opacity=1.0 if is_highlighted else 0.8,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Cluster: %{customdata[1]}<br>"
                "Avg Temp: %{y:.2f}°C<br>"
                "<extra></extra>"
            ),
            customdata=list(zip(cluster_data['Country Name'], 
                              [f'Cluster {cluster}'] * len(cluster_data)))
        ))
    
    top_fig.update_layout(
        title="Top 5 Countries by Temperature Change",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Country", tickangle=45, gridcolor='#444', showgrid=True),
        yaxis=dict(title="Avg Temperature Anomaly (°C)", gridcolor='#444', showgrid=True),
        barmode='group',
        showlegend=False
    )
    
    # 5. SEASONAL ANALYSIS
    seasonal_fig = go.Figure()
    seasonal_data = df_seasonal[df_seasonal['Cluster'].isin(selected_clusters)]
    
    for i, cluster in enumerate(sorted(selected_clusters)):
        cluster_data = seasonal_data[seasonal_data['Cluster'] == cluster].sort_values('Months')
        base_color = COLOR_SCHEME.get(cluster, '#2E86AB')
        highlight_color = HIGHLIGHT_COLORS.get(cluster, '#5BC0EB')
        
        # Determine if this cluster should be highlighted
        is_highlighted = (highlighted_cluster == cluster)
        
        seasonal_fig.add_trace(go.Scatter(
            x=cluster_data['Months'],
            y=cluster_data['tem_change'],
            mode='lines+markers',
            name=f'Cluster {cluster}',
            line=dict(
                width=3 if is_highlighted else 1.5,
                color=highlight_color if is_highlighted else base_color,
                dash='solid'
            ),
            marker=dict(
                size=10 if is_highlighted else 6,
                color=highlight_color if is_highlighted else base_color,
                symbol='diamond' if is_highlighted else 'circle',
                line=dict(width=2 if is_highlighted else 1, color='white')
            ),
            hovertemplate=(
                "<b>Cluster %{fullData.name}</b><br>"
                "Season: %{x}<br>"
                "Avg Temp: %{y:.2f}°C<br>"
                "<extra></extra>"
            )
        ))
    
    seasonal_fig.update_layout(
        title="Seasonal Temperature Patterns",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Season", gridcolor='#444', showgrid=True),
        yaxis=dict(title="Avg Temperature Anomaly (°C)", gridcolor='#444', showgrid=True)
    )
    
    # 6. DECADAL TRENDS
    decadal_fig = go.Figure()
    decadal_data = df_decadal[df_decadal['Cluster'].isin(selected_clusters)]
    
    for i, cluster in enumerate(sorted(selected_clusters)):
        cluster_data = decadal_data[decadal_data['Cluster'] == cluster].sort_values('decade')
        base_color = COLOR_SCHEME.get(cluster, '#2E86AB')
        highlight_color = HIGHLIGHT_COLORS.get(cluster, '#5BC0EB')
        
        # Determine if this cluster should be highlighted
        is_highlighted = (highlighted_cluster == cluster)
        
        decadal_fig.add_trace(go.Bar(
            x=cluster_data['decade'].astype(str) + 's',
            y=cluster_data['tem_change'],
            name=f'Cluster {cluster}',
            marker_color=highlight_color if is_highlighted else base_color,
            marker_line=dict(
                color='white' if is_highlighted else 'rgba(255,255,255,0.5)',
                width=2 if is_highlighted else 1
            ),
            opacity=1.0 if is_highlighted else 0.8,
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Decade: %{x}<br>"
                "Avg Temp: %{y:.2f}°C<br>"
                "<extra></extra>"
            )
        ))
    
    decadal_fig.update_layout(
        title="Decadal Temperature Trends",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis=dict(title="Decade", gridcolor='#444', showgrid=True),
        yaxis=dict(title="Avg Temperature Anomaly (°C)", gridcolor='#444', showgrid=True),
        barmode='group',
        showlegend=False
    )
    
    # 7. STATISTICS PANEL
    stats_text = []
    
    if highlighted_cluster is not None and highlighted_cluster in selected_clusters:
        stats_text.append(html.H4(f"🎯 Currently Highlighted: Cluster {highlighted_cluster}", 
                                style={'color': HIGHLIGHT_COLORS.get(highlighted_cluster, '#FFD700')}))
    
    for cluster in selected_clusters:
        cluster_data = filtered_df[filtered_df['Cluster'] == cluster]
        is_highlighted = (highlighted_cluster == cluster)
        text_color = HIGHLIGHT_COLORS.get(cluster, '#FFFFFF') if is_highlighted else COLOR_SCHEME.get(cluster, '#FFFFFF')
        
        stats_text.append(html.H5(f"📊 Cluster {cluster}", style={'color': text_color}))
        stats_text.append(html.P(f"Countries: {cluster_data['Country Name'].nunique()}"))
        stats_text.append(html.P(f"Avg Temp Change: {cluster_data['tem_change'].mean():.2f}°C"))
        stats_text.append(html.P(f"Range: {cluster_data['tem_change'].min():.2f}°C to {cluster_data['tem_change'].max():.2f}°C"))
        stats_text.append(html.Hr())
    
    stats_text.append(html.P(f"📅 Year Range: {year_range[0]} - {year_range[1]}"))
    stats_text.append(html.P(f"🌍 Total Countries: {filtered_df['Country Name'].nunique()}"))
    stats_text.append(html.P(f"📈 Total Observations: {len(filtered_df):,}"))
    
    if highlighted_cluster is None:
        stats_text.append(html.P("💡 Click on any chart to highlight a cluster", 
                               className="text-warning mt-3"))
    
    return (scatter_fig, trend_fig, box_fig, top_fig, seasonal_fig, decadal_fig, stats_text)

# ========== RESET HIGHLIGHTING ON DOUBLE-CLICK ==========
@app.callback(
    [Output('scatter-plot', 'selectedData', allow_duplicate=True),
     Output('box-plot', 'clickData', allow_duplicate=True),
     Output('trend-plot', 'clickData', allow_duplicate=True)],
    [Input('scatter-plot', 'clickData'),
     Input('box-plot', 'clickData'),
     Input('trend-plot', 'clickData')],
    prevent_initial_call=True
)
def reset_on_double_click(scatter_click, box_click, trend_click):
    # This function doesn't modify data, just clears selections on double-click
    # The actual reset happens in the main callback when no selection is active
    return None, None, None

# ========== RUN THE DASHBOARD ==========
if __name__ == '__main__':
    print("Dashboard setup complete!")
    print("Features:")
    print("   • Brushing & Linking across all charts")
    print("   • Click to highlight clusters")
    print("   • Selection boxes in scatter plot")
    print("   • Double-click to reset highlighting")
    print("   • Professional color scheme")
    print("\nOpening dashboard at: http://127.0.0.1:8050")
    print("Please wait a moment for the data to load...")
    app.run(debug=True, port=8050)

server = app.server

if __name__ == '__main__':
    app.run(debug=False)  # Set debug=False for production