# app_dash.py (final Dash app with both flat map and 3D globe support)

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.ndimage import gaussian_filter
import plotly.graph_objects as go
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon
import pandas as pd
import dash
from dash import dcc, html
import io
import base64
import matplotlib
matplotlib.use('Agg')  # for headless environments (like Render)

# === Load GISTEMP NetCDF ===
ds = xr.open_dataset("gistemp1200_GHCNv4_ERSSTv5.nc")
years = np.arange(1880, 2025)
anomaly_data = {year: ds.sel(time=slice(f"{year}-01", f"{year}-12"))["tempanomaly"].mean(dim="time").fillna(0)
                for year in years}

# === Load shapefile ===
world = gpd.read_file("ne_110m_admin_0_countries.shp")

def get_camera_eye_from_latlon(lat_deg, lon_deg, distance=1.5):
    lat_rad = np.radians(lat_deg)
    lon_rad = np.radians(lon_deg)
    x = distance * np.cos(lat_rad) * np.cos(lon_rad)
    y = distance * np.cos(lat_rad) * np.sin(lon_rad)
    z = distance * np.sin(lat_rad)
    return dict(x=x, y=y, z=z)

def plot_flat_map_image(year, region):
    temp = anomaly_data[year]
    lon_new = np.linspace(temp.lon.min().item(), temp.lon.max().item(), 360)
    lat_new = np.linspace(temp.lat.min().item(), temp.lat.max().item(), 180)
    temp_interp = temp.interp(lon=lon_new, lat=lat_new, method='linear')
    filled = temp_interp.fillna(0)
    smoothed = gaussian_filter(filled, sigma=1.2)
    mask = np.isnan(temp_interp)
    masked_smoothed = np.ma.masked_where(mask, smoothed)

    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.Robinson())

    extents = {
        'north_america': [-170, -50, 10, 80],
        'south_america': [-90, -30, -60, 15],
        'europe': [-25, 45, 30, 70],
        'africa': [-20, 55, -35, 40],
        'asia': [50, 150, 5, 60],
        'oceania': [110, 180, -50, 10],
        'antarctica': [-180, 180, -90, -60]
    }
    if region in extents:
        ax.set_extent(extents[region], crs=ccrs.PlateCarree())
    else:
        ax.set_global()

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('lightgray')

    lon2d, lat2d = np.meshgrid(lon_new, lat_new)
    mesh = ax.pcolormesh(lon2d, lat2d, masked_smoothed,
                         transform=ccrs.PlateCarree(),
                         cmap=cmap, shading='auto', vmin=-5, vmax=5)

    cb = plt.colorbar(mesh, orientation='horizontal', pad=0.03, shrink=1.0, aspect=40)
    cb.set_label("Temperature Anomaly (°C)", fontsize=12)
    cb.set_ticks([-5, -2.5, 0, 2.5, 5])
    cb.set_ticklabels(["-5°C", "-2.5°C", "0°C", "+2.5°C", "+5°C"])
    plt.title(f"GISTEMP Global Temperature Anomaly Map (Smoothed) - {year}", fontsize=12)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(fig)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"

def plot_globe(year, region):
    temp = anomaly_data[year].load()
    df = temp.to_dataframe().reset_index().dropna()

    # Adjust longitudes to 0-360
    df['adj_lon'] = df['lon'] % 360

    # Explicitly wrap 360° for smooth globe closure
    # df_360 = df[df['lon'] == -180.0].copy()
    # df_360['adj_lon'] = 360.0
    # df = pd.concat([df, df_360])
    df_360 = df[df['lon'] == -180.0].copy()
    if not df_360.empty:
        df_360['adj_lon'] = 360.0
        df = pd.concat([df, df_360])

    lat = np.sort(df['lat'].unique())
    lon = np.linspace(0, 360, 181)
    LON, LAT = np.meshgrid(lon, lat)

    anomaly_matrix = df.pivot_table(
        index='lat',
        columns='adj_lon',
        values='tempanomaly',
        aggfunc='first',
        dropna=False
    ).reindex(index=lat, columns=lon, method='nearest').values

    lat_rad = np.radians(LAT)
    lon_rad = np.radians(LON - 180)

    R = 1 * (1 - 1e-6 * np.abs(LAT)/90)
    X = R * np.cos(lat_rad) * np.cos(lon_rad)
    Y = R * np.cos(lat_rad) * np.sin(lon_rad)
    Z = R * np.sin(lat_rad)

    text_matrix = np.array([
        [f"Lat: {lat:.1f}°<br>Lon: {(lon-180):.1f}°<br>Anomaly: {anom:.2f}°C"
         for lon, anom in zip(lon_row, anom_row)]
        for lat, lon_row, anom_row in zip(lat, LON, anomaly_matrix)
    ])

    surface = go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=anomaly_matrix,
        colorscale='RdBu_r',
        cmin=-5,
        cmax=5,
        colorbar=dict(title="Temperature Anomaly (°C)", thickness=20, tickvals=np.arange(-4, 5, 1)),
        text=text_matrix,
        hoverinfo='text',
        connectgaps=False,
        # lighting=dict(ambient=0.95, diffuse=0.75, fresnel=0.3, specular=0.3),
        # lightposition=dict(x=1000, y=1000, z=1000),
        contours={"z": {"show": True, "width": 1, "color": "rgba(0,0,0,0.1)"}}
    )
    coastlines = []
    for geom in world.geometry:
        if isinstance(geom, Polygon):
            polygons = [geom]
        elif isinstance(geom, MultiPolygon):
            polygons = list(geom.geoms)
        else:
            continue
        for poly in polygons:
            lons, lats = poly.exterior.coords.xy
            lons = np.array(lons)
            lats = np.array(lats)
            x_line = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
            y_line = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
            z_line = np.sin(np.radians(lats))
            coastlines.append(go.Scatter3d(
                x=x_line, y=y_line, z=z_line,
                mode='lines',
                line=dict(color='black', width=1),
                hoverinfo='skip',
                showlegend=False
            ))

    region_centers = {
        'global': (20, 0),
        'north_america': (45, -100),
        'south_america': (-15, -60),
        'europe': (50, 10),
        'africa': (0, 20),
        'asia': (30, 100),
        'oceania': (-25, 135),
        'antarctica': (-85, 0)
    }
    center_lat, center_lon = region_centers.get(region, (20, 0))
    camera_eye = get_camera_eye_from_latlon(center_lat, center_lon)

    fig = go.Figure(data=[surface] + coastlines)
    fig.update_layout(
        title=f"Temperature Anomalies and Coastlines – {year}",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=1.2, z=1.2),
            camera=dict(eye=camera_eye)
        ),
        margin=dict(t=50, l=0, r=0, b=0)
    )
    return fig

# === Dash Layout ===
app = dash.Dash(__name__)
app.title = "GISTEMP Viewer"

app.layout = html.Div([
    html.H2("GISTEMP Temperature Anomaly Viewer"),
    html.Label("Select Year:"),
    dcc.Slider(id='year-slider', min=1880, max=2024, step=1, value=2024,
               marks={y: str(y) for y in range(1880, 2025, 20)},
               tooltip={"placement": "bottom", "always_visible": True}),
    html.Label("Select Region:"),
    dcc.Dropdown(id='region-dropdown', value='global', options=[
        {'label': 'Global', 'value': 'global'},
        {'label': 'North America', 'value': 'north_america'},
        {'label': 'South America', 'value': 'south_america'},
        {'label': 'Europe', 'value': 'europe'},
        {'label': 'Africa', 'value': 'africa'},
        {'label': 'Asia', 'value': 'asia'},
        {'label': 'Australia/Oceania', 'value': 'oceania'},
        {'label': 'Antarctica', 'value': 'antarctica'},
    ]),
    html.Label("Map Type:"),
    dcc.RadioItems(id='map-type', value='flat', options=[
        {'label': 'Flat Map', 'value': 'flat'},
        {'label': '3D Globe', 'value': '3d'},
    ], inline=True),
    html.Div(id='map-container')
])

@app.callback(
    dash.dependencies.Output('map-container', 'children'),
    [dash.dependencies.Input('year-slider', 'value'),
     dash.dependencies.Input('region-dropdown', 'value'),
     dash.dependencies.Input('map-type', 'value')]
)
def update_map(year, region, map_type):
    if map_type == '3d':
        fig = plot_globe(year, region)
        return dcc.Graph(figure=fig)
    else:
        img_data = plot_flat_map_image(year, region)
        return html.Img(src=img_data, style={"width": "100%"})

if __name__ == '__main__':
    app.run(debug=True)
