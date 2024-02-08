
import dash
import datetime

import numpy as np
import plotly.graph_objs as go

from dash import dcc, html
from dash.dependencies import Input, Output

class Dashboard:
    def __init__(self, title, obstacles, robots):
        self.obstacles = obstacles
        self.robots = robots
        self._app = dash.Dash(title)
        self.setup_layout()
        self.setup_callbacks()

    def setup_layout(self):
        self._app.layout = html.Div([
            html.H1("Current Time", style={'textAlign': 'center'}),
            html.Div(id='live-clock', style={'fontSize': '48px', 'textAlign': 'center'}),
            dcc.Interval(
                id='interval-component',
                interval=1000,  # Update every 1 second
                n_intervals=0
            ),
            html.Hr(),  # Add a horizontal line
            dcc.Graph(id='environment-plot')  # Div to hold the plot
        ])

    def setup_callbacks(self):
        self._app.callback(Output('live-clock', 'children'), [Input('interval-component', 'n_intervals')])(self._update_clock)
        self._app.callback(Output('environment-plot', 'figure'), [Input('interval-component', 'n_intervals')])(self._update_map)

    def _update_clock(self, *args):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return current_time
    
    def _update_map(self, *args):
        fig = go.Figure()

        for obstacle_name, obstacle in self.obstacles.items():
            pos, size, rot = obstacle.pos, obstacle.size, np.degrees(obstacle.rot[-1])

            x_vals = [pos[0] - size[0] / 2, pos[0] + size[0] / 2, pos[0] + size[0] / 2, pos[0] - size[0] / 2, pos[0] - size[0] / 2]
            y_vals = [pos[1] - size[1] / 2, pos[1] - size[1] / 2, pos[1] + size[1] / 2, pos[1] + size[1] / 2, pos[1] - size[1] / 2]

            x_center = pos[0]
            y_center = pos[1]
            rot_rad = np.radians(rot)
            x_rotated = [(x - x_center) * np.cos(rot_rad) - (y - y_center) * np.sin(rot_rad) + x_center for x, y in zip(x_vals, y_vals)]
            y_rotated = [(x - x_center) * np.sin(rot_rad) + (y - y_center) * np.cos(rot_rad) + y_center for x, y in zip(x_vals, y_vals)]

            rgb_color = f'rgb({int(obstacle.COLOR[0]*255)},{int(obstacle.COLOR[1]*255)},{int(obstacle.COLOR[2]*255)})'
            fig.add_trace(go.Scatter(x=x_rotated, y=y_rotated, mode='lines', fill='toself', name=obstacle_name, line=dict(color=rgb_color)))

        fig.update_layout(
            xaxis=dict(title="X Axis", scaleanchor="y", scaleratio=1),
            yaxis=dict(title="Y Axis"),
            title="Top View of Environment",
            showlegend=True,
            plot_bgcolor="#FFFFFF",  
            paper_bgcolor="#FFFFFF", 
            font=dict(color="#000000"),
            margin=dict(l=50, r=50, t=50, b=50),
            title_font=dict(size=20),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
        )

        return fig


    def run_server(self):
        self._app.run_server(debug=True, use_reloader=False)
