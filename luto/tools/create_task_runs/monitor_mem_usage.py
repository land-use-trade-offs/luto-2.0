import os, sys
import pandas as pd
import plotly.graph_objects as go
import socket

import luto.settings as settings

from math import ceil
from plotly.subplots import make_subplots
from dash import Dash, dcc, html
from dash.dependencies import Input, Output


def read_data(log_path):
    if not os.path.exists(log_path):
        print(f"File {log_path} does not exist.")
        return pd.DataFrame(columns=['timestamp', 'memory_usage_GB'])
    data = pd.read_csv(log_path, sep='\t', header=None, names=['timestamp', 'memory_usage_GB'])
    data['timestamp'] = pd.to_datetime(data['timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    data['memory_usage_GB'] = pd.to_numeric(data['memory_usage_GB'], errors='coerce')
    data = data.dropna()
    return data

def save_plot(fig, output_path):
    fig.write_image(output_path)

def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def start_app():
    app = Dash(__name__)
    
    port = find_free_port()  # Move this line here

    app.layout = html.Div([
        dcc.Graph(id='live-update-graph'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000,  # in milliseconds
            n_intervals=0
        )
    ])

    @app.callback(Output('live-update-graph', 'figure'), Input('interval-component', 'n_intervals'))
    def update_graph_live(n):
        fig = go.Figure()
        res = settings.RESFACTOR
        mode = settings.MODE
        mem_data_path = f"{settings.OUTPUT_DIR}/RES_{res}_{mode}_mem_log.txt"
        output_image_path = f"{settings.OUTPUT_DIR}/RES_{res}_{mode}_mem_log.png"
        data = read_data(mem_data_path)
        if not data.empty:
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'], 
                    y=data['memory_usage_GB'], 
                    mode='lines', 
                    name=f'Memory Usage ({mem_data_path})'
                )
            )
            fig.update_xaxes(title_text='Timestamp')
            fig.update_yaxes(title_text='Memory Usage (GB)')
            fig.update_layout(title=f'Memory Usage  (RES {res} / {mode})')
            save_plot(fig, output_image_path)
        else:
            fig.add_annotation(text=f'No data available for {mem_data_path}', xref='paper', yref='paper', showarrow=False, font=dict(size=20))
        return fig
        
    print(f"Starting app on port {port}")
    app.run(debug=True, port=port)


