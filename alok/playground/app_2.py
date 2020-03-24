import json
import os
import dash
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import dash_core_components as dcc
import dash_html_components as html
from aquabyte.data_access_utils import RDSAccessUtils
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

rds_access_utils = RDSAccessUtils(json.load(open(os.environ['PROD_SQL_CREDENTIALS'])))
query = """
    select * from keypoint_annotations where pen_id=61 and captured_at between '2019-11-27' and '2019-12-01';
"""
df = rds_access_utils.extract_from_database(query)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Dropdown(
                id='dropdown',
                options=[{'label': i, 'value': i} for i in ['2019-11-27', '2019-11-28']],
                value='a'

    ),
    html.Div(id='output'),
    dcc.Graph(
        id='x-time-series'
    ),
    dcc.Graph(
        id='time-series'
    )

])

@app.callback(Output('output', 'children'),
                            [Input('dropdown', 'value')])
def update_output_1(value):
        # Safely reassign the filter to a new variable
    filtered_df = df[(df.captured_at > value)]
    return filtered_df.shape[0]

def create_ts(vals):
    return {
        'data': [dict(
                        x=list(range(len(vals))),
                        y=vals,
                        mode='lines+markers'

        )],
        'layout': {
                        'height': 300,
                        'width': 1000,
                        'margin': {'l': 100, 'b': 30, 'r': 100, 't': 10},
            'annotations': [{
                                'x': 0, 'y': 0.85, 'xanchor': 'left', 'yanchor': 'bottom',
                                'xref': 'paper', 'yref': 'paper', 'showarrow': False,
                                'align': 'left', 'bgcolor': 'rgba(255, 255, 255, 0.5)',
                                'text': 'Hello'

            }],
                        'yaxis': {'type': 'linear'},
                        'xaxis': {'showgrid': True}

        }

    }

def create_ts_2(df):
    return {
        'data': [dict(
            x = pd.to_datetime(df.captured_at),
            y = list(range(df.shape[0])),
            name = 'Total crops'
        ),
                 dict(
                     x = pd.to_datetime(df.captured_at),
                     y = np.cumsum((df.is_skipped==False).astype(int)),
                     name = 'Total accepted',
                     marker=dict(color='rgb(28, 118, 255)')

                 )

        ],
        'layout': dict(
                        title='Data Progression over Time',
                        showlegend=True,
            legend=dict(
                                x=0,
                                y=1.0

            ),
                        margin=dict(l=40, r=0, t=40, b=30)

        )
    }

@app.callback(Output('x-time-series', 'figure'),
              [Input('dropdown', 'value')])
def create_graph(value):
    end_date = datetime.strftime(datetime.strptime(value, '%Y-%m-%d') + timedelta(days=1), '%Y-%m-%d')
    filtered_df = df[(df.captured_at > value) & (df.captured_at < end_date)]
    is_submitted = (filtered_df.is_skipped == False).astype(int)
    vals = np.cumsum(is_submitted)
    return create_ts(vals)

@app.callback(Output('time-series', 'figure'),
              [Input('dropdown', 'value')])
def create_graph_2(value):
    end_date = datetime.strftime(datetime.strptime(value, '%Y-%m-%d') + timedelta(days=1), '%Y-%m-%d')
    filtered_df = df[(df.captured_at > value) & (df.captured_at < end_date)]
    return create_ts_2(filtered_df)

if __name__ == '__main__':
    app.run_server(debug=True, port=9999, host='0.0.0.0')
