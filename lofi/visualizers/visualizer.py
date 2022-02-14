import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import State, Input, Output, ALL, MATCH
import plotly.express as px
import os
import pandas as pd
from ..cluster import cluster
import logging

class Visualizer():
    @cluster.on_master
    def __init__(self, name=""):

        self.name = name
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        self.app = dash.Dash(external_stylesheets=external_stylesheets) 

        # visual components
        self.displays = []
        self.tabs = []
        self.figures = []

        # trigers
        self.figure_interval = 1000
        self.display_interval = 500

    @cluster.on_master
    def build_app(self):
        self.set_layout()
        self.set_callbacks()

    @cluster.on_master
    def start_app(self):
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        thread = cluster.threading.Thread(target = self.app.run_server,
                                          daemon=True)
        thread.start()

    @cluster.on_master
    def debug_mode(self):
        self.app.run_server(debug=True)

    def display_recipe(self, none=None):
        if cluster.global_rank == 0:
            def decorator(f):
                self.displays.append(f)
        else:
            def decorator(f):
                pass
        return decorator

    def figure_recipe(self, name="Unnamed"):
        if cluster.global_rank == 0:
            def decorator(f):
                self.tabs.append(graph_tab(name, len(self.tabs)+1))
                self.figures.append(f)
        else:
            def decorator(f):
                pass
        return decorator


    def set_layout(self):
        self.app.layout = html.Div([
            html.H1(self.name, style={'text-align': 'center'}),
            html.Div(
                id='main-menu',
                children=[
                    menu_button('Refresh', 'single-refresh-btn'),
                    menu_button('Automatic Refreshing', 'auto-refresh-btn')
                ],
                style=main_menu
            ),
            html.Div(
                id='display-bar',
                children=[],
                style=display_bar
            ),
            html.Div(
                dcc.Tabs(
                    id="tabs",
                    value=0,
                    parent_className='custom-tabs',
                    className='custom-tabs-container',
                    children=self.tabs,
                ),
            ),
            html.Div(
                id='tabs-content',
                style={
                    'width':'100%',
                    'height':'70vh',
                },
            ),
            interval_trigger('displays', self.display_interval),
            interval_trigger('figures', self.figure_interval),
        ],
        style={'height':'90vh'}
        )

    @cluster.on_master
    def set_callbacks(self):

        # AUTOMATIC REFRESHING - BUTTON COLOR
        @self.app.callback(
            Output('auto-refresh-btn','style'),
            Input('auto-refresh-btn','n_clicks'),
        )
        def toogle_auto_refresh(n_clicks):
            if n_clicks % 2 == 1:
                return {'background-color': 'red'}
            else:
                return {'background-color': 'green'}

        # ENABLE/DISABLE ALL INTERVAL TRIGGERS
        @self.app.callback(
            Output({'type':'refresh-trigger','trigers': ALL},'disabled'),
            Input('auto-refresh-btn','n_clicks'),
            State({'type':'refresh-trigger','trigers': ALL}, 'interval'),
        )
        def toogle_auto_refresh(n_clicks, intervals):
            if n_clicks % 2 == 1:
                return [True]*len(intervals)
            else:
                return [False]*len(intervals)

        # UPDATE ALL DISPLAYS
        @self.app.callback(
            Output('display-bar', 'children'),
            Input({'type': 'refresh-trigger', 'trigers': 'displays'},'n_intervals'),
            Input('single-refresh-btn','n_clicks'),
        )
        def update_displays(n_intervals, n_clicks):
            children = []
            for recipe in self.displays:
                for item in recipe().items():
                    children.append(display(item[0], item[1]))
            return children

        @self.app.callback(
            Output('tabs-content', 'children'),
            Input({'type': 'refresh-trigger', 'trigers': 'figures'},'n_intervals'),
            Input('single-refresh-btn','n_clicks'),
            Input('tabs', 'value')
        )
        def update_figures(n_intervals, n_clicks, value):
            div = dcc.Graph(
                figure=self.figures[value-1](),
                config={'responsive':True},
                style={'width': '100%', 'height': '100%'},
            )
            return div

# BLOCKS AND CSS
def menu_button(text, id=None, style=None):
    if style is None:
        b = html.Button(text, id=id, n_clicks=0)
    else:
        b = html.Button(text, id=id, n_clicks=0, style=style)
    return b

def display(label, value):
    div = html.Div(
        children=[
            html.H5(
                children=label),
            html.H1(
                id={'type':'display-value','label':label},
                children=f'{value}')],
        style={'width': 240,
               'margin': '1px',
               'display': 'inline-block',
               'text-align': 'center',
               'border': '2px solid',
               'border-color': '#bbb',
               'border-radius': 10})
    return div

def interval_trigger(triggers, interval):
    trigger=dcc.Interval(
        id={'type':'refresh-trigger','trigers':triggers},
        interval=interval,
        n_intervals=1)
    return trigger

def graph_tab(label, value):
    tab = dcc.Tab(
        label=label,
        value=value,
        className='custom-tab',
        selected_className='custom-tab--selected',
        style={
            'height':'80px'
        }
    )
    return tab


def graph_div(index, span=[1,1]):
    div = html.Div(className=f'item',
        children=dcc.Graph(
                id={'type': 'live-graph', 'index':index},
                config={'responsive':True},
                style={'width': '100%', 'height': '100%'}),
        style={'grid-column':f'span {span[0]}',
               'overflow':'hidden',
               'grid-row':f'span {span[1]}',
               'border':'2px solid',
               'border-color': '#bbb',
               'border-radius': 20,
               }
    )
    return div

lofi_logo = html.Img(
    src="https://raw.githubusercontent.com/LiborKudela/lofi/master/logo.svg",
    style={'width':200},
    title="Github repository"
)

logo_div = html.Div(
    html.A(
        children=lofi_logo,
        href='https://github.com/LiborKudela/lofi',
        target="_blank",
        ),
    style={
        'position':'absolute',
        'top':'0',
        'padding':'25px',
    }
)

main_menu={
    'width':'100%',
    'margin-top':'40px',
    'text-align':'center'
}

display_bar={
    'width':'100%',
    'marginTop': 25,
    'marginBottom': 25,
    'display':'inline-block',
    'vertical-align':'top',
    'text-align':'center'
}



