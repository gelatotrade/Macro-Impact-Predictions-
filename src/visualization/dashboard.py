"""
Real-Time Macro Event Impact Dashboard

Interactive dashboard for viewing predicted market impacts
of upcoming macroeconomic events.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any

import pandas as pd
import dash
from dash import dcc, html, callback, Input, Output, State
import dash_bootstrap_components as dbc
from loguru import logger

from ..models.prediction_engine import PredictionEngine, EventPrediction
from ..data.economic_calendar import EventImpact
from .market_charts import MarketCharts


class MacroDashboard:
    """Interactive dashboard for macro event predictions."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8050):
        """Initialize the dashboard.

        Args:
            host: Host address
            port: Port number
        """
        self.host = host
        self.port = port
        self.prediction_engine = PredictionEngine()
        self.charts = MarketCharts()

        # Initialize Dash app
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[dbc.themes.DARKLY],
            title="Macro Event Impact Predictor"
        )

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self) -> None:
        """Set up the dashboard layout."""
        self.app.layout = dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Macro Event Impact Predictor",
                            className="text-center mb-2 mt-3"),
                    html.H5("Real-Time Market-Implied Predictions",
                            className="text-center text-muted mb-4"),
                ])
            ]),

            # Market Expectations Summary
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market Expectations (Live)"),
                        dbc.CardBody(id="market-expectations-body")
                    ])
                ], width=12)
            ], className="mb-4"),

            # Controls
            dbc.Row([
                dbc.Col([
                    dbc.Label("Days Ahead"),
                    dcc.Dropdown(
                        id='days-dropdown',
                        options=[
                            {'label': '3 Days', 'value': 3},
                            {'label': '7 Days', 'value': 7},
                            {'label': '14 Days', 'value': 14},
                            {'label': '30 Days', 'value': 30}
                        ],
                        value=7,
                        clearable=False
                    )
                ], width=2),
                dbc.Col([
                    dbc.Label("Minimum Impact"),
                    dcc.Dropdown(
                        id='impact-dropdown',
                        options=[
                            {'label': 'All Events', 'value': 1},
                            {'label': 'Medium+', 'value': 2},
                            {'label': 'High+', 'value': 3},
                            {'label': 'Critical Only', 'value': 4}
                        ],
                        value=3,
                        clearable=False
                    )
                ], width=2),
                dbc.Col([
                    dbc.Button(
                        "Refresh Predictions",
                        id="refresh-button",
                        color="primary",
                        className="mt-4"
                    )
                ], width=2),
                dbc.Col([
                    html.Div(id="last-update", className="mt-4 text-muted")
                ], width=6)
            ], className="mb-4"),

            # Upcoming Events Table
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Upcoming High-Impact Events"),
                        dbc.CardBody([
                            html.Div(id="events-table")
                        ])
                    ])
                ])
            ], className="mb-4"),

            # Main Prediction Chart
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Predicted Market Impact"),
                        dbc.CardBody([
                            dcc.Loading(
                                dcc.Graph(id="multi-event-chart"),
                                type="circle"
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),

            # Event Detail Section
            dbc.Row([
                dbc.Col([
                    dbc.Label("Select Event for Detailed Analysis"),
                    dcc.Dropdown(
                        id='event-dropdown',
                        placeholder="Select an event..."
                    )
                ], width=6)
            ], className="mb-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Event Prediction Detail"),
                        dbc.CardBody([
                            dcc.Loading(
                                dcc.Graph(id="event-detail-chart"),
                                type="circle"
                            )
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Prediction Summary"),
                        dbc.CardBody(id="prediction-summary")
                    ])
                ], width=4)
            ], className="mb-4"),

            # Scenario Analysis
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Scenario Analysis"),
                        dbc.CardBody([
                            dcc.Loading(
                                dcc.Graph(id="scenario-chart"),
                                type="circle"
                            )
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Expected Move Distribution (SPY)"),
                        dbc.CardBody([
                            dcc.Loading(
                                dcc.Graph(id="distribution-chart"),
                                type="circle"
                            )
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),

            # Market Expectations Dashboard
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Market-Implied Expectations Dashboard"),
                        dbc.CardBody([
                            dcc.Loading(
                                dcc.Graph(id="expectations-dashboard"),
                                type="circle"
                            )
                        ])
                    ])
                ])
            ], className="mb-4"),

            # Store for predictions data
            dcc.Store(id='predictions-store'),
            dcc.Store(id='market-exp-store'),

            # Auto-refresh interval
            dcc.Interval(
                id='auto-refresh',
                interval=60*1000,  # 1 minute
                n_intervals=0
            ),

            # Footer
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(
                        "Predictions derived from VIX, yield curve, Fed Funds Futures, and TIPS spreads. "
                        "Not financial advice.",
                        className="text-center text-muted small"
                    )
                ])
            ])

        ], fluid=True)

    def _setup_callbacks(self) -> None:
        """Set up dashboard callbacks."""

        @self.app.callback(
            [Output('predictions-store', 'data'),
             Output('market-exp-store', 'data'),
             Output('last-update', 'children'),
             Output('event-dropdown', 'options')],
            [Input('refresh-button', 'n_clicks'),
             Input('auto-refresh', 'n_intervals')],
            [State('days-dropdown', 'value'),
             State('impact-dropdown', 'value')]
        )
        def refresh_predictions(n_clicks, n_intervals, days, min_impact):
            """Refresh predictions from engine."""
            logger.info("Refreshing predictions...")

            impact_map = {1: EventImpact.LOW, 2: EventImpact.MEDIUM, 3: EventImpact.HIGH, 4: EventImpact.CRITICAL}

            predictions = self.prediction_engine.get_upcoming_predictions(
                days_ahead=days,
                min_impact=impact_map.get(min_impact, EventImpact.HIGH)
            )

            market_exp = self.prediction_engine.get_market_implied_expectations()

            predictions_data = [p.to_dict() for p in predictions]
            event_options = [
                {'label': f"{p.event.event_name} ({p.event.date.strftime('%m/%d %H:%M')})",
                 'value': i}
                for i, p in enumerate(predictions)
            ]

            last_update = f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

            return predictions_data, market_exp, last_update, event_options

        @self.app.callback(
            Output('market-expectations-body', 'children'),
            Input('market-exp-store', 'data')
        )
        def update_market_expectations(market_exp):
            """Update market expectations summary."""
            if not market_exp:
                return "Loading..."

            vix = market_exp.get('vix_current', 'N/A')
            daily_move = market_exp.get('daily_expected_move_spx', 'N/A')
            fed_exp = market_exp.get('fed_next_meeting', 'N/A')
            prob_cut = market_exp.get('probability_cut', 0) * 100
            regime = market_exp.get('risk_regime', 'N/A')

            return dbc.Row([
                dbc.Col([
                    html.H4(f"{vix:.1f}" if isinstance(vix, (int, float)) else vix),
                    html.P("VIX", className="text-muted small")
                ], className="text-center"),
                dbc.Col([
                    html.H4(f"±{daily_move:.2f}%" if isinstance(daily_move, (int, float)) else daily_move),
                    html.P("Daily SPX Move", className="text-muted small")
                ], className="text-center"),
                dbc.Col([
                    html.H4(fed_exp.title().replace('_', ' ')),
                    html.P("Fed Expectation", className="text-muted small")
                ], className="text-center"),
                dbc.Col([
                    html.H4(f"{prob_cut:.0f}%"),
                    html.P("Prob. of Cut", className="text-muted small")
                ], className="text-center"),
                dbc.Col([
                    html.H4(regime.title().replace('_', ' ')),
                    html.P("Risk Regime", className="text-muted small")
                ], className="text-center"),
            ])

        @self.app.callback(
            Output('events-table', 'children'),
            Input('predictions-store', 'data')
        )
        def update_events_table(predictions_data):
            """Update events table."""
            if not predictions_data:
                return "No upcoming events"

            rows = []
            for pred in predictions_data:
                event = pred['event']
                spy_move = pred['expected_moves'].get('SPY', {})

                rows.append(html.Tr([
                    html.Td(event['date'][:16].replace('T', ' ')),
                    html.Td(event['event_name']),
                    html.Td(event['impact']),
                    html.Td(f"{event.get('consensus', 'N/A')} {event.get('unit', '')}"),
                    html.Td(f"±{spy_move.get('expected_move_pct', 0):.2f}%",
                            style={'color': '#4CAF50' if spy_move.get('probability_up', 0.5) > 0.5 else '#ff4444'}),
                    html.Td(f"{spy_move.get('probability_up', 0.5)*100:.0f}%"),
                ]))

            return dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Date/Time"),
                    html.Th("Event"),
                    html.Th("Impact"),
                    html.Th("Consensus"),
                    html.Th("SPY Expected"),
                    html.Th("P(Up)")
                ])),
                html.Tbody(rows)
            ], bordered=True, hover=True, responsive=True, striped=True)

        @self.app.callback(
            Output('multi-event-chart', 'figure'),
            Input('predictions-store', 'data')
        )
        def update_multi_event_chart(predictions_data):
            """Update multi-event prediction chart."""
            if not predictions_data:
                return self.charts._create_empty_chart("No predictions available")

            # Reconstruct predictions (simplified)
            predictions = []
            for p_data in predictions_data:
                pred = self._reconstruct_prediction(p_data)
                if pred:
                    predictions.append(pred)

            return self.charts.create_multi_event_prediction_chart(predictions)

        @self.app.callback(
            [Output('event-detail-chart', 'figure'),
             Output('prediction-summary', 'children'),
             Output('scenario-chart', 'figure'),
             Output('distribution-chart', 'figure')],
            [Input('event-dropdown', 'value')],
            [State('predictions-store', 'data')]
        )
        def update_event_detail(event_index, predictions_data):
            """Update event detail charts."""
            empty_fig = self.charts._create_empty_chart("Select an event")

            if event_index is None or not predictions_data:
                return empty_fig, "Select an event to view details", empty_fig, empty_fig

            if event_index >= len(predictions_data):
                return empty_fig, "Event not found", empty_fig, empty_fig

            p_data = predictions_data[event_index]
            pred = self._reconstruct_prediction(p_data)

            if not pred:
                return empty_fig, "Error loading prediction", empty_fig, empty_fig

            # Main chart
            main_chart = self.charts.create_expected_moves_chart(pred)

            # Summary
            summary = self._create_summary_html(pred)

            # Scenario chart
            scenario_chart = self.charts.create_scenario_comparison_chart(pred)

            # Distribution chart
            dist_chart = self.charts.create_expected_move_distribution_chart(pred, 'SPY')

            return main_chart, summary, scenario_chart, dist_chart

        @self.app.callback(
            Output('expectations-dashboard', 'figure'),
            Input('market-exp-store', 'data')
        )
        def update_expectations_dashboard(market_exp):
            """Update market expectations dashboard."""
            if not market_exp:
                return self.charts._create_empty_chart("Loading market data...")

            return self.charts.create_market_expectations_dashboard(market_exp)

    def _reconstruct_prediction(self, p_data: Dict) -> Optional[EventPrediction]:
        """Reconstruct EventPrediction from serialized data."""
        from ..data.economic_calendar import EconomicEvent, EventImpact
        from ..models.prediction_engine import ExpectedMove, PredictionConfidence

        try:
            # Reconstruct event
            e = p_data['event']
            event = EconomicEvent(
                date=datetime.fromisoformat(e['date']),
                time=e.get('time'),
                country=e.get('country', 'US'),
                event_name=e['event_name'],
                impact=EventImpact[e['impact']],
                previous=e.get('previous'),
                consensus=e.get('consensus'),
                unit=e.get('unit', ''),
                event_type=e.get('event_type', ''),
                affects=e.get('affects', [])
            )

            # Reconstruct expected moves
            moves = {}
            for symbol, m in p_data['expected_moves'].items():
                moves[symbol] = ExpectedMove(
                    symbol=m['symbol'],
                    name=m['name'],
                    expected_move_pct=m['expected_move_pct'],
                    move_1sd_pct=m['move_1sd_pct'],
                    move_2sd_pct=m['move_2sd_pct'],
                    direction=m['direction'],
                    probability_up=m['probability_up'],
                    probability_down=m['probability_down'],
                    confidence=PredictionConfidence[m['confidence'].upper()],
                    drivers=m.get('drivers', [])
                )

            return EventPrediction(
                event=event,
                timestamp=datetime.fromisoformat(p_data['timestamp']),
                expected_moves=moves,
                market_expectations=p_data['market_expectations'],
                scenario_analysis=p_data['scenario_analysis'],
                risk_assessment=p_data['risk_assessment'],
                summary=p_data['summary']
            )

        except Exception as e:
            logger.error(f"Error reconstructing prediction: {e}")
            return None

    def _create_summary_html(self, pred: EventPrediction) -> html.Div:
        """Create summary HTML for prediction."""
        spy = pred.expected_moves.get('SPY')
        risk = pred.risk_assessment

        items = [
            html.H5(pred.event.event_name, className="mb-3"),
            html.P([
                html.Strong("Date: "),
                pred.event.date.strftime('%Y-%m-%d %H:%M ET')
            ]),
            html.P([
                html.Strong("Consensus: "),
                f"{pred.event.consensus} {pred.event.unit}" if pred.event.consensus else "N/A"
            ]),
            html.Hr(),
            html.H6("Expected Moves:", className="mt-3"),
        ]

        for symbol in ['SPY', 'QQQ', 'TLT', 'DXY']:
            move = pred.expected_moves.get(symbol)
            if move:
                color = "#4CAF50" if move.probability_up > 0.5 else "#ff4444"
                items.append(html.P([
                    html.Strong(f"{symbol}: "),
                    html.Span(f"±{move.expected_move_pct:.2f}%", style={'color': color}),
                    f" (P↑: {move.probability_up*100:.0f}%)"
                ]))

        items.extend([
            html.Hr(),
            html.P([
                html.Strong("Risk Level: "),
                html.Span(
                    risk.get('risk_level', 'N/A').upper(),
                    className=f"badge bg-{'danger' if risk.get('overall_risk', 0) > 6 else 'warning' if risk.get('overall_risk', 0) > 3 else 'success'}"
                )
            ]),
            html.P([
                html.Strong("Confidence: "),
                spy.confidence.value.title() if spy else "N/A"
            ]),
        ])

        # Add drivers
        if spy and spy.drivers:
            items.append(html.Hr())
            items.append(html.H6("Key Drivers:"))
            for driver in spy.drivers[:3]:
                items.append(html.P(f"• {driver}", className="small text-muted"))

        return html.Div(items)

    def run(self, debug: bool = True) -> None:
        """Run the dashboard server.

        Args:
            debug: Enable debug mode
        """
        logger.info(f"Starting dashboard on http://{self.host}:{self.port}")
        self.app.run_server(host=self.host, port=self.port, debug=debug)
