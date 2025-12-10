"""
Market Charts Module

Creates visualizations for PREDICTED/EXPECTED market moves based on
market-implied expectations. Shows anticipated reactions to upcoming
macro events.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from loguru import logger

from ..models.prediction_engine import EventPrediction, ExpectedMove
from ..data.economic_calendar import EconomicEvent, EventImpact


class MarketCharts:
    """Create visualizations for expected market moves."""

    # Color schemes
    COLORS = {
        'up': '#00C851',        # Green for positive
        'down': '#ff4444',      # Red for negative
        'neutral': '#ffbb33',   # Amber for neutral
        'primary': '#2196F3',   # Blue primary
        'secondary': '#9E9E9E', # Gray secondary
        'background': '#1a1a2e', # Dark background
        'grid': '#2d2d44',      # Grid lines
        'text': '#ffffff',      # White text
        'equity': '#4CAF50',    # Green for equity
        'bonds': '#2196F3',     # Blue for bonds
        'fx': '#FF9800',        # Orange for FX
        'commodity': '#9C27B0', # Purple for commodity
    }

    TEMPLATE = 'plotly_dark'

    def __init__(self, output_dir: str = "output"):
        """Initialize chart generator.

        Args:
            output_dir: Directory for saving charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def create_expected_moves_chart(
        self,
        prediction: EventPrediction,
        show_scenarios: bool = True
    ) -> go.Figure:
        """Create chart showing expected moves for an event.

        This is the KEY visualization showing PREDICTED moves, not historical.

        Args:
            prediction: Event prediction with expected moves
            show_scenarios: Whether to show scenario analysis

        Returns:
            Plotly figure
        """
        moves = prediction.expected_moves
        event = prediction.event

        # Sort by expected move magnitude
        sorted_symbols = sorted(
            moves.keys(),
            key=lambda x: abs(moves[x].expected_move_pct),
            reverse=True
        )

        symbols = sorted_symbols[:8]  # Top 8 instruments
        expected = [moves[s].expected_move_pct for s in symbols]
        move_1sd = [moves[s].move_1sd_pct for s in symbols]
        move_2sd = [moves[s].move_2sd_pct for s in symbols]
        prob_up = [moves[s].probability_up for s in symbols]
        names = [moves[s].name for s in symbols]

        # Create figure
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Expected Move by Instrument',
                'Move Probability Distribution',
                'Scenario Analysis',
                'Risk Assessment'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "bar"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )

        # 1. Expected Moves Bar Chart (main visualization)
        colors = [self.COLORS['up'] if p > 0.5 else self.COLORS['down'] if p < 0.5 else self.COLORS['neutral']
                  for p in prob_up]

        fig.add_trace(
            go.Bar(
                x=names,
                y=expected,
                name='Expected Move',
                marker_color=colors,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[m - e for m, e in zip(move_1sd, expected)],
                    arrayminus=[e - m for m, e in zip(move_1sd, expected)],
                    color='rgba(255,255,255,0.3)'
                ),
                text=[f"{e:+.2f}%" for e in expected],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>Expected: %{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # Add 2σ range as scatter points
        fig.add_trace(
            go.Scatter(
                x=names,
                y=move_2sd,
                mode='markers',
                name='2σ Move',
                marker=dict(symbol='line-ew', size=12, color='white', line_width=2),
                hovertemplate='2σ: ±%{y:.2f}%<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Probability Distribution
        fig.add_trace(
            go.Bar(
                x=names,
                y=[p * 100 for p in prob_up],
                name='Probability Up',
                marker_color=self.COLORS['up'],
                opacity=0.7,
                text=[f"{p*100:.0f}%" for p in prob_up],
                textposition='inside',
            ),
            row=1, col=2
        )

        fig.add_trace(
            go.Bar(
                x=names,
                y=[(1-p) * 100 for p in prob_up],
                name='Probability Down',
                marker_color=self.COLORS['down'],
                opacity=0.7,
                base=[p * 100 for p in prob_up],
            ),
            row=1, col=2
        )

        # 3. Scenario Analysis
        scenarios = prediction.scenario_analysis
        scenario_names = list(scenarios.keys())
        spy_scenarios = [scenarios[s]['moves'].get('SPY', 0) for s in scenario_names]

        scenario_colors = [
            self.COLORS['up'] if v > 0 else self.COLORS['down'] if v < 0 else self.COLORS['neutral']
            for v in spy_scenarios
        ]

        fig.add_trace(
            go.Bar(
                x=scenario_names,
                y=spy_scenarios,
                name='SPY Move by Scenario',
                marker_color=scenario_colors,
                text=[f"{v:+.2f}%" for v in spy_scenarios],
                textposition='outside',
            ),
            row=2, col=1
        )

        # 4. Risk Gauge
        risk = prediction.risk_assessment
        risk_score = risk.get('overall_risk', 5)

        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                title={'text': "Event Risk Score", 'font': {'size': 14}},
                gauge={
                    'axis': {'range': [0, 10], 'tickwidth': 1},
                    'bar': {'color': self._get_risk_color(risk_score)},
                    'steps': [
                        {'range': [0, 3], 'color': 'rgba(0, 200, 81, 0.3)'},
                        {'range': [3, 6], 'color': 'rgba(255, 187, 51, 0.3)'},
                        {'range': [6, 10], 'color': 'rgba(255, 68, 68, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 2},
                        'thickness': 0.75,
                        'value': risk_score
                    }
                }
            ),
            row=2, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"<b>PREDICTED Market Impact: {event.event_name}</b><br>"
                     f"<sup>Based on VIX, Yield Curve, and Fed Funds Futures | "
                     f"Event Date: {event.date.strftime('%Y-%m-%d %H:%M')} ET</sup>",
                x=0.5,
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            height=800,
            margin=dict(t=100, b=80)
        )

        # Update axes
        fig.update_yaxes(title_text="Expected Move (%)", row=1, col=1)
        fig.update_yaxes(title_text="Probability (%)", range=[0, 100], row=1, col=2)
        fig.update_yaxes(title_text="SPY Move (%)", row=2, col=1)

        return fig

    def create_multi_event_prediction_chart(
        self,
        predictions: List[EventPrediction]
    ) -> go.Figure:
        """Create chart showing predictions for multiple upcoming events.

        Args:
            predictions: List of event predictions

        Returns:
            Plotly figure
        """
        if not predictions:
            return self._create_empty_chart("No upcoming events")

        # Prepare data
        events = []
        spy_moves = []
        qqq_moves = []
        tlt_moves = []
        dxy_moves = []
        impacts = []
        dates = []

        for pred in predictions:
            events.append(pred.event.event_name)
            dates.append(pred.event.date)
            spy_moves.append(pred.expected_moves.get('SPY', ExpectedMove('SPY', '', 0, 0, 0, 'neutral', 0.5, 0.5, 'low')).expected_move_pct)
            qqq_moves.append(pred.expected_moves.get('QQQ', ExpectedMove('QQQ', '', 0, 0, 0, 'neutral', 0.5, 0.5, 'low')).expected_move_pct)
            tlt_moves.append(pred.expected_moves.get('TLT', ExpectedMove('TLT', '', 0, 0, 0, 'neutral', 0.5, 0.5, 'low')).expected_move_pct)
            dxy_moves.append(pred.expected_moves.get('DXY', ExpectedMove('DXY', '', 0, 0, 0, 'neutral', 0.5, 0.5, 'low')).expected_move_pct)
            impacts.append(pred.event.impact.value)

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                'Expected Moves by Event (Equities)',
                'Expected Moves by Event (Bonds & FX)'
            ),
            vertical_spacing=0.15,
            row_heights=[0.5, 0.5]
        )

        # Event names with dates
        event_labels = [f"{e}<br><sup>{d.strftime('%m/%d')}</sup>" for e, d in zip(events, dates)]

        # Equities
        fig.add_trace(
            go.Bar(name='SPY', x=event_labels, y=spy_moves, marker_color=self.COLORS['equity']),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(name='QQQ', x=event_labels, y=qqq_moves, marker_color='#81C784'),
            row=1, col=1
        )

        # Bonds & FX
        fig.add_trace(
            go.Bar(name='TLT (Bonds)', x=event_labels, y=tlt_moves, marker_color=self.COLORS['bonds']),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(name='DXY (USD)', x=event_labels, y=dxy_moves, marker_color=self.COLORS['fx']),
            row=2, col=1
        )

        fig.update_layout(
            title=dict(
                text="<b>Upcoming Events: PREDICTED Market Impact</b><br>"
                     "<sup>Derived from current implied volatility and market expectations</sup>",
                x=0.5,
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            barmode='group',
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.15,
                xanchor="center",
                x=0.5
            ),
            height=700
        )

        fig.update_yaxes(title_text="Expected Move (%)", row=1, col=1)
        fig.update_yaxes(title_text="Expected Move (%)", row=2, col=1)

        return fig

    def create_expected_move_distribution_chart(
        self,
        prediction: EventPrediction,
        instrument: str = 'SPY'
    ) -> go.Figure:
        """Create probability distribution chart for expected move.

        Args:
            prediction: Event prediction
            instrument: Instrument to show

        Returns:
            Plotly figure
        """
        move = prediction.expected_moves.get(instrument)
        if not move:
            return self._create_empty_chart(f"No prediction for {instrument}")

        # Generate distribution
        mean = move.expected_move_pct
        std = move.move_1sd_pct

        x = np.linspace(mean - 4*std, mean + 4*std, 200)
        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

        fig = go.Figure()

        # Add distribution curve
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode='lines',
                name='Expected Distribution',
                fill='tozeroy',
                line=dict(color=self.COLORS['primary'], width=2),
                fillcolor='rgba(33, 150, 243, 0.3)'
            )
        )

        # Add vertical lines for key levels
        fig.add_vline(x=mean, line_dash="solid", line_color="white", annotation_text=f"Expected: {mean:+.2f}%")
        fig.add_vline(x=mean + std, line_dash="dash", line_color=self.COLORS['up'], annotation_text=f"+1σ: {mean+std:+.2f}%")
        fig.add_vline(x=mean - std, line_dash="dash", line_color=self.COLORS['down'], annotation_text=f"-1σ: {mean-std:+.2f}%")
        fig.add_vline(x=mean + 2*std, line_dash="dot", line_color=self.COLORS['up'])
        fig.add_vline(x=mean - 2*std, line_dash="dot", line_color=self.COLORS['down'])
        fig.add_vline(x=0, line_dash="dash", line_color="gray")

        # Shade positive/negative regions
        positive_x = x[x > 0]
        positive_y = y[x > 0]
        if len(positive_x) > 0:
            fig.add_trace(
                go.Scatter(
                    x=positive_x,
                    y=positive_y,
                    fill='tozeroy',
                    mode='none',
                    fillcolor='rgba(0, 200, 81, 0.2)',
                    showlegend=False
                )
            )

        fig.update_layout(
            title=dict(
                text=f"<b>{instrument} Expected Move Distribution</b><br>"
                     f"<sup>{prediction.event.event_name} | "
                     f"P(Up)={move.probability_up*100:.0f}% | P(Down)={move.probability_down*100:.0f}%</sup>",
                x=0.5,
                font=dict(size=16)
            ),
            template=self.TEMPLATE,
            xaxis_title="Move (%)",
            yaxis_title="Probability Density",
            showlegend=True,
            height=500
        )

        return fig

    def create_market_expectations_dashboard(
        self,
        market_exp: Dict[str, Any]
    ) -> go.Figure:
        """Create dashboard showing current market expectations.

        Args:
            market_exp: Market expectations data

        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                'VIX & Implied Volatility',
                'Fed Rate Expectations',
                'Expected Daily Move',
                'Yield Curve',
                'Inflation Expectations',
                'Market Regime'
            ),
            specs=[
                [{"type": "indicator"}, {"type": "pie"}, {"type": "indicator"}],
                [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
            ],
            vertical_spacing=0.2,
            horizontal_spacing=0.1
        )

        # 1. VIX Gauge
        vix = market_exp.get('vix_current', 18)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=vix,
                title={'text': "VIX"},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': self._get_vix_color(vix)},
                    'steps': [
                        {'range': [0, 15], 'color': 'rgba(0, 200, 81, 0.3)'},
                        {'range': [15, 25], 'color': 'rgba(255, 187, 51, 0.3)'},
                        {'range': [25, 50], 'color': 'rgba(255, 68, 68, 0.3)'}
                    ]
                }
            ),
            row=1, col=1
        )

        # 2. Fed Rate Expectations Pie
        prob_cut = market_exp.get('probability_cut', 0.3)
        prob_hike = market_exp.get('probability_hike', 0.1)
        prob_hold = market_exp.get('probability_hold', 0.6)

        fig.add_trace(
            go.Pie(
                labels=['Cut', 'Hold', 'Hike'],
                values=[prob_cut * 100, prob_hold * 100, prob_hike * 100],
                marker_colors=[self.COLORS['up'], self.COLORS['neutral'], self.COLORS['down']],
                hole=0.4,
                textinfo='label+percent',
                textposition='inside'
            ),
            row=1, col=2
        )

        # 3. Expected Daily Move
        daily_move = market_exp.get('daily_expected_move_spx', 1.0)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=daily_move,
                title={'text': "SPX Daily Move (%)"},
                number={'suffix': '%', 'font': {'size': 40}},
                delta={'reference': 1.0, 'relative': True, 'position': "bottom"}
            ),
            row=1, col=3
        )

        # 4. Yield Spread
        spread = market_exp.get('spread_10y_3m', 0)
        fig.add_trace(
            go.Indicator(
                mode="number+delta",
                value=spread if spread else 0,
                title={'text': "10Y-3M Spread (%)"},
                number={'font': {'size': 40, 'color': self.COLORS['down'] if spread and spread < 0 else self.COLORS['up']}},
                delta={'reference': 0, 'position': "bottom"}
            ),
            row=2, col=1
        )

        # 5. Breakeven Inflation
        inflation = market_exp.get('breakeven_inflation', 2.5)
        fig.add_trace(
            go.Indicator(
                mode="number",
                value=inflation,
                title={'text': "Breakeven Inflation (%)"},
                number={'suffix': '%', 'font': {'size': 40}}
            ),
            row=2, col=2
        )

        # 6. Market Regime
        risk_regime = market_exp.get('risk_regime', 'neutral')
        regime_value = {'risk_on': 80, 'neutral': 50, 'risk_off': 20}.get(risk_regime, 50)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=regime_value,
                title={'text': f"Regime: {risk_regime.replace('_', ' ').title()}"},
                gauge={
                    'axis': {'range': [0, 100], 'visible': False},
                    'bar': {'color': self.COLORS['up'] if regime_value > 60 else self.COLORS['down'] if regime_value < 40 else self.COLORS['neutral']},
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(255, 68, 68, 0.3)'},
                        {'range': [40, 60], 'color': 'rgba(255, 187, 51, 0.3)'},
                        {'range': [60, 100], 'color': 'rgba(0, 200, 81, 0.3)'}
                    ]
                }
            ),
            row=2, col=3
        )

        fig.update_layout(
            title=dict(
                text="<b>Market-Implied Expectations Dashboard</b><br>"
                     f"<sup>Real-time data from VIX, Yield Curve, Fed Funds Futures | "
                     f"Updated: {market_exp.get('timestamp', 'N/A')}</sup>",
                x=0.5,
                font=dict(size=18)
            ),
            template=self.TEMPLATE,
            showlegend=False,
            height=600
        )

        return fig

    def create_scenario_comparison_chart(
        self,
        prediction: EventPrediction
    ) -> go.Figure:
        """Create chart comparing different scenario outcomes.

        Args:
            prediction: Event prediction

        Returns:
            Plotly figure
        """
        scenarios = prediction.scenario_analysis
        instruments = ['SPY', 'QQQ', 'TLT', 'DXY']

        fig = go.Figure()

        for scenario_name, scenario_data in scenarios.items():
            moves = scenario_data.get('moves', {})
            values = [moves.get(inst, 0) for inst in instruments]

            fig.add_trace(
                go.Bar(
                    name=scenario_name.replace('_', ' ').title(),
                    x=instruments,
                    y=values,
                    text=[f"{v:+.2f}%" for v in values],
                    textposition='outside'
                )
            )

        fig.update_layout(
            title=dict(
                text=f"<b>Scenario Analysis: {prediction.event.event_name}</b><br>"
                     "<sup>Expected moves under different outcome scenarios</sup>",
                x=0.5,
                font=dict(size=16)
            ),
            template=self.TEMPLATE,
            barmode='group',
            xaxis_title="Instrument",
            yaxis_title="Expected Move (%)",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=500
        )

        return fig

    def save_chart(self, fig: go.Figure, filename: str, format: str = 'html') -> Path:
        """Save chart to file.

        Args:
            fig: Plotly figure
            filename: Output filename (without extension)
            format: Output format ('html', 'png', 'svg')

        Returns:
            Path to saved file
        """
        output_path = self.output_dir / f"{filename}.{format}"

        if format == 'html':
            fig.write_html(str(output_path), include_plotlyjs=True)
        else:
            fig.write_image(str(output_path))

        logger.info(f"Chart saved to {output_path}")
        return output_path

    def _get_risk_color(self, risk_score: float) -> str:
        """Get color for risk score."""
        if risk_score < 3:
            return self.COLORS['up']
        elif risk_score < 6:
            return self.COLORS['neutral']
        else:
            return self.COLORS['down']

    def _get_vix_color(self, vix: float) -> str:
        """Get color for VIX level."""
        if vix < 15:
            return self.COLORS['up']
        elif vix < 25:
            return self.COLORS['neutral']
        else:
            return self.COLORS['down']

    def _create_empty_chart(self, message: str) -> go.Figure:
        """Create empty chart with message."""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=20, color="white")
        )
        fig.update_layout(
            template=self.TEMPLATE,
            height=400
        )
        return fig
