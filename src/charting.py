"""
Trade Genius Agent - Chart Generation
Creates beautiful charts for price trends, predictions, and analysis
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import base64

from loguru import logger
from src.data_fetcher import DataFetcher
from src.models import Trade, Prediction

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ChartGenerator:
    """Generates various types of charts for trading analysis"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.chart_style = {
            'figure_size': (15, 10),
            'dpi': 300,
            'facecolor': 'white',
            'edgecolor': 'black'
        }
    
    def create_price_chart(self, symbol: str, days: int = 90, 
                          include_predictions: bool = True,
                          include_trades: bool = True) -> str:
        """Create comprehensive price chart with predictions and trades"""
        try:
            # Get historical data
            data = self.data_fetcher.get_historical_data(symbol, f"{days}d")
            if data is None or data.empty:
                return self._create_error_chart("No data available")
            
            # Create figure with subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                subplot_titles=[
                    f'{symbol} - Price Chart with Predictions',
                    'Volume',
                    'Technical Indicators'
                ],
                row_heights=[0.6, 0.2, 0.2]
            )
            
            # Price chart
            fig.add_trace(
                go.Candlestick(
                    x=data.index,
                    open=data['Open'],
                    high=data['High'],
                    low=data['Low'],
                    close=data['Close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=1, col=1
            )
            
            # Moving averages
            if 'SMA_20' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_20'],
                        name='SMA 20',
                        line=dict(color='orange', width=2)
                    ),
                    row=1, col=1
                )
            
            if 'SMA_50' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['SMA_50'],
                        name='SMA 50',
                        line=dict(color='blue', width=2)
                    ),
                    row=1, col=1
                )
            
            # Bollinger Bands
            if all(col in data.columns for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Upper'],
                        name='BB Upper',
                        line=dict(color='gray', width=1, dash='dash'),
                        showlegend=False
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['BB_Lower'],
                        name='BB Lower',
                        line=dict(color='gray', width=1, dash='dash'),
                        fill='tonexty',
                        fillcolor='rgba(128,128,128,0.1)',
                        showlegend=False
                    ),
                    row=1, col=1
                )
            
            # Add predictions if available
            if include_predictions:
                predictions = self._get_predictions(symbol)
                if predictions:
                    pred_dates = [datetime.now() + timedelta(days=i) for i in range(1, 6)]
                    pred_prices = [p['predicted_price'] for p in predictions[:5]]
                    
                    fig.add_trace(
                        go.Scatter(
                            x=pred_dates,
                            y=pred_prices,
                            name='Predictions',
                            line=dict(color='red', width=3, dash='dot'),
                            mode='lines+markers'
                        ),
                        row=1, col=1
                    )
            
            # Add trade markers
            if include_trades:
                trades = self._get_trades(symbol)
                if trades:
                    buy_trades = [t for t in trades if t['trade_type'] == 'BUY']
                    sell_trades = [t for t in trades if t['trade_type'] == 'SELL']
                    
                    if buy_trades:
                        fig.add_trace(
                            go.Scatter(
                                x=[t['executed_at'] for t in buy_trades],
                                y=[t['price'] for t in buy_trades],
                                name='Buy Trades',
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-up',
                                    size=15,
                                    color='green'
                                )
                            ),
                            row=1, col=1
                        )
                    
                    if sell_trades:
                        fig.add_trace(
                            go.Scatter(
                                x=[t['executed_at'] for t in sell_trades],
                                y=[t['price'] for t in sell_trades],
                                name='Sell Trades',
                                mode='markers',
                                marker=dict(
                                    symbol='triangle-down',
                                    size=15,
                                    color='red'
                                )
                            ),
                            row=1, col=1
                        )
            
            # Volume chart
            fig.add_trace(
                go.Bar(
                    x=data.index,
                    y=data['Volume'],
                    name='Volume',
                    marker_color='lightblue',
                    opacity=0.7
                ),
                row=2, col=1
            )
            
            # Technical indicators
            if 'RSI' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data['RSI'],
                        name='RSI',
                        line=dict(color='purple', width=2)
                    ),
                    row=3, col=1
                )
                
                # RSI levels
                fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
                fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            
            # Update layout
            fig.update_layout(
                title=f'{symbol} - Comprehensive Analysis',
                xaxis_rangeslider_visible=False,
                height=800,
                showlegend=True,
                template='plotly_white'
            )
            
            # Convert to base64 string
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating price chart for {symbol}: {e}")
            return self._create_error_chart(str(e))
    
    def create_portfolio_chart(self, portfolio_data: Dict) -> str:
        """Create portfolio performance chart"""
        try:
            if not portfolio_data or 'positions' not in portfolio_data:
                return self._create_error_chart("No portfolio data available")
            
            positions = portfolio_data['positions']
            if not positions:
                return self._create_error_chart("No positions found")
            
            # Create pie chart for portfolio allocation
            symbols = [pos['symbol'] for pos in positions]
            values = [pos['current_price'] * pos['quantity'] for pos in positions]
            pnl_values = [pos['unrealized_pnl'] for pos in positions]
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Portfolio Allocation',
                    'P&L by Position',
                    'Portfolio Value Trend',
                    'Risk Metrics'
                ],
                specs=[[{"type": "pie"}, {"type": "bar"}],
                       [{"type": "scatter"}, {"type": "indicator"}]]
            )
            
            # Portfolio allocation pie chart
            fig.add_trace(
                go.Pie(
                    labels=symbols,
                    values=values,
                    name="Allocation"
                ),
                row=1, col=1
            )
            
            # P&L by position
            colors = ['green' if pnl >= 0 else 'red' for pnl in pnl_values]
            fig.add_trace(
                go.Bar(
                    x=symbols,
                    y=pnl_values,
                    name="P&L",
                    marker_color=colors
                ),
                row=1, col=2
            )
            
            # Portfolio value trend (simplified)
            dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                end=datetime.now(), freq='D')
            portfolio_values = [portfolio_data.get('total_value', 0)] * len(dates)
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=portfolio_values,
                    name="Portfolio Value",
                    line=dict(color='blue', width=2)
                ),
                row=2, col=1
            )
            
            # Risk metrics gauge
            total_pnl = portfolio_data.get('total_pnl', 0)
            total_value = portfolio_data.get('total_value', 1)
            pnl_percentage = (total_pnl / total_value) * 100 if total_value > 0 else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=pnl_percentage,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Portfolio P&L %"},
                    gauge={
                        'axis': {'range': [-20, 20]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-20, 0], 'color': "lightgray"},
                            {'range': [0, 20], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 10
                        }
                    }
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                title="Portfolio Dashboard",
                height=600,
                showlegend=True,
                template='plotly_white'
            )
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating portfolio chart: {e}")
            return self._create_error_chart(str(e))
    
    def create_sentiment_chart(self, symbol: str, days: int = 7) -> str:
        """Create sentiment analysis chart"""
        try:
            # Get news sentiment data
            news_items = self.data_fetcher.get_news_sentiment(symbol, days)
            if not news_items:
                return self._create_error_chart("No sentiment data available")
            
            # Prepare data
            dates = [item['published_at'] for item in news_items]
            sentiment_scores = [item['sentiment_score'] for item in news_items]
            headlines = [item['headline'][:50] + '...' if len(item['headline']) > 50 
                        else item['headline'] for item in news_items]
            
            # Create sentiment chart
            fig = go.Figure()
            
            # Add sentiment scores
            colors = ['red' if score < 0.4 else 'green' if score > 0.6 else 'orange' 
                     for score in sentiment_scores]
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=sentiment_scores,
                    mode='markers+lines',
                    name='Sentiment Score',
                    marker=dict(
                        size=10,
                        color=colors,
                        line=dict(width=2, color='black')
                    ),
                    text=headlines,
                    hovertemplate='<b>%{text}</b><br>' +
                                'Date: %{x}<br>' +
                                'Sentiment: %{y:.2f}<br>' +
                                '<extra></extra>'
                )
            )
            
            # Add sentiment zones
            fig.add_hline(y=0.6, line_dash="dash", line_color="green", 
                         annotation_text="Positive Zone")
            fig.add_hline(y=0.4, line_dash="dash", line_color="red", 
                         annotation_text="Negative Zone")
            fig.add_hline(y=0.5, line_dash="solid", line_color="gray", 
                         annotation_text="Neutral")
            
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiment_scores)
            
            fig.update_layout(
                title=f'{symbol} - News Sentiment Analysis (Avg: {avg_sentiment:.2f})',
                xaxis_title='Date',
                yaxis_title='Sentiment Score',
                yaxis=dict(range=[0, 1]),
                height=500,
                template='plotly_white'
            )
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating sentiment chart for {symbol}: {e}")
            return self._create_error_chart(str(e))
    
    def create_prediction_chart(self, symbol: str, predictions: List[Dict]) -> str:
        """Create prediction comparison chart"""
        try:
            if not predictions:
                return self._create_error_chart("No predictions available")
            
            # Get recent historical data
            data = self.data_fetcher.get_historical_data(symbol, "30d")
            if data is None or data.empty:
                return self._create_error_chart("No historical data available")
            
            fig = go.Figure()
            
            # Historical prices
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['Close'],
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Predictions
            pred_dates = [p['prediction_date'] for p in predictions]
            pred_prices = [p['predicted_price'] for p in predictions]
            confidences = [p['confidence_score'] for p in predictions]
            
            # Color code by confidence
            colors = ['red' if conf < 0.5 else 'orange' if conf < 0.7 else 'green' 
                     for conf in confidences]
            
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=pred_prices,
                    mode='markers+lines',
                    name='Predictions',
                    line=dict(color='red', width=2, dash='dot'),
                    marker=dict(
                        size=12,
                        color=colors,
                        line=dict(width=2, color='black')
                    ),
                    text=[f"Confidence: {conf:.2f}" for conf in confidences],
                    hovertemplate='<b>Prediction</b><br>' +
                                'Date: %{x}<br>' +
                                'Price: %{y:.2f}<br>' +
                                '%{text}<br>' +
                                '<extra></extra>'
                )
            )
            
            # Add confidence bands
            if len(predictions) > 1:
                upper_band = [p['predicted_price'] * 1.05 for p in predictions]
                lower_band = [p['predicted_price'] * 0.95 for p in predictions]
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=upper_band,
                        fill=None,
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        showlegend=False
                    )
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=lower_band,
                        fill='tonexty',
                        mode='lines',
                        line_color='rgba(0,0,0,0)',
                        name='Confidence Band',
                        fillcolor='rgba(255,0,0,0.1)'
                    )
                )
            
            fig.update_layout(
                title=f'{symbol} - Price Predictions',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500,
                template='plotly_white'
            )
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating prediction chart for {symbol}: {e}")
            return self._create_error_chart(str(e))
    
    def create_market_overview_chart(self, market_data: Dict) -> str:
        """Create market overview chart"""
        try:
            if not market_data or 'indices' not in market_data:
                return self._create_error_chart("No market data available")
            
            indices = market_data['indices']
            if not indices:
                return self._create_error_chart("No index data available")
            
            # Create subplots for different indices
            fig = make_subplots(
                rows=len(indices), cols=1,
                subplot_titles=list(indices.keys()),
                vertical_spacing=0.1
            )
            
            for i, (index_name, index_data) in enumerate(indices.items(), 1):
                if index_data:
                    # Create a simple line chart for each index
                    dates = pd.date_range(start=datetime.now() - timedelta(days=30), 
                                        end=datetime.now(), freq='D')
                    prices = [index_data['price']] * len(dates)
                    
                    fig.add_trace(
                        go.Scatter(
                            x=dates,
                            y=prices,
                            name=index_name,
                            line=dict(width=2)
                        ),
                        row=i, col=1
                    )
            
            fig.update_layout(
                title="Market Overview",
                height=200 * len(indices),
                template='plotly_white'
            )
            
            return self._fig_to_base64(fig)
            
        except Exception as e:
            logger.error(f"Error creating market overview chart: {e}")
            return self._create_error_chart(str(e))
    
    def _get_predictions(self, symbol: str) -> List[Dict]:
        """Get predictions for a symbol (mock data for now)"""
        # In a real implementation, this would fetch from database
        return []
    
    def _get_trades(self, symbol: str) -> List[Dict]:
        """Get trades for a symbol (mock data for now)"""
        # In a real implementation, this would fetch from database
        return []
    
    def _fig_to_base64(self, fig) -> str:
        """Convert plotly figure to base64 string"""
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            img_base64 = base64.b64encode(img_bytes).decode()
            return f"data:image/png;base64,{img_base64}"
        except Exception as e:
            logger.error(f"Error converting figure to base64: {e}")
            return self._create_error_chart("Chart generation failed")
    
    def _create_error_chart(self, error_message: str) -> str:
        """Create a simple error chart"""
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error: {error_message}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=400
        )
        return self._fig_to_base64(fig)
    
    def save_chart(self, chart_base64: str, filename: str) -> str:
        """Save chart to file"""
        try:
            # Remove data URL prefix
            if chart_base64.startswith('data:image/png;base64,'):
                chart_base64 = chart_base64.split(',')[1]
            
            # Decode and save
            img_data = base64.b64decode(chart_base64)
            filepath = f"charts/{filename}"
            
            with open(filepath, 'wb') as f:
                f.write(img_data)
            
            return filepath
            
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
            return ""
