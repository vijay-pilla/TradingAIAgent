"""
Trade Genius Agent - Trading Logic
Handles automated trade execution and risk management
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger
from dataclasses import dataclass
from enum import Enum

from src.config import settings, RISK_PARAMS
from src.data_fetcher import DataFetcher
from src.models import Trade, Stock, Alert

class TradeType(Enum):
    BUY = "BUY"
    SELL = "SELL"

class TradeStatus(Enum):
    PENDING = "PENDING"
    EXECUTED = "EXECUTED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

@dataclass
class TradeSignal:
    """Represents a trading signal"""
    symbol: str
    action: TradeType
    quantity: int
    price: float
    confidence: float
    reason: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    entry_time: datetime

class TradingEngine:
    """Main trading engine that handles all trading logic"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.positions: Dict[str, Position] = {}
        self.pending_orders: List[TradeSignal] = []
        self.daily_pnl = 0.0
        self.max_daily_loss = RISK_PARAMS["max_daily_loss"]
        
    def analyze_and_generate_signals(self, symbol: str) -> List[TradeSignal]:
        """Analyze stock and generate trading signals"""
        signals = []
        
        try:
            # Get current price and historical data
            current_data = self.data_fetcher.get_live_price(symbol)
            if not current_data:
                return signals
            
            historical_data = self.data_fetcher.get_historical_data(symbol, "3mo")
            if historical_data is None or historical_data.empty:
                return signals
            
            # Technical analysis signals
            tech_signals = self._technical_analysis(symbol, current_data, historical_data)
            signals.extend(tech_signals)
            
            # Sentiment analysis signals
            sentiment_signals = self._sentiment_analysis(symbol, current_data)
            signals.extend(sentiment_signals)
            
            # Risk management signals
            risk_signals = self._risk_management_analysis(symbol, current_data)
            signals.extend(risk_signals)
            
            # Combine and filter signals
            final_signals = self._combine_signals(signals)
            
            return final_signals
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return signals
    
    def _technical_analysis(self, symbol: str, current_data: Dict, historical_data: pd.DataFrame) -> List[TradeSignal]:
        """Generate signals based on technical analysis"""
        signals = []
        
        try:
            current_price = current_data['price']
            latest = historical_data.iloc[-1]
            
            # Moving Average Crossovers
            if 'SMA_20' in latest and 'SMA_50' in latest:
                if latest['SMA_20'] > latest['SMA_50'] and historical_data.iloc[-2]['SMA_20'] <= historical_data.iloc[-2]['SMA_50']:
                    # Golden cross - bullish signal
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.BUY,
                        quantity=self._calculate_position_size(symbol, current_price),
                        price=current_price,
                        confidence=0.7,
                        reason="Golden Cross (SMA 20 > SMA 50)"
                    ))
                elif latest['SMA_20'] < latest['SMA_50'] and historical_data.iloc[-2]['SMA_20'] >= historical_data.iloc[-2]['SMA_50']:
                    # Death cross - bearish signal
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.SELL,
                        quantity=self._get_position_quantity(symbol),
                        price=current_price,
                        confidence=0.7,
                        reason="Death Cross (SMA 20 < SMA 50)"
                    ))
            
            # RSI Overbought/Oversold
            if 'RSI' in latest:
                if latest['RSI'] < 30:  # Oversold
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.BUY,
                        quantity=self._calculate_position_size(symbol, current_price),
                        price=current_price,
                        confidence=0.6,
                        reason=f"RSI Oversold ({latest['RSI']:.1f})"
                    ))
                elif latest['RSI'] > 70:  # Overbought
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.SELL,
                        quantity=self._get_position_quantity(symbol),
                        price=current_price,
                        confidence=0.6,
                        reason=f"RSI Overbought ({latest['RSI']:.1f})"
                    ))
            
            # MACD Signal
            if 'MACD' in latest and 'MACD_Signal' in latest:
                if latest['MACD'] > latest['MACD_Signal'] and historical_data.iloc[-2]['MACD'] <= historical_data.iloc[-2]['MACD_Signal']:
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.BUY,
                        quantity=self._calculate_position_size(symbol, current_price),
                        price=current_price,
                        confidence=0.65,
                        reason="MACD Bullish Crossover"
                    ))
                elif latest['MACD'] < latest['MACD_Signal'] and historical_data.iloc[-2]['MACD'] >= historical_data.iloc[-2]['MACD_Signal']:
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.SELL,
                        quantity=self._get_position_quantity(symbol),
                        price=current_price,
                        confidence=0.65,
                        reason="MACD Bearish Crossover"
                    ))
            
            # Bollinger Bands
            if all(col in latest for col in ['BB_Upper', 'BB_Lower', 'BB_Middle']):
                if current_price <= latest['BB_Lower']:
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.BUY,
                        quantity=self._calculate_position_size(symbol, current_price),
                        price=current_price,
                        confidence=0.6,
                        reason="Price at Lower Bollinger Band"
                    ))
                elif current_price >= latest['BB_Upper']:
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.SELL,
                        quantity=self._get_position_quantity(symbol),
                        price=current_price,
                        confidence=0.6,
                        reason="Price at Upper Bollinger Band"
                    ))
            
        except Exception as e:
            logger.error(f"Error in technical analysis for {symbol}: {e}")
        
        return signals
    
    def _sentiment_analysis(self, symbol: str, current_data: Dict) -> List[TradeSignal]:
        """Generate signals based on news sentiment"""
        signals = []
        
        try:
            # Get news sentiment
            news_items = self.data_fetcher.get_news_sentiment(symbol, days=3)
            if not news_items:
                return signals
            
            # Calculate average sentiment
            sentiment_scores = [item['sentiment_score'] for item in news_items]
            avg_sentiment = np.mean(sentiment_scores)
            
            # Generate signals based on sentiment
            if avg_sentiment > 0.7:  # Very positive sentiment
                signals.append(TradeSignal(
                    symbol=symbol,
                    action=TradeType.BUY,
                    quantity=self._calculate_position_size(symbol, current_data['price']),
                    price=current_data['price'],
                    confidence=0.5,
                    reason=f"Positive News Sentiment ({avg_sentiment:.2f})"
                ))
            elif avg_sentiment < 0.3:  # Very negative sentiment
                signals.append(TradeSignal(
                    symbol=symbol,
                    action=TradeType.SELL,
                    quantity=self._get_position_quantity(symbol),
                    price=current_data['price'],
                    confidence=0.5,
                    reason=f"Negative News Sentiment ({avg_sentiment:.2f})"
                ))
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {symbol}: {e}")
        
        return signals
    
    def _risk_management_analysis(self, symbol: str, current_data: Dict) -> List[TradeSignal]:
        """Generate signals based on risk management rules"""
        signals = []
        
        try:
            current_price = current_data['price']
            
            # Check if we have a position
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Stop loss check
                stop_loss_price = position.avg_price * (1 - settings.stop_loss_percent / 100)
                if current_price <= stop_loss_price:
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.SELL,
                        quantity=position.quantity,
                        price=current_price,
                        confidence=1.0,
                        reason=f"Stop Loss Triggered ({settings.stop_loss_percent}%)"
                    ))
                
                # Take profit check
                take_profit_price = position.avg_price * (1 + settings.sell_threshold_percent / 100)
                if current_price >= take_profit_price:
                    signals.append(TradeSignal(
                        symbol=symbol,
                        action=TradeType.SELL,
                        quantity=position.quantity,
                        price=current_price,
                        confidence=0.8,
                        reason=f"Take Profit Target ({settings.sell_threshold_percent}%)"
                    ))
            
            # Daily loss limit check
            if self.daily_pnl <= -self.max_daily_loss:
                # Close all positions
                for pos_symbol, position in self.positions.items():
                    signals.append(TradeSignal(
                        symbol=pos_symbol,
                        action=TradeType.SELL,
                        quantity=position.quantity,
                        price=current_price,
                        confidence=1.0,
                        reason="Daily Loss Limit Reached"
                    ))
            
        except Exception as e:
            logger.error(f"Error in risk management analysis for {symbol}: {e}")
        
        return signals
    
    def _combine_signals(self, signals: List[TradeSignal]) -> List[TradeSignal]:
        """Combine and filter trading signals"""
        if not signals:
            return []
        
        # Group signals by symbol and action
        signal_groups = {}
        for signal in signals:
            key = (signal.symbol, signal.action)
            if key not in signal_groups:
                signal_groups[key] = []
            signal_groups[key].append(signal)
        
        # Combine signals for each group
        final_signals = []
        for (symbol, action), group_signals in signal_groups.items():
            if len(group_signals) == 1:
                final_signals.append(group_signals[0])
            else:
                # Combine multiple signals
                combined_signal = self._merge_signals(group_signals)
                if combined_signal.confidence >= 0.6:  # Minimum confidence threshold
                    final_signals.append(combined_signal)
        
        return final_signals
    
    def _merge_signals(self, signals: List[TradeSignal]) -> TradeSignal:
        """Merge multiple signals into one"""
        if not signals:
            return None
        
        # Use the first signal as base
        base_signal = signals[0]
        
        # Calculate weighted average confidence
        total_confidence = sum(s.confidence for s in signals)
        avg_confidence = total_confidence / len(signals)
        
        # Combine reasons
        reasons = [s.reason for s in signals]
        combined_reason = " | ".join(reasons)
        
        # Use average quantity
        avg_quantity = int(np.mean([s.quantity for s in signals]))
        
        return TradeSignal(
            symbol=base_signal.symbol,
            action=base_signal.action,
            quantity=avg_quantity,
            price=base_signal.price,
            confidence=avg_confidence,
            reason=combined_reason
        )
    
    def _calculate_position_size(self, symbol: str, price: float) -> int:
        """Calculate position size based on risk management rules"""
        try:
            # Maximum position value
            max_position_value = settings.max_position_size
            
            # Calculate quantity based on max position size
            quantity = int(max_position_value / price)
            
            # Ensure minimum quantity
            quantity = max(1, quantity)
            
            return quantity
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1
    
    def _get_position_quantity(self, symbol: str) -> int:
        """Get current position quantity for a symbol"""
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return 0
    
    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute a trading signal"""
        try:
            logger.info(f"Executing trade: {signal.action.value} {signal.quantity} {signal.symbol} at {signal.price}")
            
            # Here you would integrate with your broker API
            # For now, we'll simulate the trade execution
            
            # Update positions
            if signal.action == TradeType.BUY:
                self._update_position_buy(signal)
            else:
                self._update_position_sell(signal)
            
            # Log the trade
            self._log_trade(signal)
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _update_position_buy(self, signal: TradeSignal):
        """Update position after a buy trade"""
        if signal.symbol in self.positions:
            # Update existing position
            position = self.positions[signal.symbol]
            total_quantity = position.quantity + signal.quantity
            total_value = (position.quantity * position.avg_price) + (signal.quantity * signal.price)
            new_avg_price = total_value / total_quantity
            
            position.quantity = total_quantity
            position.avg_price = new_avg_price
        else:
            # Create new position
            self.positions[signal.symbol] = Position(
                symbol=signal.symbol,
                quantity=signal.quantity,
                avg_price=signal.price,
                current_price=signal.price,
                unrealized_pnl=0.0,
                entry_time=datetime.now()
            )
    
    def _update_position_sell(self, signal: TradeSignal):
        """Update position after a sell trade"""
        if signal.symbol in self.positions:
            position = self.positions[signal.symbol]
            
            # Calculate realized P&L
            realized_pnl = (signal.price - position.avg_price) * signal.quantity
            self.daily_pnl += realized_pnl
            
            # Update position quantity
            position.quantity -= signal.quantity
            
            # Remove position if quantity becomes zero
            if position.quantity <= 0:
                del self.positions[signal.symbol]
    
    def _log_trade(self, signal: TradeSignal):
        """Log trade to database"""
        try:
            # This would save to database in a real implementation
            logger.info(f"Trade logged: {signal.symbol} {signal.action.value} {signal.quantity} @ {signal.price}")
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary"""
        try:
            total_value = 0.0
            total_pnl = 0.0
            
            for symbol, position in self.positions.items():
                current_data = self.data_fetcher.get_live_price(symbol)
                if current_data:
                    position.current_price = current_data['price']
                    position.unrealized_pnl = (position.current_price - position.avg_price) * position.quantity
                    total_value += position.current_price * position.quantity
                    total_pnl += position.unrealized_pnl
            
            return {
                'total_positions': len(self.positions),
                'total_value': total_value,
                'unrealized_pnl': total_pnl,
                'realized_pnl': self.daily_pnl,
                'total_pnl': total_pnl + self.daily_pnl,
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'quantity': pos.quantity,
                        'avg_price': pos.avg_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'entry_time': pos.entry_time
                    }
                    for pos in self.positions.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def check_price_alerts(self, symbol: str, current_price: float) -> List[Alert]:
        """Check if any price alerts should be triggered"""
        alerts = []
        
        try:
            # This would check against database alerts in a real implementation
            # For now, we'll implement basic threshold alerts
            
            if symbol in self.positions:
                position = self.positions[symbol]
                
                # Price change alerts
                price_change_percent = ((current_price - position.avg_price) / position.avg_price) * 100
                
                if price_change_percent >= settings.sell_threshold_percent:
                    alerts.append(Alert(
                        stock_symbol=symbol,
                        alert_type="PRICE_ABOVE",
                        threshold_value=position.avg_price * (1 + settings.sell_threshold_percent / 100),
                        current_value=current_price,
                        is_triggered=True,
                        triggered_at=datetime.now()
                    ))
                elif price_change_percent <= -settings.buy_threshold_percent:
                    alerts.append(Alert(
                        stock_symbol=symbol,
                        alert_type="PRICE_BELOW",
                        threshold_value=position.avg_price * (1 - settings.buy_threshold_percent / 100),
                        current_value=current_price,
                        is_triggered=True,
                        triggered_at=datetime.now()
                    ))
            
        except Exception as e:
            logger.error(f"Error checking price alerts for {symbol}: {e}")
        
        return alerts
