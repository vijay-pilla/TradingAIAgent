"""
Trade Genius Agent - AI Prediction Engine
Handles trend prediction using ARIMA, LSTM, and sentiment analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from loguru import logger
from src.data_fetcher import DataFetcher
from src.models import Prediction

class AIPredictor:
    """AI-powered prediction engine for stock price forecasting"""
    
    def __init__(self, data_fetcher: DataFetcher):
        self.data_fetcher = data_fetcher
        self.scaler = MinMaxScaler()
        self.models = {}  # Cache trained models
        
    def predict_price(self, symbol: str, days_ahead: int = 1) -> Dict:
        """Predict stock price for the next N days"""
        try:
            # Get historical data
            historical_data = self.data_fetcher.get_historical_data(symbol, "2y")
            if historical_data is None or historical_data.empty:
                return self._create_empty_prediction(symbol)
            
            # Get news sentiment
            news_sentiment = self._get_aggregated_sentiment(symbol)
            
            # Multiple prediction methods
            arima_prediction = self._arima_prediction(historical_data, days_ahead)
            lstm_prediction = self._lstm_prediction(historical_data, days_ahead)
            technical_prediction = self._technical_prediction(historical_data, days_ahead)
            
            # Combine predictions
            final_prediction = self._ensemble_prediction(
                arima_prediction, lstm_prediction, technical_prediction, news_sentiment
            )
            
            return {
                'symbol': symbol,
                'prediction_date': datetime.now() + timedelta(days=days_ahead),
                'predicted_price': final_prediction['price'],
                'confidence_score': final_prediction['confidence'],
                'model_used': 'Ensemble (ARIMA + LSTM + Technical)',
                'sentiment_score': news_sentiment['score'],
                'sentiment_label': news_sentiment['label'],
                'technical_indicators': self._get_technical_indicators(historical_data),
                'arima_prediction': arima_prediction,
                'lstm_prediction': lstm_prediction,
                'technical_prediction': technical_prediction,
                'created_at': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error predicting price for {symbol}: {e}")
            return self._create_empty_prediction(symbol)
    
    def _arima_prediction(self, data: pd.DataFrame, days_ahead: int) -> Dict:
        """ARIMA model prediction"""
        try:
            # Use closing prices
            prices = data['Close'].dropna()
            
            # Auto ARIMA parameter selection
            best_aic = float('inf')
            best_params = (1, 1, 1)
            
            # Grid search for best parameters
            for p in range(0, 3):
                for d in range(0, 2):
                    for q in range(0, 3):
                        try:
                            model = ARIMA(prices, order=(p, d, q))
                            fitted_model = model.fit()
                            if fitted_model.aic < best_aic:
                                best_aic = fitted_model.aic
                                best_params = (p, d, q)
                        except:
                            continue
            
            # Train final model
            model = ARIMA(prices, order=best_params)
            fitted_model = model.fit()
            
            # Make prediction
            forecast = fitted_model.forecast(steps=days_ahead)
            prediction = forecast.iloc[-1] if days_ahead == 1 else forecast.iloc[-1]
            
            # Calculate confidence based on model fit
            confidence = max(0.3, min(0.9, 1 - (fitted_model.aic / 10000)))
            
            return {
                'price': float(prediction),
                'confidence': confidence,
                'model_params': best_params,
                'aic': fitted_model.aic
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {e}")
            return {'price': 0, 'confidence': 0, 'error': str(e)}
    
    def _lstm_prediction(self, data: pd.DataFrame, days_ahead: int) -> Dict:
        """LSTM neural network prediction"""
        try:
            # Prepare data
            prices = data['Close'].values.reshape(-1, 1)
            scaled_prices = self.scaler.fit_transform(prices)
            
            # Create sequences
            sequence_length = 60  # Use 60 days of data to predict next day
            X, y = self._create_sequences(scaled_prices, sequence_length)
            
            if len(X) < 100:  # Need sufficient data
                return {'price': 0, 'confidence': 0, 'error': 'Insufficient data'}
            
            # Split data
            split_index = int(0.8 * len(X))
            X_train, X_test = X[:split_index], X[split_index:]
            y_train, y_test = y[:split_index], y[split_index:]
            
            # Build LSTM model
            model = Sequential([
                LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train model
            model.fit(X_train, y_train, 
                     batch_size=32, 
                     epochs=50, 
                     validation_data=(X_test, y_test),
                     verbose=0)
            
            # Make prediction
            last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
            prediction_scaled = model.predict(last_sequence, verbose=0)
            prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
            
            # Calculate confidence based on validation loss
            val_loss = model.evaluate(X_test, y_test, verbose=0)
            confidence = max(0.3, min(0.9, 1 - val_loss))
            
            return {
                'price': float(prediction),
                'confidence': confidence,
                'val_loss': val_loss
            }
            
        except Exception as e:
            logger.error(f"Error in LSTM prediction: {e}")
            return {'price': 0, 'confidence': 0, 'error': str(e)}
    
    def _technical_prediction(self, data: pd.DataFrame, days_ahead: int) -> Dict:
        """Technical analysis-based prediction"""
        try:
            latest = data.iloc[-1]
            current_price = latest['Close']
            
            # Calculate trend indicators
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            ema_12 = data['Close'].ewm(span=12).mean().iloc[-1]
            ema_26 = data['Close'].ewm(span=26).mean().iloc[-1]
            
            # Trend analysis
            trend_score = 0
            if current_price > sma_20:
                trend_score += 0.2
            if sma_20 > sma_50:
                trend_score += 0.2
            if ema_12 > ema_26:
                trend_score += 0.2
            
            # Momentum indicators
            rsi = self._calculate_rsi(data['Close'])
            macd = ema_12 - ema_26
            
            # RSI analysis
            if rsi < 30:  # Oversold
                trend_score += 0.2
            elif rsi > 70:  # Overbought
                trend_score -= 0.2
            
            # MACD analysis
            if macd > 0:
                trend_score += 0.1
            else:
                trend_score -= 0.1
            
            # Volume analysis
            avg_volume = data['Volume'].rolling(window=20).mean().iloc[-1]
            current_volume = latest['Volume']
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            
            if volume_ratio > 1.5:  # High volume
                trend_score += 0.1
            
            # Calculate price change based on trend
            price_change_percent = (trend_score - 0.5) * 0.05  # Max 5% change
            predicted_price = current_price * (1 + price_change_percent)
            
            # Confidence based on trend strength
            confidence = abs(trend_score - 0.5) * 2  # 0 to 1
            
            return {
                'price': float(predicted_price),
                'confidence': confidence,
                'trend_score': trend_score,
                'rsi': rsi,
                'macd': macd,
                'volume_ratio': volume_ratio
            }
            
        except Exception as e:
            logger.error(f"Error in technical prediction: {e}")
            return {'price': 0, 'confidence': 0, 'error': str(e)}
    
    def _ensemble_prediction(self, arima: Dict, lstm: Dict, technical: Dict, sentiment: Dict) -> Dict:
        """Combine multiple predictions using ensemble method"""
        try:
            predictions = []
            weights = []
            
            # ARIMA prediction
            if arima['price'] > 0:
                predictions.append(arima['price'])
                weights.append(arima['confidence'] * 0.3)
            
            # LSTM prediction
            if lstm['price'] > 0:
                predictions.append(lstm['price'])
                weights.append(lstm['confidence'] * 0.4)
            
            # Technical prediction
            if technical['price'] > 0:
                predictions.append(technical['price'])
                weights.append(technical['confidence'] * 0.2)
            
            # Sentiment adjustment
            sentiment_weight = 0.1
            sentiment_adjustment = (sentiment['score'] - 0.5) * 0.02  # Max 2% adjustment
            
            if not predictions:
                return {'price': 0, 'confidence': 0}
            
            # Weighted average
            total_weight = sum(weights)
            if total_weight > 0:
                weighted_price = sum(p * w for p, w in zip(predictions, weights)) / total_weight
                avg_confidence = sum(weights) / len(weights)
            else:
                weighted_price = sum(predictions) / len(predictions)
                avg_confidence = 0.5
            
            # Apply sentiment adjustment
            final_price = weighted_price * (1 + sentiment_adjustment)
            
            # Adjust confidence based on sentiment
            sentiment_confidence = abs(sentiment['score'] - 0.5) * 2
            final_confidence = (avg_confidence + sentiment_confidence) / 2
            
            return {
                'price': final_price,
                'confidence': min(0.95, max(0.1, final_confidence))
            }
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'price': 0, 'confidence': 0}
    
    def _create_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i])
        return np.array(X), np.array(y)
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except:
            return 50
    
    def _get_aggregated_sentiment(self, symbol: str) -> Dict:
        """Get aggregated sentiment score"""
        try:
            news_items = self.data_fetcher.get_news_sentiment(symbol, days=3)
            if not news_items:
                return {'score': 0.5, 'label': 'NEUTRAL'}
            
            scores = [item['sentiment_score'] for item in news_items]
            avg_score = np.mean(scores)
            
            if avg_score > 0.6:
                label = 'POSITIVE'
            elif avg_score < 0.4:
                label = 'NEGATIVE'
            else:
                label = 'NEUTRAL'
            
            return {'score': avg_score, 'label': label}
            
        except Exception as e:
            logger.error(f"Error getting sentiment for {symbol}: {e}")
            return {'score': 0.5, 'label': 'NEUTRAL'}
    
    def _get_technical_indicators(self, data: pd.DataFrame) -> Dict:
        """Get current technical indicators"""
        try:
            latest = data.iloc[-1]
            
            return {
                'sma_20': float(data['Close'].rolling(window=20).mean().iloc[-1]),
                'sma_50': float(data['Close'].rolling(window=50).mean().iloc[-1]),
                'rsi': float(self._calculate_rsi(data['Close'])),
                'macd': float(data['Close'].ewm(span=12).mean().iloc[-1] - data['Close'].ewm(span=26).mean().iloc[-1]),
                'volume_ratio': float(latest['Volume'] / data['Volume'].rolling(window=20).mean().iloc[-1])
            }
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            return {}
    
    def _create_empty_prediction(self, symbol: str) -> Dict:
        """Create empty prediction when data is insufficient"""
        return {
            'symbol': symbol,
            'prediction_date': datetime.now() + timedelta(days=1),
            'predicted_price': 0,
            'confidence_score': 0,
            'model_used': 'No Data',
            'sentiment_score': 0.5,
            'sentiment_label': 'NEUTRAL',
            'technical_indicators': {},
            'created_at': datetime.now()
        }
    
    def batch_predict(self, symbols: List[str], days_ahead: int = 1) -> List[Dict]:
        """Predict prices for multiple symbols"""
        predictions = []
        
        for symbol in symbols:
            try:
                prediction = self.predict_price(symbol, days_ahead)
                predictions.append(prediction)
            except Exception as e:
                logger.error(f"Error predicting {symbol}: {e}")
                predictions.append(self._create_empty_prediction(symbol))
        
        return predictions
    
    def get_prediction_accuracy(self, symbol: str, days_back: int = 30) -> Dict:
        """Calculate prediction accuracy for a symbol"""
        try:
            # Get historical data
            data = self.data_fetcher.get_historical_data(symbol, "3mo")
            if data is None or len(data) < days_back + 10:
                return {'accuracy': 0, 'mae': 0, 'rmse': 0}
            
            # Get predictions for past days
            actual_prices = []
            predicted_prices = []
            
            for i in range(days_back, 0, -1):
                # Use data up to i days ago for prediction
                train_data = data.iloc[:-i]
                actual_price = data['Close'].iloc[-i]
                
                # Make prediction
                prediction = self.predict_price(symbol, 1)
                if prediction['predicted_price'] > 0:
                    actual_prices.append(actual_price)
                    predicted_prices.append(prediction['predicted_price'])
            
            if len(actual_prices) < 5:
                return {'accuracy': 0, 'mae': 0, 'rmse': 0}
            
            # Calculate metrics
            mae = mean_absolute_error(actual_prices, predicted_prices)
            rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
            
            # Calculate directional accuracy
            actual_direction = np.diff(actual_prices)
            predicted_direction = np.diff(predicted_prices)
            directional_accuracy = np.mean(np.sign(actual_direction) == np.sign(predicted_direction))
            
            return {
                'accuracy': directional_accuracy,
                'mae': mae,
                'rmse': rmse,
                'samples': len(actual_prices)
            }
            
        except Exception as e:
            logger.error(f"Error calculating accuracy for {symbol}: {e}")
            return {'accuracy': 0, 'mae': 0, 'rmse': 0}
