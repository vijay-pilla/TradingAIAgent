"""
Trade Genius Agent - Data Fetcher
Handles real-time price monitoring and news sentiment analysis
"""

import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time
from loguru import logger
from bs4 import BeautifulSoup
from textblob import TextBlob
import json

from src.config import settings, STOCK_SYMBOLS
from src.models import StockPrice, NewsSentiment

class StockDataNotFound(Exception):
    """Exception raised when stock data is not found"""
    pass

class DataFetcher:
    """Handles all data fetching operations"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def get_live_price(self, symbol: str) -> Dict:
        """Fetch live price for a stock symbol"""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d", interval="1m")

            if hist.empty or hist['Close'].isnull().all():
                logger.warning(f"No data available for {symbol} (possibly delisted or inactive)")
                raise StockDataNotFound(f"No data available for {symbol}")

            latest = hist.dropna(subset=['Close']).iloc[-1]

            return {
                'symbol': symbol,
                'price': float(latest['Close']),
                'open': float(latest['Open']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'volume': int(latest['Volume']),
                'timestamp': datetime.now(),
                'change': float(latest['Close'] - latest['Open']),
                'change_percent': float((latest['Close'] - latest['Open']) / latest['Open'] * 100)
            }
        except StockDataNotFound:
            raise
        except Exception as e:
            logger.error(f"Error fetching live price for {symbol}: {e}")
            raise StockDataNotFound(f"Error fetching data for {symbol}: {e}")
    
    def get_historical_data(self, symbol: str, period: str = "1y") -> Optional[pd.DataFrame]:
        """Fetch historical data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            if data.empty:
                logger.warning(f"No historical data for {symbol}")
                return None
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return None
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to price data"""
        try:
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            df['SMA_200'] = df['Close'].rolling(window=200).mean()
            
            # Exponential Moving Averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['BB_Middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            
            # Volume indicators
            df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
            
            return df
            
        except Exception as e:
            logger.error(f"Error adding technical indicators: {e}")
            return df
    
    def get_news_sentiment(self, symbol: str, days: int = 7) -> List[Dict]:
        """Fetch and analyze news sentiment for a stock"""
        try:
            # Get company name from symbol
            company_name = self._get_company_name(symbol)
            if not company_name:
                return []
            
            # Fetch news from multiple sources
            news_items = []
            
            # Yahoo Finance news
            yahoo_news = self._fetch_yahoo_news(symbol)
            news_items.extend(yahoo_news)
            
            # Google News (using web scraping)
            google_news = self._fetch_google_news(company_name, days)
            news_items.extend(google_news)
            
            # Analyze sentiment for each news item
            analyzed_news = []
            for item in news_items:
                sentiment = self._analyze_sentiment(item['headline'])
                item.update(sentiment)
                analyzed_news.append(item)
            
            return analyzed_news
            
        except Exception as e:
            logger.error(f"Error fetching news sentiment for {symbol}: {e}")
            return []
    
    def _get_company_name(self, symbol: str) -> Optional[str]:
        """Get company name from symbol"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info.get('longName', info.get('shortName'))
        except:
            return None
    
    def _fetch_yahoo_news(self, symbol: str) -> List[Dict]:
        """Fetch news from Yahoo Finance"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            news_items = []
            for item in news[:10]:  # Limit to 10 recent news
                news_items.append({
                    'headline': item.get('title', ''),
                    'summary': item.get('summary', ''),
                    'source': item.get('publisher', 'Yahoo Finance'),
                    'published_at': datetime.fromtimestamp(item.get('providerPublishTime', 0)),
                    'url': item.get('link', '')
                })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching Yahoo news: {e}")
            return []
    
    def _fetch_google_news(self, company_name: str, days: int) -> List[Dict]:
        """Fetch news from Google News using web scraping"""
        try:
            # This is a simplified version - in production, use proper news APIs
            query = f"{company_name} stock news"
            url = f"https://news.google.com/search?q={query}&hl=en&gl=US&ceid=US:en"
            
            response = self.session.get(url, timeout=10)
            if response.status_code != 200:
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            articles = soup.find_all('article')[:5]  # Limit to 5 articles
            
            news_items = []
            for article in articles:
                headline_elem = article.find('h3')
                if headline_elem:
                    headline = headline_elem.get_text().strip()
                    news_items.append({
                        'headline': headline,
                        'summary': '',
                        'source': 'Google News',
                        'published_at': datetime.now() - timedelta(hours=1),  # Approximate
                        'url': ''
                    })
            
            return news_items
            
        except Exception as e:
            logger.error(f"Error fetching Google news: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of news text"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 to 1
            subjectivity = blob.sentiment.subjectivity  # 0 to 1
            
            # Convert to 0-1 scale for easier interpretation
            sentiment_score = (polarity + 1) / 2  # 0 to 1
            
            # Determine sentiment label
            if sentiment_score > 0.6:
                sentiment_label = "POSITIVE"
            elif sentiment_score < 0.4:
                sentiment_label = "NEGATIVE"
            else:
                sentiment_label = "NEUTRAL"
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'polarity': polarity,
                'subjectivity': subjectivity
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'sentiment_score': 0.5,
                'sentiment_label': "NEUTRAL",
                'polarity': 0.0,
                'subjectivity': 0.5
            }
    
    def get_market_overview(self) -> Dict:
        """Get overall market overview"""
        try:
            # Fetch major indices
            indices = {
                'NIFTY_50': '^NSEI',
                'SENSEX': '^BSESN',
                'NIFTY_BANK': '^NSEBANK'
            }
            
            market_data = {}
            for name, symbol in indices.items():
                data = self.get_live_price(symbol)
                if data:
                    market_data[name] = data
            
            return {
                'timestamp': datetime.now(),
                'indices': market_data,
                'market_status': self._get_market_status()
            }
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {}
    
    def _get_market_status(self) -> str:
        """Check if market is open"""
        now = datetime.now()
        # Indian market hours: 9:15 AM to 3:30 PM IST
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if market_open <= now <= market_close and now.weekday() < 5:
            return "OPEN"
        else:
            return "CLOSED"
    
    def monitor_stocks(self, symbols: List[str] = None) -> Dict[str, Dict]:
        """Monitor multiple stocks simultaneously"""
        if symbols is None:
            symbols = settings.get_default_stocks_list()
        
        results = {}
        
        for symbol in symbols:
            try:
                price_data = self.get_live_price(symbol)
                if price_data:
                    results[symbol] = price_data
                time.sleep(0.1)  # Rate limiting
            except Exception as e:
                logger.error(f"Error monitoring {symbol}: {e}")
                results[symbol] = None
        
        return results
