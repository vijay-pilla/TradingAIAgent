"""
Trade Genius Agent - Demo Script
Test the core functionality of the AI trading agent
"""

import asyncio
import sys
from datetime import datetime
from loguru import logger

# Configure logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

from src.data_fetcher import DataFetcher
from src.trading_logic import TradingEngine
from src.predictor import AIPredictor
from src.charting import ChartGenerator
from src.emailer import EmailReporter

async def demo_data_fetcher():
    """Demo the data fetcher functionality"""
    print("\n" + "="*60)
    print("🔍 TESTING DATA FETCHER")
    print("="*60)
    
    data_fetcher = DataFetcher()
    
    # Test live price fetching
    print("\n📈 Testing Live Price Fetching...")
    symbols = ["TCS.NS", "INFY.NS", "RELIANCE.NS"]
    
    for symbol in symbols:
        try:
            price_data = data_fetcher.get_live_price(symbol)
            if price_data:
                print(f"✅ {symbol}: ₹{price_data['price']:.2f} ({price_data['change_percent']:+.2f}%)")
            else:
                print(f"❌ {symbol}: No data available")
        except Exception as e:
            print(f"❌ {symbol}: Error - {e}")
    
    # Test historical data
    print("\n📊 Testing Historical Data...")
    try:
        historical_data = data_fetcher.get_historical_data("TCS.NS", "1mo")
        if historical_data is not None and not historical_data.empty:
            print(f"✅ Historical data: {len(historical_data)} records")
            print(f"   Latest close: ₹{historical_data['Close'].iloc[-1]:.2f}")
            print(f"   Date range: {historical_data.index[0].date()} to {historical_data.index[-1].date()}")
        else:
            print("❌ No historical data available")
    except Exception as e:
        print(f"❌ Historical data error: {e}")
    
    # Test news sentiment
    print("\n📰 Testing News Sentiment...")
    try:
        news_items = data_fetcher.get_news_sentiment("TCS.NS", days=3)
        if news_items:
            avg_sentiment = sum(item['sentiment_score'] for item in news_items) / len(news_items)
            print(f"✅ News sentiment: {len(news_items)} articles, avg score: {avg_sentiment:.2f}")
        else:
            print("❌ No news sentiment data available")
    except Exception as e:
        print(f"❌ News sentiment error: {e}")
    
    # Test market overview
    print("\n🌍 Testing Market Overview...")
    try:
        market_data = data_fetcher.get_market_overview()
        if market_data and 'indices' in market_data:
            print(f"✅ Market overview: {len(market_data['indices'])} indices")
            for index_name, index_data in market_data['indices'].items():
                if index_data:
                    print(f"   {index_name}: ₹{index_data['price']:.2f}")
        else:
            print("❌ No market overview data available")
    except Exception as e:
        print(f"❌ Market overview error: {e}")

async def demo_predictor():
    """Demo the AI predictor functionality"""
    print("\n" + "="*60)
    print("🧠 TESTING AI PREDICTOR")
    print("="*60)
    
    data_fetcher = DataFetcher()
    predictor = AIPredictor(data_fetcher)
    
    # Test price prediction
    print("\n🔮 Testing Price Predictions...")
    symbols = ["TCS.NS", "INFY.NS"]
    
    for symbol in symbols:
        try:
            prediction = predictor.predict_price(symbol, days_ahead=1)
            if prediction and prediction['predicted_price'] > 0:
                print(f"✅ {symbol}:")
                print(f"   Predicted Price: ₹{prediction['predicted_price']:.2f}")
                print(f"   Confidence: {prediction['confidence_score']:.2f}")
                print(f"   Sentiment: {prediction['sentiment_label']}")
                print(f"   Model: {prediction['model_used']}")
            else:
                print(f"❌ {symbol}: No prediction available")
        except Exception as e:
            print(f"❌ {symbol}: Prediction error - {e}")
    
    # Test prediction accuracy
    print("\n📊 Testing Prediction Accuracy...")
    try:
        accuracy = predictor.get_prediction_accuracy("TCS.NS", days_back=7)
        if accuracy['samples'] > 0:
            print(f"✅ Prediction Accuracy:")
            print(f"   Directional Accuracy: {accuracy['accuracy']:.2%}")
            print(f"   MAE: ₹{accuracy['mae']:.2f}")
            print(f"   RMSE: ₹{accuracy['rmse']:.2f}")
            print(f"   Samples: {accuracy['samples']}")
        else:
            print("❌ Insufficient data for accuracy calculation")
    except Exception as e:
        print(f"❌ Accuracy calculation error: {e}")

async def demo_trading_engine():
    """Demo the trading engine functionality"""
    print("\n" + "="*60)
    print("⚡ TESTING TRADING ENGINE")
    print("="*60)
    
    data_fetcher = DataFetcher()
    trading_engine = TradingEngine(data_fetcher)
    
    # Test signal generation
    print("\n🎯 Testing Signal Generation...")
    symbols = ["TCS.NS", "INFY.NS"]
    
    for symbol in symbols:
        try:
            signals = trading_engine.analyze_and_generate_signals(symbol)
            if signals:
                print(f"✅ {symbol}: {len(signals)} signals generated")
                for signal in signals:
                    print(f"   {signal.action.value}: {signal.quantity} @ ₹{signal.price:.2f} (Confidence: {signal.confidence:.2f})")
                    print(f"   Reason: {signal.reason}")
            else:
                print(f"ℹ️ {symbol}: No signals generated")
        except Exception as e:
            print(f"❌ {symbol}: Signal generation error - {e}")
    
    # Test portfolio summary
    print("\n💼 Testing Portfolio Summary...")
    try:
        portfolio_data = trading_engine.get_portfolio_summary()
        print(f"✅ Portfolio Summary:")
        print(f"   Total Positions: {portfolio_data.get('total_positions', 0)}")
        print(f"   Total Value: ₹{portfolio_data.get('total_value', 0):,.2f}")
        print(f"   Total P&L: ₹{portfolio_data.get('total_pnl', 0):,.2f}")
        print(f"   Unrealized P&L: ₹{portfolio_data.get('unrealized_pnl', 0):,.2f}")
    except Exception as e:
        print(f"❌ Portfolio summary error: {e}")

async def demo_chart_generator():
    """Demo the chart generator functionality"""
    print("\n" + "="*60)
    print("📊 TESTING CHART GENERATOR")
    print("="*60)
    
    data_fetcher = DataFetcher()
    chart_generator = ChartGenerator(data_fetcher)
    
    # Test price chart generation
    print("\n📈 Testing Price Chart Generation...")
    try:
        chart_base64 = chart_generator.create_price_chart("TCS.NS", days=30)
        if chart_base64 and chart_base64.startswith('data:image'):
            print("✅ Price chart generated successfully")
            print(f"   Chart size: {len(chart_base64)} characters")
        else:
            print("❌ Price chart generation failed")
    except Exception as e:
        print(f"❌ Price chart error: {e}")
    
    # Test portfolio chart generation
    print("\n💼 Testing Portfolio Chart Generation...")
    try:
        portfolio_data = {
            'total_value': 100000,
            'total_pnl': 5000,
            'total_positions': 3,
            'positions': [
                {'symbol': 'TCS.NS', 'quantity': 10, 'current_price': 3500, 'unrealized_pnl': 2000},
                {'symbol': 'INFY.NS', 'quantity': 20, 'current_price': 1500, 'unrealized_pnl': 1500},
                {'symbol': 'RELIANCE.NS', 'quantity': 5, 'current_price': 2500, 'unrealized_pnl': 1500}
            ]
        }
        chart_base64 = chart_generator.create_portfolio_chart(portfolio_data)
        if chart_base64 and chart_base64.startswith('data:image'):
            print("✅ Portfolio chart generated successfully")
        else:
            print("❌ Portfolio chart generation failed")
    except Exception as e:
        print(f"❌ Portfolio chart error: {e}")
    
    # Test sentiment chart generation
    print("\n📰 Testing Sentiment Chart Generation...")
    try:
        chart_base64 = chart_generator.create_sentiment_chart("TCS.NS", days=7)
        if chart_base64 and chart_base64.startswith('data:image'):
            print("✅ Sentiment chart generated successfully")
        else:
            print("❌ Sentiment chart generation failed")
    except Exception as e:
        print(f"❌ Sentiment chart error: {e}")

async def demo_email_reporter():
    """Demo the email reporter functionality"""
    print("\n" + "="*60)
    print("📧 TESTING EMAIL REPORTER")
    print("="*60)
    
    email_reporter = EmailReporter()
    
    # Test email connection
    print("\n🔗 Testing Email Connection...")
    try:
        connection_ok = email_reporter.test_email_connection()
        if connection_ok:
            print("✅ Email connection successful")
        else:
            print("❌ Email connection failed - check your SMTP settings")
    except Exception as e:
        print(f"❌ Email connection error: {e}")
    
    # Test report generation (without sending)
    print("\n📄 Testing Report Generation...")
    try:
        portfolio_data = {
            'total_value': 100000,
            'total_pnl': 5000,
            'total_positions': 3
        }
        
        market_data = {
            'indices': {
                'NIFTY_50': {'price': 18000, 'change_percent': 1.5},
                'SENSEX': {'price': 60000, 'change_percent': 1.2}
            }
        }
        
        predictions = [
            {
                'symbol': 'TCS.NS',
                'predicted_price': 3600,
                'confidence_score': 0.75,
                'sentiment_label': 'POSITIVE'
            }
        ]
        
        html_content = email_reporter._create_daily_report_html(
            portfolio_data, market_data, predictions
        )
        
        if html_content and len(html_content) > 1000:
            print("✅ Daily report HTML generated successfully")
            print(f"   Report size: {len(html_content)} characters")
        else:
            print("❌ Daily report generation failed")
    except Exception as e:
        print(f"❌ Report generation error: {e}")

async def main():
    """Run all demo tests"""
    print("🤖 TRADE GENIUS AGENT - DEMO")
    print("="*60)
    print(f"Demo started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Run all demos
        await demo_data_fetcher()
        await demo_predictor()
        await demo_trading_engine()
        await demo_chart_generator()
        await demo_email_reporter()
        
        print("\n" + "="*60)
        print("✅ DEMO COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n🎉 All core components are working!")
        print("\n📋 Next Steps:")
        print("1. Configure your .env file with API keys")
        print("2. Run 'python main.py' to start the web interface")
        print("3. Open http://localhost:8000 in your browser")
        print("4. Start with paper trading before live trading")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        logger.exception("Demo error details:")

if __name__ == "__main__":
    asyncio.run(main())
