# 🤖 Trade Genius Agent

> **An AI-Powered Trading Agent with Real-time Analysis, Automated Trading, and Beautiful Web Interface**

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🚀 Features

### 📈 Real-Time Trading
- **Live Price Monitoring** - Track multiple stocks with yfinance
- **Automated Trade Execution** - Execute trades based on AI signals
- **Risk Management** - Built-in stop-loss and position sizing
- **Portfolio Tracking** - Real-time P&L and position monitoring

### 🧠 AI-Powered Analysis
- **Multi-Model Predictions** - ARIMA, LSTM, and Technical Analysis
- **News Sentiment Analysis** - Real-time sentiment from financial news
- **Ensemble Forecasting** - Combines multiple models for better accuracy
- **Confidence Scoring** - AI confidence levels for each prediction

### 📊 Beautiful Web Interface
- **Real-time Dashboard** - Live updates via WebSocket
- **Interactive Charts** - Plotly-powered visualizations
- **Portfolio Analytics** - Comprehensive performance metrics
- **Mobile Responsive** - Works on all devices

### 📧 Automated Reporting
- **Daily Email Reports** - Automated summaries with charts
- **Trade Notifications** - Instant alerts for executed trades
- **Price Alerts** - Customizable threshold notifications
- **Error Monitoring** - System health notifications

### 🔧 Advanced Features
- **Scheduled Trading** - Automated tasks with APScheduler
- **Broker Integration** - Ready for Zerodha Kite Connect
- **Database Storage** - SQLite/PostgreSQL support
- **RESTful API** - Complete API for external integrations

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Quick Start

1. **Clone the repository**
```bash
git clone <repository-url>
cd trade_genius_agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure environment**
```bash
cp env_example.txt .env
# Edit .env with your API keys and settings
```

4. **Run the application**
```bash
python main.py
```

5. **Access the web interface**
Open your browser and go to: `http://localhost:8000`

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Zerodha Kite Connect API
KITE_API_KEY=your_kite_api_key
KITE_API_SECRET=your_kite_api_secret
KITE_ACCESS_TOKEN=your_kite_access_token

# Email Configuration
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_USERNAME=your_email@gmail.com
EMAIL_PASSWORD=your_app_password
RECIPIENT_EMAIL=recipient@gmail.com

# Trading Configuration
DEFAULT_STOCKS=TCS.NS,INFY.NS,RELIANCE.NS
BUY_THRESHOLD_PERCENT=5.0
SELL_THRESHOLD_PERCENT=10.0
MAX_POSITION_SIZE=10000
STOP_LOSS_PERCENT=5.0

# Monitoring Configuration
PRICE_CHECK_INTERVAL=60
REPORT_TIME=18:00
TIMEZONE=Asia/Kolkata
```

### Trading Parameters

- **DEFAULT_STOCKS**: Comma-separated list of stock symbols to monitor
- **BUY_THRESHOLD_PERCENT**: Percentage drop to trigger buy signals
- **SELL_THRESHOLD_PERCENT**: Percentage gain to trigger sell signals
- **MAX_POSITION_SIZE**: Maximum amount to invest per position
- **STOP_LOSS_PERCENT**: Stop-loss percentage for risk management

## 📱 Web Interface

### Dashboard Features

1. **Real-time Metrics**
   - Portfolio value and P&L
   - Active positions count
   - Market status indicator

2. **Interactive Charts**
   - Portfolio performance over time
   - Market overview with indices
   - Individual stock price charts

3. **AI Predictions**
   - Next-day price forecasts
   - Confidence scores
   - Sentiment analysis results

4. **Trade Management**
   - Recent trade history
   - Manual trade execution
   - Position monitoring

### API Endpoints

- `GET /api/stocks/{symbol}/price` - Get current stock price
- `GET /api/stocks/{symbol}/chart` - Get price chart
- `GET /api/stocks/{symbol}/prediction` - Get AI prediction
- `GET /api/portfolio` - Get portfolio summary
- `POST /api/trades` - Execute manual trade
- `GET /api/market/overview` - Get market overview
- `POST /api/reports/email` - Send email report

## 🤖 AI Models

### Prediction Models

1. **ARIMA (AutoRegressive Integrated Moving Average)**
   - Time series forecasting
   - Automatic parameter selection
   - Confidence scoring

2. **LSTM (Long Short-Term Memory)**
   - Deep learning neural network
   - Pattern recognition
   - Non-linear relationships

3. **Technical Analysis**
   - Moving averages (SMA, EMA)
   - RSI, MACD, Bollinger Bands
   - Volume analysis

4. **Sentiment Analysis**
   - News sentiment scoring
   - TextBlob NLP processing
   - Multi-source aggregation

### Ensemble Method

The system combines all models using weighted averaging:
- LSTM: 40% weight
- ARIMA: 30% weight
- Technical: 20% weight
- Sentiment: 10% weight

## 📊 Trading Logic

### Signal Generation

1. **Technical Signals**
   - Moving average crossovers
   - RSI overbought/oversold
   - MACD signals
   - Bollinger Band touches

2. **Sentiment Signals**
   - Positive news sentiment → Buy signal
   - Negative news sentiment → Sell signal
   - Neutral sentiment → No action

3. **Risk Management**
   - Stop-loss triggers
   - Position size limits
   - Daily loss limits
   - Portfolio rebalancing

### Trade Execution

- **Automated Execution**: Based on AI signals with confidence > 70%
- **Manual Override**: Execute trades through web interface
- **Risk Controls**: Built-in position sizing and stop-losses
- **Order Management**: Track pending and executed orders

## 📧 Email Reports

### Daily Report Contents

1. **Portfolio Summary**
   - Total value and P&L
   - Active positions
   - Performance metrics

2. **Market Overview**
   - Major indices performance
   - Market sentiment
   - Economic indicators

3. **AI Predictions**
   - Next-day forecasts
   - Confidence levels
   - Sentiment analysis

4. **Trade Summary**
   - Executed trades
   - Pending orders
   - Performance analysis

5. **Charts and Visualizations**
   - Portfolio performance chart
   - Market overview chart
   - Prediction accuracy metrics

## 🔧 Advanced Configuration

### Broker Integration

The system is designed to work with Zerodha Kite Connect:

1. **Get API Credentials**
   - Register at [Zerodha Kite Connect](https://kite.trade/)
   - Generate API key and secret
   - Get access token

2. **Configure Trading**
   - Set up paper trading first
   - Configure position limits
   - Test with small amounts

### Database Setup

Default uses SQLite, but PostgreSQL is supported:

```env
# SQLite (default)
DATABASE_URL=sqlite:///./trade_genius.db

# PostgreSQL
DATABASE_URL=postgresql://user:password@localhost/trade_genius
```

### Custom Indicators

Add custom technical indicators in `data_fetcher.py`:

```python
def _add_custom_indicators(self, df):
    # Add your custom indicators here
    df['CUSTOM_INDICATOR'] = your_calculation(df)
    return df
```

## 🚨 Risk Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. 

- **No Financial Advice**: This is not financial advice
- **Use at Your Own Risk**: Trading involves substantial risk
- **Paper Trading First**: Test thoroughly before live trading
- **Start Small**: Begin with small position sizes
- **Monitor Closely**: Always supervise automated trading

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: Check this README and code comments
- **Issues**: Report bugs via GitHub issues
- **Discussions**: Join our community discussions
- **Email**: Contact support for urgent issues

## 🎯 Roadmap

### Upcoming Features

- [ ] **Multi-Broker Support** - Support for more brokers
- [ ] **Advanced ML Models** - Transformer models for prediction
- [ ] **Options Trading** - Support for options strategies
- [ ] **Mobile App** - Native mobile application
- [ ] **Social Trading** - Copy trading features
- [ ] **Backtesting** - Historical strategy testing
- [ ] **Paper Trading** - Simulated trading environment

### Version History

- **v1.0.0** - Initial release with core features
- **v1.1.0** - Enhanced AI models and web interface
- **v1.2.0** - Broker integration and risk management
- **v2.0.0** - Multi-asset support and advanced analytics

---

**Built with ❤️ for the trading community**

*Trade Genius Agent - Where AI meets Trading*
