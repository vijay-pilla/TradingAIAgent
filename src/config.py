"""
Trade Genius Agent - Configuration Management
Handles all environment variables and application settings
"""

import os
from typing import List, Optional
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Zerodha Kite Connect API
    kite_api_key: Optional[str] = Field(None, env="KITE_API_KEY")
    kite_api_secret: Optional[str] = Field(None, env="KITE_API_SECRET")
    kite_access_token: Optional[str] = Field(None, env="KITE_ACCESS_TOKEN")

    # Email Configuration
    smtp_server: str = Field("smtp.gmail.com", env="SMTP_SERVER")
    smtp_port: int = Field(587, env="SMTP_PORT")
    email_username: Optional[str] = Field(None, env="EMAIL_USERNAME")  # Not used - configured via UI
    email_password: Optional[str] = Field(None, env="EMAIL_PASSWORD")
    recipient_email: Optional[str] = Field(None, env="RECIPIENT_EMAIL")  # Not used - configured via UI
    
    # SendGrid Configuration
    sendgrid_api_key: Optional[str] = Field(None, env="SENDGRID_API_KEY")

    # WhatsApp Configuration (Twilio)
    whatsapp_enabled: bool = Field(False, env="WHATSAPP_ENABLED")
    twilio_account_sid: Optional[str] = Field(None, env="TWILIO_ACCOUNT_SID")
    twilio_auth_token: Optional[str] = Field(None, env="TWILIO_AUTH_TOKEN")
    whatsapp_recipient: Optional[str] = Field(None, env="WHATSAPP_RECIPIENT")  # Phone number with country code

    # Database Configuration
    database_url: str = Field("sqlite:///./trade_genius.db", env="DATABASE_URL")
    
    # Trading Configuration
    default_stocks: str = Field("TCS.NS,INFY.NS,RELIANCE.NS,HDFCBANK.NS,ICICIBANK.NS,IDFC.NS", env="DEFAULT_STOCKS")
    buy_threshold_percent: float = Field(5.0, env="BUY_THRESHOLD_PERCENT")
    sell_threshold_percent: float = Field(10.0, env="SELL_THRESHOLD_PERCENT")
    max_position_size: float = Field(10000, env="MAX_POSITION_SIZE")
    stop_loss_percent: float = Field(5.0, env="STOP_LOSS_PERCENT")
    
    # Monitoring Configuration
    price_check_interval: int = Field(60, env="PRICE_CHECK_INTERVAL")  # seconds
    report_time: str = Field("18:00", env="REPORT_TIME")
    timezone: str = Field("Asia/Kolkata", env="TIMEZONE")
    
    # News & Sentiment
    news_api_key: Optional[str] = Field(None, env="NEWS_API_KEY")
    sentiment_weight: float = Field(0.3, env="SENTIMENT_WEIGHT")
    
    # Web Interface
    secret_key: str = Field("your-secret-key-change-this", env="SECRET_KEY")
    debug: bool = Field(True, env="DEBUG")
    host: str = Field("127.0.0.1", env="HOST")
    port: int = Field(8000, env="PORT")
    
    def get_default_stocks_list(self) -> List[str]:
        """Get default stocks as a list"""
        return [stock.strip() for stock in self.default_stocks.split(',')]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()

# Trading symbols mapping
STOCK_SYMBOLS = {
    "TCS": "TCS.NS",
    "INFY": "INFY.NS", 
    "RELIANCE": "RELIANCE.NS",
    "HDFC": "HDFC.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "ITC": "ITC.NS",
    "SBIN": "SBIN.NS",
    "IDFCBANK": "IDFC.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "AXISBANK": "AXISBANK.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS"
}

# Risk management parameters
RISK_PARAMS = {
    "max_daily_loss": 5000,  # Maximum daily loss in INR
    "max_positions": 10,     # Maximum number of open positions
    "position_size_limit": 0.1,  # Maximum 10% of portfolio per position
    "volatility_threshold": 0.05,  # 5% daily volatility threshold
}

# Notification settings
NOTIFICATION_SETTINGS = {
    "email_enabled": True,
    "sms_enabled": False,
    "telegram_enabled": False,
    "critical_alerts": True,
    "daily_reports": True,
    "trade_notifications": True,
}
