"""
Trade Genius Agent - Database Models
Defines all database schemas and Pydantic models
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

Base = declarative_base()

# Database Models
class Stock(Base):
    """Stock information table"""
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True)
    name = Column(String(100))
    exchange = Column(String(10))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock")
    trades = relationship("Trade", back_populates="stock")
    predictions = relationship("Prediction", back_populates="stock")

class StockPrice(Base):
    """Historical stock prices"""
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    timestamp = Column(DateTime, index=True)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")

class Trade(Base):
    """Trade execution records"""
    __tablename__ = "trades"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    trade_type = Column(String(10))  # BUY/SELL
    quantity = Column(Integer)
    price = Column(Float)
    total_amount = Column(Float)
    status = Column(String(20))  # PENDING/EXECUTED/FAILED
    order_id = Column(String(50))
    executed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text)
    
    # Relationships
    stock = relationship("Stock", back_populates="trades")

class Prediction(Base):
    """AI predictions and forecasts"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"))
    prediction_date = Column(DateTime)
    predicted_price = Column(Float)
    confidence_score = Column(Float)
    model_used = Column(String(50))
    sentiment_score = Column(Float)
    technical_indicators = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="predictions")

class NewsSentiment(Base):
    """News sentiment analysis"""
    __tablename__ = "news_sentiment"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(20), index=True)
    headline = Column(Text)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))  # POSITIVE/NEGATIVE/NEUTRAL
    source = Column(String(100))
    published_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class Alert(Base):
    """Price alerts and notifications"""
    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(20), index=True)
    alert_type = Column(String(20))  # PRICE_ABOVE/PRICE_BELOW/VOLUME_SPIKE
    threshold_value = Column(Float)
    current_value = Column(Float)
    is_triggered = Column(Boolean, default=False)
    triggered_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserConfig(Base):
    """User-configurable settings"""
    __tablename__ = "user_configs"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True)  # e.g., 'default_stocks', 'buy_threshold_percent'
    value = Column(JSON)  # Store as JSON for lists/dicts
    description = Column(String(255))
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

# Pydantic Models for API
class StockCreate(BaseModel):
    symbol: str
    name: str
    exchange: str = "NSE"

class StockResponse(BaseModel):
    id: int
    symbol: str
    name: str
    exchange: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

class TradeCreate(BaseModel):
    stock_symbol: str
    trade_type: str
    quantity: int
    price: Optional[float] = None

class TradeResponse(BaseModel):
    id: int
    stock_symbol: str
    trade_type: str
    quantity: int
    price: float
    total_amount: float
    status: str
    order_id: Optional[str]
    executed_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class PredictionResponse(BaseModel):
    id: int
    stock_symbol: str
    prediction_date: datetime
    predicted_price: float
    confidence_score: float
    model_used: str
    sentiment_score: float
    created_at: datetime
    
    class Config:
        from_attributes = True

class AlertCreate(BaseModel):
    stock_symbol: str
    alert_type: str
    threshold_value: float

class AlertResponse(BaseModel):
    id: int
    stock_symbol: str
    alert_type: str
    threshold_value: float
    current_value: Optional[float]
    is_triggered: bool
    triggered_at: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True

class DashboardStats(BaseModel):
    total_trades: int
    successful_trades: int
    total_profit: float
    active_positions: int
    today_pnl: float
    portfolio_value: float
    top_performers: List[dict]
    recent_trades: List[TradeResponse]
    upcoming_predictions: List[PredictionResponse]

class EmailReport(BaseModel):
    subject: str
    recipient: str
    trades_summary: dict
    predictions_summary: dict
    portfolio_performance: dict
    chart_path: Optional[str]
    recommendations: List[str]

class UserConfigCreate(BaseModel):
    key: str
    value: dict  # JSON value
    description: str

class UserConfigResponse(BaseModel):
    id: int
    key: str
    value: dict
    description: str
    updated_at: datetime

    class Config:
        from_attributes = True

class SettingsUpdate(BaseModel):
    default_stocks: List[str] = Field(default_factory=list)
    buy_threshold_percent: float = 5.0
    sell_threshold_percent: float = 10.0
    max_position_size: float = 10000.0
    stop_loss_percent: float = 5.0
    email_username: Optional[str] = None
    email_password: Optional[str] = None
    recipient_email: Optional[str] = None
    whatsapp_enabled: bool = False
    twilio_account_sid: Optional[str] = None
    twilio_auth_token: Optional[str] = None
    whatsapp_recipient: Optional[str] = None
