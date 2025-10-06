"""
Trade Genius Agent - Main FastAPI Application
Beautiful web interface for the AI trading agent
"""

import os
os.environ["DATABASE_URL"] = "sqlite:///./trade_genius.db"

from fastapi import FastAPI, Request, Depends, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket
import uvicorn
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import asyncio
from contextlib import asynccontextmanager
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy import create_engine
import json

from src.config import settings
from src.data_fetcher import DataFetcher, StockDataNotFound
from src.trading_logic import TradingEngine
from src.predictor import AIPredictor
from src.charting import ChartGenerator
from src.models import *
from src.scheduler import TradingScheduler
from src.emailer import EmailReporter
from src.whatsapp_reporter import WhatsAppReporter

import logging
logger = logging.getLogger(__name__)

# Force SQLite for local development
settings.database_url = "sqlite:///./trade_genius.db"

# Database setup
if "sqlite" in settings.database_url:
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
else:
    engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Global instances
data_fetcher = None
trading_engine = None
predictor = None
chart_generator = None
scheduler = None
email_reporter = None
whatsapp_reporter = None

def get_db():
    """Database dependency"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_watchlist() -> List[str]:
    """Get current watchlist from database, fallback to default settings"""
    db = SessionLocal()
    try:
        stock_config = db.query(UserConfig).filter(UserConfig.key == "default_stocks").first()
        if stock_config and stock_config.value:
            if isinstance(stock_config.value, list):
                return [stock.strip() for stock in stock_config.value]
            elif isinstance(stock_config.value, str):
                return [stock.strip() for stock in stock_config.value.split(',')]
        return settings.get_default_stocks_list()
    finally:
        db.close()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global data_fetcher, trading_engine, predictor, chart_generator, scheduler, email_reporter, whatsapp_reporter

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Initialize components
    data_fetcher = DataFetcher()
    trading_engine = TradingEngine(data_fetcher)
    predictor = AIPredictor(data_fetcher)
    chart_generator = ChartGenerator(data_fetcher)
    email_reporter = EmailReporter()
    whatsapp_reporter = WhatsAppReporter()

    # Start scheduler
    scheduler = TradingScheduler(trading_engine, predictor, email_reporter, get_current_watchlist)
    scheduler.start()

    yield

    # Cleanup
    if scheduler:
        scheduler.stop()

class ExceptionHandlerMiddleware(BaseHTTPMiddleware):
    """Global exception handler middleware"""
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as exc:
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail, "error": True}
            )
        except Exception as exc:
            logger.exception(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error", "error": True}
            )

# Create FastAPI app
app = FastAPI(
    title="Trade Genius Agent",
    description="AI-Powered Trading Agent with Real-time Analysis",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(ExceptionHandlerMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# WebSocket endpoint for real-time updates (placed early for proper routing)
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    logger.info("WebSocket connection attempt from: %s", websocket.client.host if hasattr(websocket, 'client') else 'unknown')

    # Accept the connection immediately
    await websocket.accept()
    logger.info("WebSocket connection accepted")

    try:
        # Send a simple test message first
        try:
            await websocket.send_json({
                "type": "connection_established",
                "message": "WebSocket connected successfully",
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Failed to send initial WebSocket message: {e}")
            return

        while True:
            # Send real-time data every 30 seconds
            try:
                market_data = data_fetcher.get_market_overview()
                portfolio_data = trading_engine.get_portfolio_summary()

                # Custom JSON encoder to handle datetime objects recursively
                def serialize_datetime(obj):
                    if isinstance(obj, datetime):
                        return obj.isoformat()
                    elif isinstance(obj, dict):
                        return {k: serialize_datetime(v) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [serialize_datetime(item) for item in obj]
                    elif isinstance(obj, tuple):
                        return tuple(serialize_datetime(item) for item in obj)
                    else:
                        return obj

                # Serialize data to JSON string with custom serializer
                import json as json_lib
                data_to_send = {
                    "type": "update",
                    "market_data": market_data,
                    "portfolio_data": portfolio_data,
                    "timestamp": datetime.now().isoformat()
                }
                json_str = json_lib.dumps(data_to_send, default=lambda x: serialize_datetime(x))
                await websocket.send_text(json_str)
            except Exception as data_error:
                logger.error(f"Error getting data for WebSocket: {data_error}")
                # Send error message but keep connection alive
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Error retrieving data",
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as send_error:
                    logger.error(f"Failed to send error message to WebSocket: {send_error}")
                    break  # Exit the loop if we can't send

            await asyncio.sleep(30)

    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
            logger.info("WebSocket connection closed")
        except Exception as close_e:
            logger.error(f"WebSocket close error: {close_e}")

# Routes
@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request, db: Session = Depends(get_db)):
    """Settings configuration page"""
    try:
        # Load current settings from DB or defaults
        configs = db.query(UserConfig).all()
        settings_dict = {}
        for config in configs:
            settings_dict[config.key] = config.value

        # Merge with defaults
        current_settings = {
            "default_stocks": settings_dict.get("default_stocks", settings.get_default_stocks_list()),
            "buy_threshold_percent": settings_dict.get("buy_threshold_percent", settings.buy_threshold_percent),
            "sell_threshold_percent": settings_dict.get("sell_threshold_percent", settings.sell_threshold_percent),
            "max_position_size": settings_dict.get("max_position_size", settings.max_position_size),
            "stop_loss_percent": settings_dict.get("stop_loss_percent", settings.stop_loss_percent),
            "email_username": settings_dict.get("email_username", settings.email_username),
            "recipient_email": settings_dict.get("recipient_email", settings.recipient_email),
            "whatsapp_enabled": settings_dict.get("whatsapp_enabled", settings.whatsapp_enabled),
            "whatsapp_recipient": settings_dict.get("whatsapp_recipient", settings.whatsapp_recipient),
        }

        return templates.TemplateResponse("settings.html", {
            "request": request,
            "current_settings": current_settings
        })
    except Exception as e:
        logger.exception(f"Error loading settings page: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    """Main dashboard page"""
    try:
        # Load user configs for stocks
        stock_config = db.query(UserConfig).filter(UserConfig.key == "default_stocks").first()
        monitored_stocks = stock_config.value if stock_config else settings.get_default_stocks_list()

        # Get market overview
        market_data = data_fetcher.get_market_overview()
        
        # Get portfolio summary
        portfolio_data = trading_engine.get_portfolio_summary()
        
        # Get recent predictions
        recent_predictions = []
        for symbol in monitored_stocks:
            try:
                prediction = predictor.predict_price(symbol, 1)
                if prediction and prediction['predicted_price'] > 0:
                    recent_predictions.append(prediction)
            except Exception as pred_e:
                logger.warning(f"Prediction failed for {symbol}: {pred_e}")
                continue
        
        # Get recent trades from DB
        recent_trades = db.query(Trade).order_by(Trade.created_at.desc()).limit(10).all()
        recent_trades = [TradeResponse.from_orm(trade) for trade in recent_trades]
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "market_data": market_data,
            "portfolio_data": portfolio_data,
            "recent_predictions": recent_predictions[:5],
            "recent_trades": recent_trades,
            "monitored_stocks": monitored_stocks,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        logger.exception(f"Error loading dashboard: {e}")
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error": str(e)
        })

@app.get("/api/stocks/{symbol}/price")
async def get_stock_price(symbol: str):
    """Get current stock price"""
    try:
        price_data = data_fetcher.get_live_price(symbol)
        return price_data
    except StockDataNotFound as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/{symbol}/chart")
async def get_stock_chart(symbol: str, days: int = 90):
    """Get stock price chart"""
    try:
        chart_base64 = chart_generator.create_price_chart(symbol, days)
        return {"chart": chart_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/{symbol}/prediction")
async def get_stock_prediction(symbol: str, days: int = 1):
    """Get stock price prediction"""
    try:
        prediction = predictor.predict_price(symbol, days)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/stocks/{symbol}/sentiment")
async def get_stock_sentiment(symbol: str, days: int = 7):
    """Get news sentiment analysis"""
    try:
        sentiment_data = data_fetcher.get_news_sentiment(symbol, days)
        return {"sentiment_data": sentiment_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio")
async def get_portfolio():
    """Get portfolio summary"""
    try:
        portfolio_data = trading_engine.get_portfolio_summary()
        return portfolio_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/portfolio/chart")
async def get_portfolio_chart():
    """Get portfolio performance chart"""
    try:
        portfolio_data = trading_engine.get_portfolio_summary()
        chart_base64 = chart_generator.create_portfolio_chart(portfolio_data)
        return {"chart": chart_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/overview")
async def get_market_overview():
    """Get market overview"""
    try:
        market_data = data_fetcher.get_market_overview()
        return market_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/market/chart")
async def get_market_chart():
    """Get market overview chart"""
    try:
        market_data = data_fetcher.get_market_overview()
        chart_base64 = chart_generator.create_market_overview_chart(market_data)
        return {"chart": chart_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/trades")
async def create_trade(trade_data: TradeCreate):
    """Create a new trade"""
    try:
        # Create trade signal
        from trading_logic import TradeSignal, TradeType
        
        signal = TradeSignal(
            symbol=trade_data.stock_symbol,
            action=TradeType(trade_data.trade_type),
            quantity=trade_data.quantity,
            price=trade_data.price or 0,
            confidence=0.8,
            reason="Manual trade"
        )
        
        # Execute trade
        success = trading_engine.execute_trade(signal)
        
        if success:
            return {"message": "Trade executed successfully", "status": "success"}
        else:
            raise HTTPException(status_code=400, detail="Trade execution failed")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades"""
    try:
        # In a real implementation, this would fetch from database
        trades = []
        return {"trades": trades}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/alerts")
async def create_alert(alert_data: AlertCreate):
    """Create a price alert"""
    try:
        # In a real implementation, this would save to database
        return {"message": "Alert created successfully", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
async def get_alerts():
    """Get all alerts"""
    try:
        # In a real implementation, this would fetch from database
        alerts = []
        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reports/email")
async def send_email_report(background_tasks: BackgroundTasks):
    """Send email report"""
    try:
        background_tasks.add_task(email_reporter.send_daily_report)
        return {"message": "Email report queued", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reports/whatsapp")
async def send_whatsapp_report(background_tasks: BackgroundTasks):
    """Send WhatsApp report"""
    try:
        if not whatsapp_reporter.enabled:
            raise HTTPException(status_code=400, detail="WhatsApp not configured")

        background_tasks.add_task(whatsapp_reporter.send_daily_report,
                                trading_engine.get_portfolio_summary(),
                                data_fetcher.get_market_overview(),
                                [])
        return {"message": "WhatsApp report queued", "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings")
async def update_settings(settings_update: SettingsUpdate, db: Session = Depends(get_db)):
    """Update user settings"""
    try:
        # Update each setting in DB
        settings_map = {
            "default_stocks": settings_update.default_stocks,
            "buy_threshold_percent": settings_update.buy_threshold_percent,
            "sell_threshold_percent": settings_update.sell_threshold_percent,
            "max_position_size": settings_update.max_position_size,
            "stop_loss_percent": settings_update.stop_loss_percent,
            "email_username": settings_update.email_username,
            "email_password": settings_update.email_password,
            "recipient_email": settings_update.recipient_email,
            "whatsapp_enabled": settings_update.whatsapp_enabled,
            "twilio_account_sid": settings_update.twilio_account_sid,
            "twilio_auth_token": settings_update.twilio_auth_token,
            "whatsapp_recipient": settings_update.whatsapp_recipient,
        }

        for key, value in settings_map.items():
            if value is not None:
                # Upsert config
                config = db.query(UserConfig).filter(UserConfig.key == key).first()
                if config:
                    config.value = value
                else:
                    config = UserConfig(
                        key=key,
                        value=value,
                        description=f"User configurable {key.replace('_', ' ')}"
                    )
                    db.add(config)

        db.commit()
        return {"message": "Settings updated successfully", "status": "success"}
    except Exception as e:
        db.rollback()
        logger.exception(f"Error updating settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/settings")
async def get_settings(db: Session = Depends(get_db)):
    """Get current user settings"""
    try:
        configs = db.query(UserConfig).all()
        settings_dict = {config.key: config.value for config in configs}
        return {"settings": settings_dict}
    except Exception as e:
        logger.exception(f"Error fetching settings: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/performance")
async def get_performance_analytics():
    """Get performance analytics"""
    try:
        # Calculate performance metrics
        portfolio_data = trading_engine.get_portfolio_summary()
        
        # Get prediction accuracy for each stock
        accuracy_data = {}
        current_watchlist = get_current_watchlist()
        for symbol in current_watchlist:
            accuracy = predictor.get_prediction_accuracy(symbol)
            accuracy_data[symbol] = accuracy
        
        return {
            "portfolio_performance": portfolio_data,
            "prediction_accuracy": accuracy_data,
            "generated_at": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0",
        "components": {
            "data_fetcher": "active",
            "trading_engine": "active",
            "predictor": "active",
            "chart_generator": "active",
            "scheduler": "active" if scheduler else "inactive"
        }
    }

@app.get("/api/test-websocket")
async def test_websocket():
    """Test WebSocket endpoint availability"""
    return {
        "websocket_available": True,
        "websocket_url": "/ws",
        "message": "WebSocket endpoint is registered"
    }

@app.get("/api/stocks/search")
async def search_stocks(q: str = ""):
    """Search for stock symbols"""
    if not q:
        return {"stocks": []}
    
    # Common NSE stocks for search (expand as needed)
    common_stocks = [
        "TCS.NS", "INFY.NS", "RELIANCE.NS", "HDFCBANK.NS", "HDFC.NS", "ICICIBANK.NS",
        "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "HINDUNILVR.NS", "ITC.NS",
        "LT.NS", "ASIANPAINT.NS", "MARUTI.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
        "AXISBANK.NS", "TITAN.NS", "SUNPHARMA.NS", "WIPRO.NS", "NTPC.NS",
        "POWERGRID.NS", "TATAMOTORS.NS", "ONGC.NS", "TECHM.NS", "HCLTECH.NS"
    ]
    
    # Simple fuzzy search: match if query is substring (case insensitive)
    matching_stocks = [stock for stock in common_stocks if q.upper() in stock.upper()]
    
    return {"stocks": matching_stocks[:10]}



if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info"
    )
