"""
Trade Genius Agent - Scheduler
Handles automated trading tasks and scheduling
"""

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from datetime import datetime, time
import asyncio
from typing import Dict, List
from loguru import logger

from src.config import settings
from src.trading_logic import TradingEngine
from src.predictor import AIPredictor
from src.emailer import EmailReporter

class TradingScheduler:
    """Handles all scheduled trading tasks"""

    def __init__(self, trading_engine: TradingEngine, predictor: AIPredictor, email_reporter: EmailReporter, get_watchlist_func=None):
        self.trading_engine = trading_engine
        self.predictor = predictor
        self.email_reporter = email_reporter
        self.get_watchlist_func = get_watchlist_func or (lambda: settings.get_default_stocks_list())
        self.scheduler = AsyncIOScheduler()
        self.is_running = False
        
    def start(self):
        """Start the scheduler"""
        try:
            if self.is_running:
                logger.warning("Scheduler is already running")
                return
            
            # Schedule price monitoring
            self.scheduler.add_job(
                self.monitor_prices,
                trigger=IntervalTrigger(seconds=settings.price_check_interval),
                id='price_monitoring',
                name='Monitor Stock Prices',
                replace_existing=True
            )
            
            # Schedule daily predictions
            self.scheduler.add_job(
                self.generate_daily_predictions,
                trigger=CronTrigger(hour=9, minute=0),  # 9:00 AM daily
                id='daily_predictions',
                name='Generate Daily Predictions',
                replace_existing=True
            )
            
            # Schedule daily email report
            report_time = settings.report_time.split(':')
            self.scheduler.add_job(
                self.send_daily_report,
                trigger=CronTrigger(hour=int(report_time[0]), minute=int(report_time[1])),
                id='daily_report',
                name='Send Daily Email Report',
                replace_existing=True
            )
            
            # Schedule portfolio rebalancing (weekly)
            self.scheduler.add_job(
                self.rebalance_portfolio,
                trigger=CronTrigger(day_of_week=0, hour=10, minute=0),  # Sunday 10:00 AM
                id='portfolio_rebalancing',
                name='Weekly Portfolio Rebalancing',
                replace_existing=True
            )
            
            # Schedule risk management check
            self.scheduler.add_job(
                self.risk_management_check,
                trigger=IntervalTrigger(minutes=15),  # Every 15 minutes
                id='risk_management',
                name='Risk Management Check',
                replace_existing=True
            )
            
            # Schedule market hours check
            self.scheduler.add_job(
                self.market_hours_check,
                trigger=IntervalTrigger(minutes=5),  # Every 5 minutes
                id='market_hours',
                name='Market Hours Check',
                replace_existing=True
            )
            
            self.scheduler.start()
            self.is_running = True
            logger.info("Trading scheduler started successfully")
            
        except Exception as e:
            logger.error(f"Error starting scheduler: {e}")
    
    def stop(self):
        """Stop the scheduler"""
        try:
            if self.scheduler.running:
                self.scheduler.shutdown()
            self.is_running = False
            logger.info("Trading scheduler stopped")
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")
    
    async def monitor_prices(self):
        """Monitor stock prices and execute trades"""
        try:
            logger.info("Starting price monitoring cycle")
            
            # Check if market is open
            if not self._is_market_open():
                logger.info("Market is closed, skipping price monitoring")
                return
            
            # Monitor each stock
            for symbol in self.get_watchlist_func():
                try:
                    # Generate trading signals
                    signals = self.trading_engine.analyze_and_generate_signals(symbol)
                    
                    # Execute trades based on signals
                    for signal in signals:
                        if signal.confidence >= 0.7:  # High confidence threshold
                            success = self.trading_engine.execute_trade(signal)
                            if success:
                                logger.info(f"Trade executed: {signal.action.value} {signal.quantity} {signal.symbol}")
                            else:
                                logger.warning(f"Trade execution failed: {signal.symbol}")
                    
                    # Check price alerts
                    current_data = self.trading_engine.data_fetcher.get_live_price(symbol)
                    if current_data:
                        alerts = self.trading_engine.check_price_alerts(symbol, current_data['price'])
                        for alert in alerts:
                            if alert.is_triggered:
                                logger.info(f"Alert triggered: {alert.alert_type} for {alert.stock_symbol}")
                                # Send notification (implement notification system)
                
                except Exception as e:
                    logger.error(f"Error monitoring {symbol}: {e}")
            
            logger.info("Price monitoring cycle completed")
            
        except Exception as e:
            logger.error(f"Error in price monitoring: {e}")
    
    async def generate_daily_predictions(self):
        """Generate daily predictions for all stocks"""
        try:
            logger.info("Generating daily predictions")

            predictions = []
            for symbol in self.get_watchlist_func():
                try:
                    # Generate 1-day and 5-day predictions
                    prediction_1d = self.predictor.predict_price(symbol, 1)
                    prediction_5d = self.predictor.predict_price(symbol, 5)
                    
                    predictions.extend([prediction_1d, prediction_5d])
                    
                    logger.info(f"Generated predictions for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Error generating predictions for {symbol}: {e}")
            
            # Store predictions (implement database storage)
            logger.info(f"Generated {len(predictions)} predictions")
            
        except Exception as e:
            logger.error(f"Error in daily predictions: {e}")
    
    async def send_daily_report(self):
        """Send daily email report"""
        try:
            logger.info("Sending daily email report")

            # Generate report data
            portfolio_data = self.trading_engine.get_portfolio_summary()
            market_data = self.trading_engine.data_fetcher.get_market_overview()

            # Get recent predictions with enhanced data
            recent_predictions = []
            for symbol in self.get_watchlist_func():
                try:
                    prediction = self.predictor.predict_price(symbol, 1)
                    if prediction['predicted_price'] > 0:
                        # Get current price
                        current_data = self.trading_engine.data_fetcher.get_live_price(symbol)
                        current_price = current_data['price'] if current_data else prediction['predicted_price'] * 0.95

                        # Calculate expected return
                        expected_return = ((prediction['predicted_price'] - current_price) / current_price) * 100

                        # Add enhanced fields
                        enhanced_prediction = prediction.copy()
                        enhanced_prediction.update({
                            'current_price': current_price,
                            'expected_return': expected_return
                        })

                        recent_predictions.append(enhanced_prediction)
                except Exception as e:
                    logger.error(f"Error getting prediction for {symbol}: {e}")

            # Sort predictions by expected return (highest first)
            recent_predictions.sort(key=lambda x: x.get('expected_return', 0), reverse=True)

            # Send email report
            await self.email_reporter.send_daily_report(
                portfolio_data=portfolio_data,
                market_data=market_data,
                predictions=recent_predictions
            )

            logger.info("Daily email report sent successfully")

        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    async def rebalance_portfolio(self):
        """Weekly portfolio rebalancing"""
        try:
            logger.info("Starting portfolio rebalancing")
            
            # Get current portfolio
            portfolio_data = self.trading_engine.get_portfolio_summary()
            
            # Calculate target allocation (equal weight for now)
            current_watchlist = self.get_watchlist_func()
            target_allocation = 1.0 / len(current_watchlist)
            
            # Check if rebalancing is needed
            rebalance_needed = False
            for position in portfolio_data.get('positions', []):
                current_weight = (position['current_price'] * position['quantity']) / portfolio_data.get('total_value', 1)
                if abs(current_weight - target_allocation) > 0.1:  # 10% deviation threshold
                    rebalance_needed = True
                    break
            
            if rebalance_needed:
                logger.info("Portfolio rebalancing required")
                # Implement rebalancing logic
                # This would involve selling overweight positions and buying underweight ones
            else:
                logger.info("Portfolio is balanced, no rebalancing needed")
            
        except Exception as e:
            logger.error(f"Error in portfolio rebalancing: {e}")
    
    async def risk_management_check(self):
        """Risk management monitoring"""
        try:
            # Check daily loss limits
            portfolio_data = self.trading_engine.get_portfolio_summary()
            daily_pnl = portfolio_data.get('realized_pnl', 0)
            
            if daily_pnl <= -5000:  # Daily loss limit
                logger.warning(f"Daily loss limit reached: {daily_pnl}")
                # Implement emergency stop logic
                # Close all positions or reduce exposure
            
            # Check position sizes
            for position in portfolio_data.get('positions', []):
                position_value = position['current_price'] * position['quantity']
                portfolio_value = portfolio_data.get('total_value', 1)
                position_weight = position_value / portfolio_value
                
                if position_weight > 0.2:  # 20% position limit
                    logger.warning(f"Position size limit exceeded for {position['symbol']}: {position_weight:.2%}")
                    # Implement position size reduction
            
        except Exception as e:
            logger.error(f"Error in risk management check: {e}")
    
    async def market_hours_check(self):
        """Check market hours and adjust trading behavior"""
        try:
            market_status = self._get_market_status()
            
            if market_status == "CLOSED":
                # Market is closed, reduce monitoring frequency
                logger.debug("Market is closed")
            elif market_status == "PRE_MARKET":
                # Pre-market hours, limited trading
                logger.debug("Pre-market hours")
            elif market_status == "AFTER_HOURS":
                # After hours, limited trading
                logger.debug("After hours trading")
            else:
                # Regular market hours
                logger.debug("Regular market hours")
                
        except Exception as e:
            logger.error(f"Error in market hours check: {e}")
    
    def _is_market_open(self) -> bool:
        """Check if market is currently open"""
        now = datetime.now()
        
        # Indian market hours: 9:15 AM to 3:30 PM IST, Monday to Friday
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return False
        
        # Check if it's within market hours
        return market_open <= now <= market_close
    
    def _get_market_status(self) -> str:
        """Get current market status"""
        now = datetime.now()
        
        # Market hours
        pre_market_start = now.replace(hour=9, minute=0, second=0, microsecond=0)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
        after_hours_end = now.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Check if it's a weekday
        if now.weekday() >= 5:  # Saturday or Sunday
            return "CLOSED"
        
        if pre_market_start <= now < market_open:
            return "PRE_MARKET"
        elif market_open <= now <= market_close:
            return "OPEN"
        elif market_close < now <= after_hours_end:
            return "AFTER_HOURS"
        else:
            return "CLOSED"
    
    def get_scheduler_status(self) -> Dict:
        """Get current scheduler status"""
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    'id': job.id,
                    'name': job.name,
                    'next_run': job.next_run_time.isoformat() if job.next_run_time else None,
                    'trigger': str(job.trigger)
                })
            
            return {
                'running': self.is_running,
                'jobs': jobs,
                'total_jobs': len(jobs)
            }
        except Exception as e:
            logger.error(f"Error getting scheduler status: {e}")
            return {'running': False, 'jobs': [], 'total_jobs': 0}
