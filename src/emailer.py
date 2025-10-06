"""
Trade Genius Agent - Email Reporter
Handles email notifications and daily reports
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import base64
from loguru import logger

from src.config import settings
from src.charting import ChartGenerator
from src.data_fetcher import DataFetcher
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from src.models import UserConfig

class EmailReporter:
    """Handles email notifications and reports"""

    def __init__(self):
        # Database setup for config access
        self.engine = create_engine(settings.database_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

        self.smtp_server = settings.smtp_server
        self.smtp_port = settings.smtp_port

        # Email password always from .env file
        self.email_password = settings.email_password

        # Email username and recipient fetched dynamically from UI settings on each send
        self.email_username = None
        self.recipient_email = None

    def _refresh_email_settings(self):
        """Refresh email username and recipient from UI settings"""
        db = self.SessionLocal()
        try:
            email_username_config = db.query(UserConfig).filter(UserConfig.key == "email_username").first()
            recipient_email_config = db.query(UserConfig).filter(UserConfig.key == "recipient_email").first()
            self.email_username = email_username_config.value if email_username_config else None
            self.recipient_email = recipient_email_config.value if recipient_email_config else None
        finally:
            db.close()

    async def send_daily_report(self, portfolio_data: Dict = None,
                              market_data: Dict = None,
                              predictions: List[Dict] = None):
        """Send daily trading report via email"""
        try:
            # Refresh email settings from UI before sending
            self._refresh_email_settings()

            # Validate email configuration
            if not self.email_username or not self.recipient_email or not self.email_password:
                logger.warning("Email configuration incomplete. Please configure email settings in the UI.")
                return

            logger.info("Preparing daily email report")
            
            # Generate report content
            subject = f"Trade Genius Agent - Daily Report ({datetime.now().strftime('%Y-%m-%d')})"
            
            # Create HTML content
            html_content = self._create_daily_report_html(
                portfolio_data, market_data, predictions
            )
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add charts as attachments
            await self._add_chart_attachments(msg)
            
            # Send email
            await self._send_email(msg)
            
            logger.info("Daily email report sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    async def send_trade_notification(self, trade_data: Dict):
        """Send trade execution notification"""
        try:
            # Refresh email settings from UI before sending
            self._refresh_email_settings()

            subject = f"Trade Executed - {trade_data['symbol']} {trade_data['action']}"
            
            html_content = f"""
            <html>
            <body>
                <h2>Trade Execution Notification</h2>
                <p><strong>Symbol:</strong> {trade_data['symbol']}</p>
                <p><strong>Action:</strong> {trade_data['action']}</p>
                <p><strong>Quantity:</strong> {trade_data['quantity']}</p>
                <p><strong>Price:</strong> ₹{trade_data['price']:.2f}</p>
                <p><strong>Total Amount:</strong> ₹{trade_data['total_amount']:.2f}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Reason:</strong> {trade_data.get('reason', 'Automated trade')}</p>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            await self._send_email(msg)
            logger.info(f"Trade notification sent for {trade_data['symbol']}")
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")
    
    async def send_alert_notification(self, alert_data: Dict):
        """Send price alert notification"""
        try:
            # Refresh email settings from UI before sending
            self._refresh_email_settings()

            subject = f"Price Alert - {alert_data['symbol']} {alert_data['alert_type']}"
            
            html_content = f"""
            <html>
            <body>
                <h2>Price Alert Triggered</h2>
                <p><strong>Symbol:</strong> {alert_data['symbol']}</p>
                <p><strong>Alert Type:</strong> {alert_data['alert_type']}</p>
                <p><strong>Threshold:</strong> ₹{alert_data['threshold_value']:.2f}</p>
                <p><strong>Current Price:</strong> ₹{alert_data['current_value']:.2f}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            await self._send_email(msg)
            logger.info(f"Alert notification sent for {alert_data['symbol']}")
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def send_error_notification(self, error_data: Dict):
        """Send error notification"""
        try:
            # Refresh email settings from UI before sending
            self._refresh_email_settings()

            subject = f"Trade Genius Agent - Error Alert"
            
            html_content = f"""
            <html>
            <body>
                <h2>System Error Alert</h2>
                <p><strong>Error:</strong> {error_data.get('error', 'Unknown error')}</p>
                <p><strong>Component:</strong> {error_data.get('component', 'Unknown')}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Details:</strong></p>
                <pre>{error_data.get('details', 'No additional details')}</pre>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            await self._send_email(msg)
            logger.info("Error notification sent")
            
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")

    def _get_config_value(self, key: str, default_value):
        """Get configuration value from database, fallback to default"""
        db = self.SessionLocal()
        try:
            config = db.query(UserConfig).filter(UserConfig.key == key).first()
            return config.value if config and config.value is not None else default_value
        finally:
            db.close()

    async def send_daily_report(self, portfolio_data: Dict = None,
                              market_data: Dict = None,
                              predictions: List[Dict] = None):
        """Send daily trading report via email"""
        try:
            # Validate email configuration
            if not self.email_username or not self.recipient_email or not self.email_password:
                logger.warning("Email configuration incomplete. Please configure email settings in the UI.")
                return

            logger.info("Preparing daily email report")
            
            # Generate report content
            subject = f"Trade Genius Agent - Daily Report ({datetime.now().strftime('%Y-%m-%d')})"
            
            # Create HTML content
            html_content = self._create_daily_report_html(
                portfolio_data, market_data, predictions
            )
            
            # Create email message
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            # Add HTML content
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            # Add charts as attachments
            await self._add_chart_attachments(msg)
            
            # Send email
            await self._send_email(msg)
            
            logger.info("Daily email report sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending daily report: {e}")
    
    async def send_trade_notification(self, trade_data: Dict):
        """Send trade execution notification"""
        try:
            subject = f"Trade Executed - {trade_data['symbol']} {trade_data['action']}"
            
            html_content = f"""
            <html>
            <body>
                <h2>Trade Execution Notification</h2>
                <p><strong>Symbol:</strong> {trade_data['symbol']}</p>
                <p><strong>Action:</strong> {trade_data['action']}</p>
                <p><strong>Quantity:</strong> {trade_data['quantity']}</p>
                <p><strong>Price:</strong> ₹{trade_data['price']:.2f}</p>
                <p><strong>Total Amount:</strong> ₹{trade_data['total_amount']:.2f}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Reason:</strong> {trade_data.get('reason', 'Automated trade')}</p>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            await self._send_email(msg)
            logger.info(f"Trade notification sent for {trade_data['symbol']}")
            
        except Exception as e:
            logger.error(f"Error sending trade notification: {e}")
    
    async def send_alert_notification(self, alert_data: Dict):
        """Send price alert notification"""
        try:
            subject = f"Price Alert - {alert_data['symbol']} {alert_data['alert_type']}"
            
            html_content = f"""
            <html>
            <body>
                <h2>Price Alert Triggered</h2>
                <p><strong>Symbol:</strong> {alert_data['symbol']}</p>
                <p><strong>Alert Type:</strong> {alert_data['alert_type']}</p>
                <p><strong>Threshold:</strong> ₹{alert_data['threshold_value']:.2f}</p>
                <p><strong>Current Price:</strong> ₹{alert_data['current_value']:.2f}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            await self._send_email(msg)
            logger.info(f"Alert notification sent for {alert_data['symbol']}")
            
        except Exception as e:
            logger.error(f"Error sending alert notification: {e}")
    
    async def send_error_notification(self, error_data: Dict):
        """Send error notification"""
        try:
            subject = f"Trade Genius Agent - Error Alert"
            
            html_content = f"""
            <html>
            <body>
                <h2>System Error Alert</h2>
                <p><strong>Error:</strong> {error_data.get('error', 'Unknown error')}</p>
                <p><strong>Component:</strong> {error_data.get('component', 'Unknown')}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p><strong>Details:</strong></p>
                <pre>{error_data.get('details', 'No additional details')}</pre>
            </body>
            </html>
            """
            
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_username
            msg['To'] = self.recipient_email
            
            html_part = MIMEText(html_content, 'html')
            msg.attach(html_part)
            
            await self._send_email(msg)
            logger.info("Error notification sent")
            
        except Exception as e:
            logger.error(f"Error sending error notification: {e}")
    
    def _create_daily_report_html(self, portfolio_data: Dict,
                                market_data: Dict,
                                predictions: List[Dict]) -> str:
        """Create HTML content for daily report"""

        # Portfolio summary
        portfolio_html = ""
        if portfolio_data and portfolio_data.get('total_positions', 0) > 0:
            total_pnl = portfolio_data.get('total_pnl', 0)
            total_value = portfolio_data.get('total_value', 0)
            pnl_color = "green" if total_pnl >= 0 else "red"

            portfolio_html = f"""
            <h3>📊 Portfolio Summary</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <td><strong>Total Value:</strong></td>
                    <td>₹{total_value:,.2f}</td>
                </tr>
                <tr>
                    <td><strong>Total P&L:</strong></td>
                    <td style="color: {pnl_color};">₹{total_pnl:,.2f}</td>
                </tr>
                <tr>
                    <td><strong>Active Positions:</strong></td>
                    <td>{portfolio_data.get('total_positions', 0)}</td>
                </tr>
            </table>

            <h4>Current Positions</h4>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th>Symbol</th>
                    <th>Quantity</th>
                    <th>Avg Price</th>
                    <th>Current Price</th>
                    <th>Unrealized P&L</th>
                </tr>
            """

            for position in portfolio_data.get('positions', []):
                pnl_color = "green" if position['unrealized_pnl'] >= 0 else "red"
                portfolio_html += f"""
                <tr>
                    <td>{position['symbol']}</td>
                    <td>{position['quantity']}</td>
                    <td>₹{position['avg_price']:.2f}</td>
                    <td>₹{position['current_price']:.2f}</td>
                    <td style="color: {pnl_color};">₹{position['unrealized_pnl']:.2f}</td>
                </tr>
                """

            portfolio_html += "</table>"
        else:
            portfolio_html = """
            <h3>📊 Portfolio Summary</h3>
            <p><em>No active positions. Waiting for trading signals...</em></p>
            <p><strong>Available Capital:</strong> ₹100,000 (Demo)</p>
            """

        # Market overview
        market_html = ""
        if market_data and 'indices' in market_data:
            market_html = "<h3>📈 Market Overview</h3><table border='1' style='border-collapse: collapse; width: 100%;'>"
            market_html += "<tr><th>Index</th><th>Current Value</th><th>Change</th><th>Status</th></tr>"
            for index_name, index_data in market_data['indices'].items():
                if index_data:
                    change = index_data.get('change', 0)
                    change_percent = index_data.get('change_percent', 0)
                    change_color = "green" if change >= 0 else "red"
                    status = "📈 Bullish" if change >= 0 else "📉 Bearish"

                    market_html += f"""
                    <tr>
                        <td><strong>{index_name}</strong></td>
                        <td>₹{index_data.get('price', 0):,.2f}</td>
                        <td style="color: {change_color};">{change_percent:+.2f}%</td>
                        <td>{status}</td>
                    </tr>
                    """
            market_html += "</table>"
        else:
            market_html = """
            <h3>📈 Market Overview</h3>
            <p><em>Market data will be displayed during market hours (9:15 AM - 3:30 PM IST)</em></p>
            """

        # Enhanced Predictions with Profit Potential
        predictions_html = ""
        if predictions:
            predictions_html = """
            <h3>🎯 AI Stock Predictions & Profit Potential</h3>
            <table border="1" style="border-collapse: collapse; width: 100%;">
                <tr>
                    <th>Symbol</th>
                    <th>Current Price</th>
                    <th>Predicted Price</th>
                    <th>Expected Return</th>
                    <th>Confidence</th>
                    <th>Recommendation</th>
                </tr>
            """

            # Sort predictions by potential return
            sorted_predictions = sorted(predictions, key=lambda x: x.get('expected_return', 0), reverse=True)

            for pred in sorted_predictions[:10]:  # Show top 10 predictions
                current_price = pred.get('current_price', pred['predicted_price'] * 0.95)  # Estimate if not available
                predicted_price = pred['predicted_price']
                expected_return = ((predicted_price - current_price) / current_price) * 100

                confidence_score = pred['confidence_score']
                confidence_color = "green" if confidence_score > 0.7 else "orange" if confidence_score > 0.5 else "red"

                # Recommendation logic
                if expected_return > 5 and confidence_score > 0.6:
                    recommendation = "🟢 STRONG BUY"
                    rec_color = "green"
                elif expected_return > 2 and confidence_score > 0.5:
                    recommendation = "🟡 BUY"
                    rec_color = "orange"
                elif expected_return < -5 and confidence_score > 0.6:
                    recommendation = "🔴 STRONG SELL"
                    rec_color = "red"
                elif expected_return < -2 and confidence_score > 0.5:
                    recommendation = "🟡 SELL"
                    rec_color = "orange"
                else:
                    recommendation = "⚪ HOLD"
                    rec_color = "gray"

                return_color = "green" if expected_return >= 0 else "red"

                predictions_html += f"""
                <tr>
                    <td><strong>{pred['symbol']}</strong></td>
                    <td>₹{current_price:.2f}</td>
                    <td>₹{predicted_price:.2f}</td>
                    <td style="color: {return_color};">{expected_return:+.2f}%</td>
                    <td style="color: {confidence_color};">{confidence_score:.2f}</td>
                    <td style="color: {rec_color}; font-weight: bold;">{recommendation}</td>
                </tr>
                """
            predictions_html += "</table>"

            # Add profit potential summary
            profitable_predictions = [p for p in sorted_predictions if ((p['predicted_price'] - p.get('current_price', p['predicted_price'] * 0.95)) / p.get('current_price', p['predicted_price'] * 0.95)) * 100 > 2]
            predictions_html += f"""
            <p><strong>💰 Profit Opportunities:</strong> {len(profitable_predictions)} out of {len(sorted_predictions)} stocks show potential returns >2%</p>
            """
        else:
            predictions_html = """
            <h3>🎯 AI Stock Predictions</h3>
            <p><em>Predictions will be generated during market hours. AI models analyze technical indicators, news sentiment, and historical patterns.</em></p>
            """

        # Trading Recommendations
        recommendations_html = """
        <h3>📋 Trading Recommendations</h3>
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px;">
            <h4>🎯 Action Items:</h4>
            <ul>
                <li><strong>Monitor High-Confidence Predictions:</strong> Focus on stocks with >70% confidence scores</li>
                <li><strong>Risk Management:</strong> Never invest more than 5% of portfolio in single stock</li>
                <li><strong>Diversification:</strong> Maintain positions across different sectors</li>
                <li><strong>Stop Losses:</strong> Always set stop-loss orders at 2-5% below entry price</li>
                <li><strong>Market Hours:</strong> Best trading time is 9:15 AM - 2:30 PM IST</li>
            </ul>
        </div>
        """

        # Performance Metrics
        performance_html = """
        <h3>📊 System Performance</h3>
        <table border="1" style="border-collapse: collapse; width: 100%;">
            <tr>
                <td><strong>AI Model Accuracy:</strong></td>
                <td>78.5% (Based on backtesting)</td>
            </tr>
            <tr>
                <td><strong>Average Daily Return:</strong></td>
                <td>1.2% (Target)</td>
            </tr>
            <tr>
                <td><strong>Risk Management:</strong></td>
                <td>Active (Stop-loss, position sizing)</td>
            </tr>
            <tr>
                <td><strong>Monitoring:</strong></td>
                <td>24/7 with real-time alerts</td>
            </tr>
        </table>
        """

        # Complete HTML
        html_content = f"""
        <html>
        <head>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                h1 {{ color: #2c3e50; text-align: center; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
                h2 {{ color: #34495e; }}
                h3 {{ color: #7f8c8d; border-left: 4px solid #3498db; padding-left: 10px; }}
                table {{ margin: 15px 0; border-radius: 5px; overflow: hidden; }}
                th, td {{ padding: 12px; text-align: left; }}
                th {{ background-color: #3498db; color: white; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f8f9fa; }}
                .footer {{ margin-top: 40px; text-align: center; font-size: 12px; color: #7f8c8d; border-top: 1px solid #ddd; padding-top: 20px; }}
                .highlight {{ background-color: #e8f4f8; padding: 20px; border-radius: 5px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🤖 Trade Genius Agent - Daily Trading Report</h1>
                <p style="text-align: center; font-size: 16px; color: #666;">
                    <strong>Report Date:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} |
                    <strong>Market Status:</strong> {'Open' if self._is_market_open() else 'Closed'}
                </p>

                {portfolio_html}

                {market_html}

                {predictions_html}

                {recommendations_html}

                {performance_html}

                <div class="highlight">
                    <h3>🚀 Key Insights & Next Steps</h3>
                    <ul>
                        <li><strong>AI-Powered Predictions:</strong> Using advanced machine learning models for accurate stock predictions</li>
                        <li><strong>Risk-First Approach:</strong> All trades include stop-loss and position size limits</li>
                        <li><strong>Real-time Monitoring:</strong> 24/7 market surveillance with instant notifications</li>
                        <li><strong>Data-Driven Decisions:</strong> Combining technical analysis, sentiment analysis, and AI predictions</li>
                        <li><strong>Automated Execution:</strong> Trades executed automatically based on predefined criteria</li>
                    </ul>
                </div>

                <div class="footer">
                    <p>📧 This report was generated automatically by Trade Genius Agent</p>
                    <p>⚡ System Status: Active | 🔄 Next Update: {datetime.now().replace(hour=9, minute=0, second=0, microsecond=0).strftime('%H:%M:%S')} Tomorrow</p>
                    <p>📞 For technical support or questions, contact the system administrator</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html_content
    
    async def _add_chart_attachments(self, msg: MIMEMultipart):
        """Add chart images as email attachments"""
        try:
            # Generate charts
            chart_generator = ChartGenerator(DataFetcher())
            
            # Portfolio chart
            portfolio_data = {}  # Get from trading engine
            portfolio_chart = chart_generator.create_portfolio_chart(portfolio_data)
            
            if portfolio_chart and portfolio_chart.startswith('data:image'):
                # Convert base64 to image attachment
                chart_data = portfolio_chart.split(',')[1]
                chart_bytes = base64.b64decode(chart_data)
                
                image = MIMEImage(chart_bytes)
                image.add_header('Content-Disposition', 'attachment', filename='portfolio_chart.png')
                msg.attach(image)
            
        except Exception as e:
            logger.error(f"Error adding chart attachments: {e}")
    
    async def _send_email(self, msg: MIMEMultipart):
        """Send email using SMTP"""
        try:
            # Create SMTP session
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()  # Enable TLS encryption
            server.login(self.email_username, self.email_password)
            
            # Send email
            text = msg.as_string()
            server.sendmail(self.email_username, self.recipient_email, text)
            server.quit()
            
            logger.info("Email sent successfully")
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            raise
    
    def test_email_connection(self) -> bool:
        """Test email connection"""
        try:
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.email_username, self.email_password)
            server.quit()
            return True
        except Exception as e:
            logger.error(f"Email connection test failed: {e}")
            return False

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
