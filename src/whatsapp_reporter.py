"""
Trade Genius Agent - WhatsApp Reporter
Handles WhatsApp notifications and reports via Twilio
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
from twilio.rest import Client
from twilio.base.exceptions import TwilioException
from loguru import logger

from src.config import settings
from src.models import DashboardStats, EmailReport

class WhatsAppReporter:
    """Handles WhatsApp messaging via Twilio"""

    def __init__(self):
        try:
            import os
            account_sid = os.getenv('TWILIO_ACCOUNT_SID', settings.twilio_account_sid)
            auth_token = os.getenv('TWILIO_AUTH_TOKEN', settings.twilio_auth_token)
            self.client = Client(account_sid, auth_token)
            self.from_whatsapp = f"whatsapp:{settings.twilio_account_sid}"  # Twilio WhatsApp number
            self.to_whatsapp = f"whatsapp:{settings.whatsapp_recipient}" if settings.whatsapp_recipient else None
            self.enabled = settings.whatsapp_enabled and self.to_whatsapp is not None
        except Exception as e:
            logger.error(f"Failed to initialize WhatsApp reporter: {e}")
            self.enabled = False

    def test_connection(self) -> bool:
        """Test WhatsApp connection"""
        if not self.enabled:
            logger.warning("WhatsApp not enabled or configured")
            return False

        try:
            # Send a test message
            message = self.client.messages.create(
                body="Trade Genius Agent: WhatsApp connection test successful!",
                from_=self.from_whatsapp,
                to=self.to_whatsapp
            )
            logger.info(f"WhatsApp test message sent: {message.sid}")
            return True
        except TwilioException as e:
            logger.error(f"WhatsApp connection test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error testing WhatsApp: {e}")
            return False

    def send_trade_notification(self, symbol: str, trade_type: str, quantity: int, price: float) -> bool:
        """Send trade execution notification"""
        if not self.enabled:
            return False

        try:
            message = f"🚀 Trade Executed\n{symbol}: {trade_type} {quantity} @ ₹{price:.2f}\nTime: {datetime.now().strftime('%H:%M:%S')}"
            self._send_message(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send trade notification: {e}")
            return False

    def send_price_alert(self, symbol: str, current_price: float, threshold: float, alert_type: str) -> bool:
        """Send price alert notification"""
        if not self.enabled:
            return False

        try:
            direction = "above" if alert_type == "PRICE_ABOVE" else "below"
            message = f"⚠️ Price Alert\n{symbol}: ₹{current_price:.2f} ({direction} threshold ₹{threshold:.2f})"
            self._send_message(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send price alert: {e}")
            return False

    def send_daily_report(self, portfolio_data: Dict, market_data: Dict, predictions: List[Dict]) -> bool:
        """Send daily summary report"""
        if not self.enabled:
            return False

        try:
            report = self._create_daily_report_text(portfolio_data, market_data, predictions)
            self._send_message(report)
            return True
        except Exception as e:
            logger.error(f"Failed to send daily report: {e}")
            return False

    def send_error_notification(self, error_message: str) -> bool:
        """Send system error notification"""
        if not self.enabled:
            return False

        try:
            message = f"❌ System Error\n{error_message}\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            self._send_message(message)
            return True
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")
            return False

    def _send_message(self, body: str) -> None:
        """Send WhatsApp message"""
        try:
            message = self.client.messages.create(
                body=body,
                from_=self.from_whatsapp,
                to=self.to_whatsapp
            )
            logger.info(f"WhatsApp message sent: {message.sid}")
        except TwilioException as e:
            logger.error(f"Twilio error sending WhatsApp: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error sending WhatsApp: {e}")
            raise

    def _create_daily_report_text(self, portfolio_data: Dict, market_data: Dict, predictions: List[Dict]) -> str:
        """Create text-based daily report"""
        try:
            report_lines = [
                "📊 Daily Trading Report",
                f"Date: {datetime.now().strftime('%Y-%m-%d')}",
                "",
                "💼 Portfolio Summary:",
                f"• Total Value: ₹{portfolio_data.get('total_value', 0):,.0f}",
                f"• Total P&L: ₹{portfolio_data.get('total_pnl', 0):,.0f}",
                f"• Active Positions: {portfolio_data.get('total_positions', 0)}",
                "",
                "🌍 Market Overview:"
            ]

            if market_data.get('indices'):
                for name, data in market_data['indices'].items():
                    if data:
                        change = data.get('change_percent', 0)
                        report_lines.append(f"• {name}: ₹{data.get('price', 0):.2f} ({change:+.2f}%)")

            report_lines.extend([
                "",
                "🔮 AI Predictions:"
            ])

            for pred in predictions[:3]:  # Limit to top 3
                report_lines.append(f"• {pred.get('symbol', 'N/A')}: ₹{pred.get('predicted_price', 0):.2f} (Conf: {pred.get('confidence_score', 0):.2f})")

            report_lines.extend([
                "",
                "📈 Trade Genius Agent"
            ])

            return "\n".join(report_lines)
        except Exception as e:
            logger.error(f"Error creating daily report text: {e}")
            return "Error generating daily report"
