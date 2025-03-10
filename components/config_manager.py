# components/config_manager.py
import yaml
import os
from datetime import datetime
import pytz
import logging

class ConfigManager:
    def __init__(self, config_path="config.yaml"):
        """Initialize the configuration manager."""
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
        self.logger = logging.getLogger('ConfigManager')
    
    def _load_config(self):
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            raise Exception(f"Error loading configuration: {e}")
    
    def _validate_config(self):
        """Validate the configuration."""
        # Check for required sections
        required_sections = ['api', 'trading', 'data_sources', 'model', 'logging']
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate API credentials
        if not self.config['api']['alpaca']['api_key'] or not self.config['api']['alpaca']['api_secret']:
            raise ValueError("Alpaca API credentials are missing")
    
    def get_alpaca_credentials(self):
        """Get Alpaca API credentials."""
        return {
            'api_key': self.config['api']['alpaca']['api_key'],
            'api_secret': self.config['api']['alpaca']['api_secret'],
            'base_url': self.config['api']['alpaca']['base_url']
        }
    
    def get_tickers(self):
        """Get list of tickers to trade."""
        return self.config['trading']['tickers']
    
    def get_rss_feeds(self):
        """Get list of RSS feed URLs."""
        return [feed['url'] for feed in self.config['data_sources']['rss_feeds']]
    
    def get_rss_feeds_with_names(self):
        """Get list of RSS feeds with names and URLs."""
        return self.config['data_sources']['rss_feeds']
    
    def get_risk_parameters(self):
        """Get risk management parameters."""
        return self.config['trading']['risk']
    
    def get_model_config(self):
        """Get AI model configuration."""
        return self.config['model']
    
    def get_logging_config(self):
        """Get logging configuration."""
        return self.config['logging']
    
    def get_check_interval(self):
        """Get the interval for checking new data."""
        return self.config['trading']['schedule']['check_interval']
    
    def is_market_open(self):
        """Check if the market is currently open based on configuration."""
        # Get current time in Eastern Time
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Check if today is a trading day
        if now.weekday() not in self.config['trading']['schedule']['trading_days']:
            return False
        
        # Parse market hours
        market_open = datetime.strptime(self.config['trading']['schedule']['market_open'], "%H:%M").time()
        market_close = datetime.strptime(self.config['trading']['schedule']['market_close'], "%H:%M").time()
        
        # Check if current time is within market hours
        return market_open <= now.time() <= market_close
