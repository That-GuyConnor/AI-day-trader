import logging
import time
import alpaca_trade_api as tradeapi
import os
import sys
from logging.handlers import RotatingFileHandler

# Add the current directory to the path so we can import our components
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from components.config_manager import ConfigManager
from components.data_collector import RSSDataCollector
from components.ai_model import SentimentAnalyzer
from components.technical_analysis import TechnicalAnalyzer
from components.trading_strategy import TradingStrategy
from components.order_executor import OrderExecutor

def setup_logging(config):
    """Set up logging based on configuration."""
    log_config = config.get_logging_config()
    
    # Configure logging
    log_level = getattr(logging, log_config['level'])
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler if enabled
    if log_config['log_to_file']:
        file_handler = RotatingFileHandler(
            log_config['log_file'],
            maxBytes=log_config['max_log_size'],
            backupCount=log_config['backup_count']
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return logging.getLogger('TradingBot')

def main():
    # Load configuration
    config = ConfigManager()
    
    # Set up logging
    logger = setup_logging(config)
    logger.info("Starting AI Trading Bot")
    
    try:
        # Initialize Alpaca API
        credentials = config.get_alpaca_credentials()
        api = tradeapi.REST(
            key_id=credentials['api_key'],
            secret_key=credentials['api_secret'],
            base_url=credentials['base_url'],
            api_version='v2'
        )
        
        # Verify connection
        account = api.get_account()
        logger.info(f"Connected to Alpaca API. Account status: {account.status}")
        logger.info(f"Account buying power: ${float(account.buying_power)}, cash: ${float(account.cash)}")
        
        # Initialize components
        data_collector = RSSDataCollector(config)
        sentiment_analyzer = SentimentAnalyzer(config.get_model_config())
        technical_analyzer = TechnicalAnalyzer(api)
        trading_strategy = TradingStrategy(config)
        order_executor = OrderExecutor(api, config)
        
        # Get tickers to monitor
        tickers = config.get_tickers()
        logger.info(f"Monitoring {len(tickers)} tickers: {', '.join(tickers)}")
        
        # Get technical indicators configuration
        indicators_config = config.get_model_config()['technical_analysis']['indicators']
        
        # Trading loop
        while True:
            try:
                # Check if market is open
                if not config.is_market_open():
                    logger.info("Market is closed. Waiting...")
                    time.sleep(60)  # Check every minute
                    continue
                
                # Collect RSS data
                news_df = data_collector.collect_rss_data()
                
                # Process each ticker
                for ticker in tickers:
                    # Filter news for this ticker
                    ticker_news = data_collector.filter_news_by_ticker(news_df, ticker)
                    
                    if not ticker_news.empty:
                        logger.info(f"Found {len(ticker_news)} news items for {ticker}")
                        
                        # Prepare text for sentiment analysis
                        texts = [f"{row['title']}: {row['summary']}" for _, row in ticker_news.iterrows()]
                        
                        # Analyze sentiment
                        sentiment_results = sentiment_analyzer.analyze_sentiment(texts)
                        
                        # Calculate technical indicators
                        technical_indicators = technical_analyzer.calculate_technical_indicators(ticker, indicators_config)
                        
                        # Make trading decision only if we have technical indicators
                        if technical_indicators and len(technical_indicators) > 0:
                            # Make trading decision
                            decision = trading_strategy.make_trading_decision(ticker, sentiment_results, technical_indicators)
                            
                            # IMPORTANT: Remove confidence threshold completely and add explicit logging
                            if decision['action'] != 'HOLD':
                                logger.info(f"MAIN: Decision made to {decision['action']} {ticker} - EXPLICITLY CALLING ORDER EXECUTION")
                                trade_result = order_executor.execute_trade_with_retry(ticker, decision)
                                if trade_result:
                                    logger.info(f"MAIN: Trade executed successfully: {trade_result}")
                                else:
                                    logger.error(f"MAIN: *** TRADE EXECUTION FAILED FOR {ticker} ***")
                            else:
                                logger.info(f"MAIN: Decision was to HOLD {ticker}, no trade executed")
                        else:
                            logger.warning(f"Skipping trading decision for {ticker} due to missing technical indicators")
                    else:
                        logger.info(f"No relevant news found for {ticker}")
                
                # Sleep until next check
                check_interval = config.get_check_interval()
                logger.info(f"Sleeping for {check_interval} seconds until next check")
                time.sleep(check_interval)
                
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)  # Wait before retrying
        
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
