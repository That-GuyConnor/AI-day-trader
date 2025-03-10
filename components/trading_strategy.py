# components/trading_strategy.py
import logging
import numpy as np

class TradingStrategy:
    def __init__(self, config_manager):
        """Initialize the trading strategy."""
        self.config_manager = config_manager
        self.logger = logging.getLogger('TradingStrategy')
        self.model_config = config_manager.get_model_config()
        self.sentiment_threshold = self.model_config['sentiment_analysis']['confidence_threshold']
    
    def make_trading_decision(self, ticker, sentiment_data, technical_indicators):
        """Make a trading decision based on sentiment and technical indicators."""
        # Default decision
        decision = 'HOLD'
        confidence = 0.0
        
        try:
            # Calculate average sentiment
            if sentiment_data and len(sentiment_data) > 0:
                # Filter out low confidence sentiments
                filtered_sentiments = [
                    s for s in sentiment_data 
                    if s['confidence'] >= self.sentiment_threshold
                ]
                
                if len(filtered_sentiments) > 0:
                    # Calculate sentiment score (-1 to 1)
                    sentiment_score = sum(
                        s['confidence'] * (1 if s['label'] == 'positive' else 
                                          -1 if s['label'] == 'negative' else 0)
                        for s in filtered_sentiments
                    ) / len(filtered_sentiments)
                    
                    # Check technical indicators
                    if 'RSI' in technical_indicators:
                        rsi = technical_indicators['RSI']['value']
                        overbought = technical_indicators['RSI']['overbought']
                        oversold = technical_indicators['RSI']['oversold']
                        
                        # Decision logic
                        if sentiment_score > 0.3 and rsi < overbought:
                            decision = 'BUY'
                            confidence = sentiment_score * (1 - rsi/100)
                        elif sentiment_score < -0.3 or rsi > overbought:
                            decision = 'SELL'
                            confidence = abs(sentiment_score) * (rsi/100)
                    
                    # Check MACD if available
                    elif 'MACD' in technical_indicators:
                        macd = technical_indicators['MACD']
                        
                        # MACD crossover (signal line crosses MACD line from below)
                        if macd['macd_line'] > macd['signal_line'] and sentiment_score > 0:
                            decision = 'BUY'
                            confidence = sentiment_score * abs(macd['histogram'])
                        # MACD crossover (signal line crosses MACD line from above)
                        elif macd['macd_line'] < macd['signal_line'] and sentiment_score < 0:
                            decision = 'SELL'
                            confidence = abs(sentiment_score) * abs(macd['histogram'])
                    else:
                        # Fallback to sentiment only
                        if sentiment_score > 0.5:
                            decision = 'BUY'
                            confidence = sentiment_score
                        elif sentiment_score < -0.5:
                            decision = 'SELL'
                            confidence = abs(sentiment_score)
            
            self.logger.info(f"Decision for {ticker}: {decision} (confidence: {confidence:.2f})")
            
        except Exception as e:
            self.logger.error(f"Error making trading decision for {ticker}: {e}")
        
        return {
            'action': decision,
            'confidence': confidence
        }
