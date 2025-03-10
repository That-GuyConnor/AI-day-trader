# components/data_collector.py
import feedparser
import pandas as pd
from datetime import datetime
import logging
import threading
import concurrent.futures
import time

class RSSDataCollector:
    def __init__(self, config_manager):
        """Initialize the RSS data collector."""
        self.config_manager = config_manager
        self.logger = logging.getLogger('RSSDataCollector')
        self.timeout = 10  # Timeout in seconds for each feed
    
    def _parse_single_feed(self, feed_info):
        """Parse a single RSS feed with timeout using threading."""
        url = feed_info['url']
        name = feed_info.get('name', url)
        result = []
        error = None
        
        def parse_feed():
            nonlocal result, error
            try:
                feed = feedparser.parse(url)
                if hasattr(feed, 'status') and feed.status != 200 and feed.status != 301:
                    error = f"Feed returned status code {feed.status}"
                    return
                    
                if not hasattr(feed, 'entries') or len(feed.entries) == 0:
                    error = "Feed returned no entries"
                    return
                    
                for entry in feed.entries:
                    result.append({
                        'title': entry.get('title', ''),
                        'summary': entry.get('summary', ''),
                        'published': entry.get('published', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                        'link': entry.get('link', ''),
                        'source': name
                    })
            except Exception as e:
                error = str(e)
        
        thread = threading.Thread(target=parse_feed)
        thread.daemon = True
        thread.start()
        thread.join(self.timeout)
        
        if thread.is_alive():
            self.logger.warning(f"Feed {name} timed out after {self.timeout} seconds")
            return []
        
        if error:
            self.logger.warning(f"Error parsing feed {name}: {error}")
            return []
        
        self.logger.info(f"Successfully fetched {len(result)} entries from {name}")
        return result
    
    def collect_rss_data(self):
        """Collect data from RSS feeds with parallel processing and error handling."""
        # Get feed configurations which include name and url
        feeds = self.config_manager.get_rss_feeds_with_names()
        self.logger.info(f"Attempting to collect data from {len(feeds)} RSS feeds")
        
        all_entries = []
        successful_feeds = 0
        
        # Process feeds with a timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_feed = {executor.submit(self._parse_single_feed, feed): feed for feed in feeds}
            
            for future in concurrent.futures.as_completed(future_to_feed):
                feed = future_to_feed[future]
                try:
                    entries = future.result()
                    if len(entries) > 0:  # Check length instead of truthiness
                        all_entries.extend(entries)
                        successful_feeds += 1
                except Exception as e:
                    self.logger.error(f"Exception processing feed {feed.get('name', 'unknown')}: {e}")
        
        # Convert to DataFrame for easier processing
        df = pd.DataFrame(all_entries)
        if not df.empty:
            # Add utc=True to fix the FutureWarning about mixed time zones
            df['published'] = pd.to_datetime(df['published'], errors='coerce', utc=True)
            df = df.sort_values('published', ascending=False)
        
        self.logger.info(f"Successfully collected from {successful_feeds}/{len(feeds)} feeds with {len(df)} total news items")
        return df
    
    def filter_news_by_ticker(self, news_df, ticker):
        """Filter news relevant to a specific ticker."""
        if news_df.empty:
            return pd.DataFrame()
        
        # Filter news that mentions the ticker in title or summary
        ticker_news = news_df[
            news_df['title'].str.contains(ticker, case=False, na=False) | 
            news_df['summary'].str.contains(ticker, case=False, na=False)
        ]
        
        return ticker_news
