# components/model_downloader.py
from huggingface_hub import snapshot_download
import os
import logging

def download_model(model_name, cache_dir="./models"):
    """
    Automatically download a model from Hugging Face Hub.
    
    Args:
        model_name: Name of the model on Hugging Face (e.g., "finbert/finbert-sentiment")
        cache_dir: Directory to store the downloaded model
    
    Returns:
        Path to the downloaded model
    """
    logger = logging.getLogger('ModelDownloader')
    
    try:
        logger.info(f"Downloading model: {model_name}")
        # Create cache directory if it doesn't exist
        os.makedirs(cache_dir, exist_ok=True)
        
        # Download the model
        model_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            local_dir=os.path.join(cache_dir, model_name.split('/')[-1])
        )
        
        logger.info(f"Model downloaded successfully to: {model_path}")
        return model_path
    
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {e}")
        raise
