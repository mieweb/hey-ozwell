import os
import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    cmd = [
        'python', 'download_drive_data.py.py',
        '--data-dir', './data',
        '--delete-after-extract'
    ]
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd='../', check=True, capture_output=True, text=True)
        logger.info("Command completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return

    cmd = [
        'python', 'download_eleven_labs.py',
        '--batch-size', '100',
        '--negative-phrase-csv', './negative_phrases.csv',
        '--data-dir', './data',
        '--split-factor', '.8',
        '--delete-after-extract'
    ]
    try:
        logger.info(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd='../', check=True, capture_output=True, text=True)
        logger.info("Command completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {e}")
        logger.error(f"Error output: {e.stderr}")

if __name__=='__main__':
    main()
    
    

        