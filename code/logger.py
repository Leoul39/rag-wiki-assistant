import logging
import os

# Create a logs directory if it doesn't exist
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Path for the log file
LOG_FILE = os.path.join(LOG_DIR, "rag_process.log")

# Basic logger configuration
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),       # Print logs to console
        logging.FileHandler(LOG_FILE)  # Save logs to file
    ]
)

# Optional: create a logger instance for more flexibility
logger = logging.getLogger(__name__)