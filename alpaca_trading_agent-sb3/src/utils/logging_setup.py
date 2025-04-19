# alpaca_trading_agent/src/utils/logging_setup.py
import logging
import logging.handlers
import os
import sys

# Determine project root assuming this file is in src/utils
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
LOG_FILE = os.path.join(LOG_DIR, 'pipeline.log')

def configure_file_logging(level_str="INFO"):
    """Configures logging to a rotating file."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger() # Get the root logger

    # --- Crucial: Set root logger level FIRST ---
    # This ensures handlers with lower levels are not ignored.
    logger.setLevel(level)

    # --- Remove existing handlers to avoid duplicates ---
    # Important if this function might be called multiple times or if
    # default handlers exist (like basicConfig might add).
    # Check specifically for file handlers to avoid removing console handlers potentially added elsewhere.
    # Or simply remove all handlers if each script should *only* have the file handler.
    # Let's remove all for simplicity in sub-scripts.
    for handler in logger.handlers[:]:
        # Check if it's our specific file handler to avoid issues if called multiple times
        # A simpler approach for sub-scripts is just to clear all handlers first.
        logger.removeHandler(handler)
        try:
            handler.close() # Close handlers before removing
        except Exception:
            pass # Ignore errors during close, e.g., if already closed

    # --- Create log directory ---
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
    except OSError as e:
        # Use basic print/stderr for logging setup errors
        print(f"Error creating log directory {LOG_DIR}: {e}", file=sys.stderr)
        return # Cannot proceed if log dir creation fails

    # --- File Handler (Rotating) ---
    try:
        file_handler = logging.handlers.RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8'
        )
        file_handler.setFormatter(log_formatter)
        # Set handler level explicitly (optional but good practice)
        # file_handler.setLevel(level)
        logger.addHandler(file_handler)
        # Use print for initial confirmation as logging might not be fully set yet
        # print(f"Process {os.getpid()}: File logging configured to {LOG_FILE} at level {level_str.upper()}", file=sys.stderr)
    except Exception as e:
        print(f"Process {os.getpid()}: Error setting up file handler for {LOG_FILE}: {e}", file=sys.stderr)

    # --- Set higher level for noisy libraries ---
    # Do this *after* setting the root logger level and adding handlers
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)

    # Initial log message *after* setup
    # Use logger.info, which will now go to the file if setup succeeded
    logger.info(f"--- File Logging Initialized in Process {os.getpid()} (Level: {level_str.upper()}) ---")
    logger.info(f"Logging to File: {LOG_FILE}")

def add_console_logging(level_str="INFO"):
    """Adds console (stderr) logging to the root logger IF it doesn't exist."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger() # Get the root logger

    # Check if a console handler already exists to avoid duplicates
    has_console_handler = any(isinstance(h, logging.StreamHandler) and h.stream in [sys.stdout, sys.stderr] for h in logger.handlers)

    if not has_console_handler:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setFormatter(log_formatter)
        # Set handler level explicitly
        # console_handler.setLevel(level)
        logger.addHandler(console_handler)
        logger.info(f"--- Console Logging Added (Level: {level_str.upper()}) ---")
    # else:
        # logger.debug("Console handler already exists.") # Avoid logging this every time