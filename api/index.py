import os
import sys

# Add project root to path so backend can be imported
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backend.app import app

# This is the entry point for Vercel
# Vercel looks for 'app' or 'application' in the file
application = app
