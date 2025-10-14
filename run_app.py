#!/usr/bin/env python3
"""
Simple script to run the Fantasy Football Prediction Web App
"""

import sys
import os
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import and run the Flask app
from app import app

if __name__ == '__main__':
    print("🏈 Starting Fantasy Football PPR Prediction Web App...")
    print("📊 Available models: RB, QB, WR/TE")
    print("🌐 Open your browser to: http://localhost:5001")
    print("⏹️  Press Ctrl+C to stop the server")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
