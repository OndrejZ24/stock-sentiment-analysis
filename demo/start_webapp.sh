#!/bin/bash

# Stock Sentiment Analysis Web App Startup Script

echo "=================================================="
echo "  Stock Sentiment Analysis - Web App Launcher"
echo "=================================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "📥 Installing required packages..."
    pip install -r requirements_webapp.txt
    echo "✅ Packages installed"
else
    echo "✅ Required packages already installed"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  WARNING: .env file not found!"
    echo "Please create a .env file with your Oracle database credentials:"
    echo ""
    echo "db-dsn=your_oracle_dsn"
    echo "db-username=your_username"
    echo "db-password=your_password"
    echo ""
    read -p "Press Enter to continue anyway, or Ctrl+C to exit..."
fi

echo ""
echo "🚀 Starting Flask web application..."
echo ""
echo "📊 Access the app at: http://localhost:5000"
echo "⚠️  Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python app.py
