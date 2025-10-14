#!/bin/bash

echo "🏈 Fantasy Football PPR Prediction Web App"
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python3 -c "import flask, pandas, numpy, sklearn, joblib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Some dependencies are missing. Installing..."
    pip install -r requirements.txt
fi

echo "✅ Dependencies ready"
echo ""

# Check if model artifacts exist
if [ ! -d "artifacts_rb2" ] || [ ! -d "artifacts_qb" ] || [ ! -d "artifacts_fixed" ]; then
    echo "⚠️  Model artifacts not found. Please ensure the following directories exist:"
    echo "   - artifacts_rb2/"
    echo "   - artifacts_qb/"
    echo "   - artifacts_fixed/"
    echo ""
    echo "You may need to train the models first using:"
    echo "   python3 train_model_fixed.py"
    echo "   python3 train_qb_model.py"
    echo ""
fi

echo "🚀 Starting web application..."
echo "📊 Available models: RB, QB, WR/TE"
echo "🌐 Open your browser to: http://localhost:5000"
echo "⏹️  Press Ctrl+C to stop the server"
echo ""

# Run the app
python3 run_app.py
