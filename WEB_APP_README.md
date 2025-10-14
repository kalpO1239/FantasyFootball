# Fantasy Football Web App

A modern, responsive web interface for the Fantasy Football PPR Prediction models.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Web App
```bash
python3 run_app.py
```

### 3. Open Your Browser
Navigate to: **http://localhost:5000**

## 🎯 Features

### **Player Search**
- **Auto-complete search** with 290+ available players
- **Instant results** as you type
- **Multi-position support** (players can appear in multiple models)

### **Detailed Predictions**
- **Base predictions** from the linear regression models
- **Enhanced predictions** with Minimax Theory and Markov Chains
- **Adjustment breakdown** showing individual enhancement components
- **Recent performance** analysis (last 5 games)

### **Multi-Model Support**
- **RB Model**: Volume-based predictions with receiving upside
- **QB Model**: Passing efficiency with mobile QB corrections
- **WR/TE Model**: Target volume and efficiency metrics

### **Visual Design**
- **Modern, responsive** design that works on all devices
- **Color-coded predictions** by position
- **Interactive elements** with smooth animations
- **Professional styling** with gradients and shadows

## 📱 Usage

### **Searching for Players**
1. Type a player name in the search box
2. Select from auto-complete suggestions
3. Click "Search" or press Enter
4. View detailed predictions and breakdowns

### **Understanding Results**
- **Base Prediction**: Raw model output
- **Enhanced Prediction**: Final prediction with adjustments
- **Adjustments**: Individual enhancement components
- **Recent Performance**: Historical context

### **Position-Specific Features**
- **RB**: Receiving upside analysis, rushing consistency
- **QB**: Mobile QB corrections, passing efficiency
- **WR/TE**: Target volume trends, efficiency metrics

## 🔧 Technical Details

### **Backend (Flask)**
- **RESTful API** with JSON responses
- **Error handling** for missing players/data
- **Model integration** with all three prediction systems
- **Performance optimization** with efficient data loading

### **Frontend (HTML/CSS/JavaScript)**
- **Vanilla JavaScript** (no frameworks required)
- **Responsive CSS Grid** and Flexbox layouts
- **Font Awesome icons** for visual elements
- **Smooth animations** and transitions

### **API Endpoints**
- `GET /api/players` - Get all available players
- `GET /api/search?q=query` - Search for players
- `GET /api/predict/<player_name>` - Get predictions for a player

## 🎨 Customization

### **Styling**
Edit `static/css/style.css` to customize:
- Color schemes
- Layout spacing
- Typography
- Animations

### **Functionality**
Edit `static/js/app.js` to modify:
- Search behavior
- Result display
- API interactions
- User interactions

### **Backend**
Edit `app.py` to add:
- New API endpoints
- Additional data processing
- Custom error handling
- Performance optimizations

## 📊 Example Output

### **Player: Josh Allen (QB)**
```
Base Prediction: 14.8 PPR
Enhanced Prediction: 26.4 PPR
Total Adjustment: +11.6 PPR

Adjustments:
- Mobile QB Correction: +11.6 PPR
- Minimax Theory: +0.3 PPR
- Markov Chain: +0.2 PPR
- Performance Penalty: -0.6 PPR

Recent Performance:
- Average PPR: 31.4
- Total Games: 33
- Recent PPR: 23.0, 12.2, 41.3
```

## 🚀 Deployment

### **Local Development**
```bash
python3 run_app.py
```

### **Production Deployment**
```bash
# Using Gunicorn
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app

# Using Docker
docker build -t fantasy-football-app .
docker run -p 5000:5000 fantasy-football-app
```

## 🔍 Troubleshooting

### **Common Issues**
1. **"Player not found"**: Check if player exists in training data
2. **"Model error"**: Ensure model artifacts are present
3. **"Import error"**: Install required dependencies
4. **"Port in use"**: Change port in run_app.py

### **Debug Mode**
Set `debug=True` in `app.py` for detailed error messages and auto-reload.

## 📈 Performance

- **290+ players** loaded efficiently
- **Sub-second response** times for predictions
- **Responsive design** works on mobile/desktop
- **Minimal dependencies** for fast loading

---

*The web app provides an intuitive interface to access the sophisticated ML models, making fantasy football predictions accessible to all users.*
