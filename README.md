# Fantasy Football PPR Prediction Models

A comprehensive machine learning system for predicting PPR (Points Per Reception) fantasy football performance across **Running Backs (RB)**, **Wide Receivers/Tight Ends (WR/TE)**, and **Quarterbacks (QB)**. The system uses linear regression with Monte Carlo simulation, enhanced by advanced mathematical techniques including Minimax Theory and Markov Chains.

## 📊 Overview

This system consists of three specialized models, each trained on position-specific statistics and enhanced with sophisticated post-processing techniques to provide accurate fantasy football projections.



## 🏃‍♂️ Running Back (RB) Model

### **Data Sources**
- **Weekly Rushing Stats**: `RushingStats/RB01.csv` through `RB35.csv` (2024-2025 seasons)
- **Weekly PPR Data**: `RBPPR2024.csv` and `RBPPR2025.csv`
- **Key Metrics**: Attempts, Yards, Touchdowns, Efficiency, 8+ Defender Box Rate

### **PPR Scoring System**
```
PPR = 1.0 × REC + 0.1 × YDS + 6.0 × TD + 0.1 × RUSH_YDS + 6.0 × RUSH_TD
```

### **Core Features**

#### **Primary Rushing Metrics**
- **ATT (Attempts)**: Volume indicator - very useful
- **YDS (Yards)**: Production metric - very useful  
- **AVG (Yards/Attempt)**: Efficiency indicator - very useful
- **TD (Touchdowns)**: Red zone utilization - useful
- **EFF (Efficiency)**: Forward progress indicator - maybe useful
- **8+D% (8+ Defenders in Box)**: Success against stacked defenses - could be useful

#### **Receiving Upside Calculation**
```python
RECV_UPSIDE = weekly_PPR - rushing_PPR
rushing_PPR = 0.1 × YDS + 6.0 × TD
```

#### **Advanced Feature Engineering**
- **Lagged Features**: Previous week performance (`ATT_lag1`, `YDS_lag1`, etc.)
- **Rolling Averages**: 3, 5, and 8-week moving averages
- **Momentum Indicators**: Short and medium-term trend analysis
- **Volume/Efficiency Trends**: Yards per attempt and touchdown rates
- **Consistency Features**: Rushing appearance rates and performance stability
- **PPR Performance Adjustment**: Penalties for poor performance, rewards for excellence
- **Tier-based Features**: Performance tier classification and trend analysis

### **Model Architecture**
- **Base Model**: Ridge Regression with comprehensive feature set
- **Enhancement Layer**: Minimax Theory + Markov Chains post-processing
- **Training Data**: 34 weeks (2024 weeks 1-18, 2025 weeks 1-16)
- **Target Variable**: Weekly PPR points (total rushing + receiving)

### **Usage**
```bash
# Generate enhanced RB projections
python3 generate_enhanced_projections.py

# Get individual prediction
python3 enhanced_rb_predictor.py --player "Derrick Henry" --show_details
```

---

## 🎯 Wide Receiver/Tight End (WR/TE) Model

### **Data Sources**
- **Weekly Receiving Stats**: `RecievingStats/Re01.csv` through `Re35.csv`
- **Key Metrics**: Targets, Receptions, Yards, Touchdowns, Air Yards, YAC

### **PPR Scoring System**
```
PPR = 1.0 × REC + 0.1 × YDS + 6.0 × TD
```

### **Core Features**

#### **Primary Receiving Metrics**
- **TAR (Targets)**: Volume indicator - critical
- **REC (Receptions)**: Catch rate and volume - critical
- **YDS (Yards)**: Production metric - critical
- **TD (Touchdowns)**: Red zone usage - important
- **TAY (Total Air Yards)**: Deep target indicator - useful
- **YAC/R (Yards After Catch per Reception)**: After-catch ability - useful

#### **Advanced Feature Engineering**
- **Target Volume Indicators**: High-volume receiver flags
- **Efficiency Metrics**: Yards per target, catch rate trends
- **Air Yards Analysis**: Deep threat classification
- **Consistency Features**: Target appearance rates and performance stability
- **Late Season Indicators**: Playoff push performance trends
- **Momentum Analysis**: Recent surge detection and trend analysis
- **Elite Performer Classification**: Multi-dimensional performance tiers

### **Model Architecture**
- **Base Model**: Ridge Regression with 100+ engineered features
- **Enhancement Layer**: Reduced-impact Minimax Theory + Markov Chains (30% scaling)
- **Training Data**: 35 weeks of receiving statistics
- **Target Variable**: Weekly PPR points

### **Usage**
```bash
# Train the WR/TE model
python3 train_model_fixed.py --data_dir ./RecievingStats --outfile_dir ./artifacts_fixed

# Generate enhanced projections
python3 generate_enhanced_wr_te_projections.py

# Get individual prediction
python3 enhanced_wr_te_predictor.py --player "Cooper Kupp" --show_details
```

---

## 🏈 Quarterback (QB) Model

### **Data Sources**
- **Weekly Passing Stats**: `PassingStats/QB01.csv` through `QB35.csv`
- **Weekly PPR Data**: `QBPPR2024.csv` and `QBPPR2025.csv`
- **Key Metrics**: Attempts, Yards, Touchdowns, Interceptions, Passer Rating, Completion %

### **PPR Scoring System**
```
PPR = 0.04 × YDS + 4 × Passing_TD - 2 × INT + Rushing_UPSIDE
```

### **Core Features**

#### **Primary Passing Metrics**
- **ATT (Attempts)**: Volume indicator - useful
- **YDS (Yards)**: Production metric - useful
- **TD (Touchdowns)**: Scoring ability - very useful
- **INT (Interceptions)**: Mistake proneness - very useful
- **RATE (Passer Rating)**: Overall efficiency - very useful
- **COMP% (Completion %)**: Accuracy - very useful
- **XCOMP% (Expected Completion %)**: Pass quality - useful
- **TT (Time to Throw)**: O-line quality indicator - useful
- **AGG% (Aggressiveness %)**: Risk-taking tendency - maybe useful

#### **Rushing Upside Calculation**
```python
PASSING_PPR = 0.04 × YDS + 4 × TD - 2 × INT
RUSHING_UPSIDE = weekly_PPR - PASSING_PPR
```

#### **Advanced Feature Engineering**
- **Passing Volume Indicators**: High-attempt QB classification
- **Efficiency Metrics**: Completion rate trends, passer rating momentum
- **Consistency Features**: Passing appearance rates and performance stability
- **Mobile QB Detection**: High rushing upside identification
- **Performance Adjustment**: Penalties for poor performance, rewards for excellence

### **Model Architecture**
- **Base Model**: Ridge Regression with SimpleImputer + StandardScaler pipeline
- **Enhancement Layer**: Minimax Theory + Markov Chains + Mobile QB Correction
- **Training Data**: 35 weeks of passing and PPR statistics
- **Target Variable**: Weekly total PPR points (passing + rushing)

### **Usage**
```bash
# Train the QB model
python3 train_qb_model.py --data_dir ./PassingStats --ppr_2024 QBPPR2024.csv --ppr_2025 QBPPR2025.csv --outfile_dir ./artifacts_qb

# Generate enhanced projections
python3 generate_enhanced_qb_projections.py

# Get individual prediction
python3 enhanced_qb_predictor.py --player "Josh Allen" --show_details
```

---

## 🧮 Mathematical Foundations

### **Linear Regression Base**
All models use **Ridge Regression** with cross-validation for regularization:

```python
model = Pipeline([
    ("imputer", SimpleImputer(strategy='mean')),
    ("scaler", StandardScaler()),
    ("ridge", RidgeCV(alphas=np.logspace(-3,3,13), cv=5))
])
```

### **Monte Carlo Simulation**
Generates probability distributions for PPR thresholds:

```python
def monte_carlo_simulation(base_prediction, residual_std, n_simulations=10000):
    # Generate random outcomes based on prediction and residual standard deviation
    outcomes = np.random.normal(base_prediction, residual_std, n_simulations)
    
    # Calculate probabilities for different PPR thresholds
    prob_15_plus = np.mean(outcomes >= 15)
    prob_20_plus = np.mean(outcomes >= 20)
    prob_25_plus = np.mean(outcomes >= 25)
    
    return {
        'mean': np.mean(outcomes),
        'std': np.std(outcomes),
        'prob_15_plus': prob_15_plus,
        'prob_20_plus': prob_20_plus,
        'prob_25_plus': prob_25_plus
    }
```

### **Minimax Theory**
Minimizes maximum possible regret by considering worst-case scenarios:

```python
def minimax_decision(player_data):
    recent_performance = player_data['PPR'].tail(5)
    
    # Define regret scenarios
    worst_case = recent_performance.quantile(0.2)
    best_case = recent_performance.quantile(0.8)
    mean_case = recent_performance.mean()
    
    # Apply conservative adjustment to minimize maximum regret
    if prediction > mean_case:
        return prediction - 0.05 * (prediction - mean_case)  # Conservative
    else:
        return prediction + 0.03 * (mean_case - prediction)  # Optimistic
```

### **Markov Chains**
Models performance state transitions for trend prediction:

```python
def markov_chain_model(player_data):
    # Discretize performance into states
    performance_states = pd.cut(player_data['PPR'], bins=5, labels=['Poor', 'Below Avg', 'Average', 'Good', 'Excellent'])
    
    # Calculate transition probability matrix
    transition_matrix = pd.crosstab(performance_states[:-1], performance_states[1:], normalize='index')
    
    # Predict next state based on current state and transition probabilities
    current_state = performance_states.iloc[-1]
    next_state_probs = transition_matrix.loc[current_state]
    predicted_state = next_state_probs.idxmax()
    
    return map_state_to_ppr_range(predicted_state)
```

---

## 📁 Project Structure

```
FantasyFootball/
├── README.md                          # This documentation
├── 
├── # WR/TE Model Files
├── train_model_fixed.py               # WR/TE model training
├── predict_player_fixed.py            # WR/TE individual predictions
├── enhanced_wr_te_predictor.py        # Enhanced WR/TE predictions
├── generate_enhanced_wr_te_projections.py  # Generate all WR/TE projections
├── artifacts_fixed/                   # WR/TE model artifacts
├── 
├── # QB Model Files  
├── train_qb_model.py                  # QB model training
├── enhanced_qb_predictor.py           # Enhanced QB predictions
├── generate_enhanced_qb_projections.py # Generate all QB projections
├── artifacts_qb/                      # QB model artifacts
├── 
├── # RB Model Files
├── enhanced_rb_predictor.py           # Enhanced RB predictions
├── generate_enhanced_projections.py   # Generate all RB projections
├── artifacts_rb2/                     # RB model artifacts
├── 
├── # Data Directories
├── RecievingStats/                    # WR/TE weekly stats (Re01.csv - Re35.csv)
├── PassingStats/                      # QB weekly stats (QB01.csv - QB35.csv)
├── RushingStats/                      # RB weekly stats (RB01.csv - RB35.csv)
├── 
├── # PPR Data Files
├── RBPPR2024.csv                      # RB weekly PPR 2024
├── RBPPR2025.csv                      # RB weekly PPR 2025
├── QBPPR2024.csv                      # QB weekly PPR 2024
├── QBPPR2025.csv                      # QB weekly PPR 2025
├── 
├── # Output Directories
├── enhanced_projections/              # RB projections and visualizations
├── enhanced_qb_projections/           # QB projections and visualizations
├── enhanced_wr_te_projections/        # WR/TE projections and visualizations
└── plots_fixed/                       # WR/TE visualizations
```

---

## 🚀 Quick Start Guide

### **1. Train All Models**
```bash
# Train WR/TE model
python3 train_model_fixed.py --data_dir ./RecievingStats --outfile_dir ./artifacts_fixed --min_games 3 --test_weeks 4

# Train QB model  
python3 train_qb_model.py --data_dir ./PassingStats --ppr_2024 QBPPR2024.csv --ppr_2025 QBPPR2025.csv --outfile_dir ./artifacts_qb --min_games 3 --test_weeks 4

# RB model uses pre-trained artifacts (artifacts_rb2/)
```

### **2. Generate Enhanced Projections**
```bash
# Generate all RB projections with visualizations
python3 generate_enhanced_projections.py

# Generate all QB projections with visualizations
python3 generate_enhanced_qb_projections.py

# Generate all WR/TE projections with visualizations
python3 generate_enhanced_wr_te_projections.py
```

### **3. Get Individual Predictions**
```bash
# RB prediction
python3 enhanced_rb_predictor.py --player "Derrick Henry" --show_details

# QB prediction
python3 enhanced_qb_predictor.py --player "Josh Allen" --show_details

# WR/TE prediction
python3 enhanced_wr_te_predictor.py --player "Cooper Kupp" --show_details
```

---

## 📈 Model Performance & Validation

### **Cross-Validation Results**
- **WR/TE Model**: R² = 0.067 (test), MAE = 4.2 PPR
- **QB Model**: R² = 0.171 (test), MAE = 6.5 PPR  
- **RB Model**: Enhanced with post-processing for improved accuracy

### **Feature Importance Analysis**
Each model provides detailed feature importance rankings:
- **Top WR/TE Features**: Target volume, efficiency metrics, momentum indicators
- **Top QB Features**: Passing volume, efficiency, rushing upside, consistency
- **Top RB Features**: Volume metrics, efficiency, receiving upside, consistency

### **Monte Carlo Probability Outputs**
All models generate probability distributions for key PPR thresholds:
- **15+ PPR**: Solid fantasy starter probability
- **20+ PPR**: High-end fantasy starter probability  
- **25+ PPR**: Elite fantasy performance probability

---

## 🔧 Advanced Configuration

### **Model Parameters**
```python
# Ridge Regression Alpha Range
alphas = np.logspace(-3, 3, 13)  # 0.001 to 1000

# Cross-validation folds
cv_folds = 5

# Rolling average windows
roll_windows = [3, 5, 8]  # Short, medium, long-term trends

# Monte Carlo simulations
n_simulations = 10000
```

### **Enhancement Layer Tuning**
```python
# Minimax adjustment scaling
minimax_scaling = 0.05  # Conservative adjustments

# Markov chain impact
markov_scaling = 0.3    # 30% of raw Markov prediction

# Mobile QB correction threshold
correction_threshold = 1.5  # 50% underprediction triggers correction
max_correction_factor = 1.8  # Maximum 1.8x correction
```

---

## 📊 Output Formats

### **CSV Projections**
Each model generates comprehensive CSV files with:
- Base predictions
- Enhanced predictions  
- Individual adjustment components
- Monte Carlo probabilities
- Confidence intervals

### **Visualization Outputs**
- **Bar charts**: Top 30 player projections
- **Distribution analysis**: Prediction spread and confidence
- **Adjustment breakdown**: Component-wise enhancement analysis
- **Comparison plots**: Base vs enhanced predictions

---

## 🎯 Key Insights & Best Practices

### **Model Selection Guidelines**
- **RB Model**: Best for volume-based RBs with receiving upside
- **WR/TE Model**: Optimal for target-dependent receivers with efficiency metrics
- **QB Model**: Ideal for mobile QBs with significant rushing upside

### **Prediction Interpretation**
- **High Confidence**: Recent consistent performance with stable metrics
- **Medium Confidence**: Volatile performance with mixed signals
- **Low Confidence**: Inconsistent data or limited recent appearances

### **Feature Engineering Philosophy**
- **Volume First**: Target volume (WR/TE), attempts (RB/QB) are primary predictors
- **Efficiency Matters**: Yards per target/attempt provide upside indicators
- **Consistency Rewarded**: Regular performers get reliability bonuses
- **Momentum Captured**: Recent trends influence short-term projections

---

## 🤝 Contributing & Extensions

### **Adding New Features**
1. Update feature engineering in training scripts
2. Retrain models with expanded feature sets
3. Validate performance improvements
4. Update enhancement layers if needed

### **Model Improvements**
- **Ensemble Methods**: Combine multiple base models
- **Deep Learning**: Neural networks for complex pattern recognition
- **External Data**: Weather, injury reports, defensive rankings
- **Advanced Time Series**: LSTM/GRU for temporal dependencies

### **Performance Optimization**
- **Feature Selection**: Remove redundant or low-importance features
- **Hyperparameter Tuning**: Grid search for optimal model parameters
- **Cross-Validation**: More sophisticated validation strategies
- **Regularization**: Elastic Net or other regularization techniques

---

## 📞 Support & Documentation

For questions about model implementation, feature engineering, or mathematical foundations, refer to the individual model files and their comprehensive inline documentation. Each script includes detailed comments explaining the mathematical approaches and implementation details.

**Model Files with Detailed Documentation:**
- `train_model_fixed.py` - WR/TE model training and feature engineering
- `train_qb_model.py` - QB model training and rushing upside integration  
- `enhanced_rb_predictor.py` - RB enhancement algorithms and Minimax/Markov implementation
- `enhanced_qb_predictor.py` - QB enhancement with mobile QB correction
- `enhanced_wr_te_predictor.py` - WR/TE enhancement with reduced Markov impact

---

## 🌐 Web Application

A modern, responsive web interface is available for easy access to all models:

### **Quick Start**
```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app
python3 run_app.py

# Open browser to: http://localhost:5000
```

### **Features**
- **Auto-complete search** with 290+ players
- **Multi-model predictions** (RB, QB, WR/TE)
- **Detailed breakdowns** with enhancement components
- **Recent performance analysis**
- **Responsive design** for all devices

### **Web App Documentation**
See `WEB_APP_README.md` for detailed web application documentation.

---

*This system represents a comprehensive approach to fantasy football prediction, combining traditional statistical methods with advanced mathematical techniques to provide accurate, interpretable projections across all skill positions. The web application makes these sophisticated models accessible through an intuitive, modern interface.*