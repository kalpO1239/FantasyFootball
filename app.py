#!/usr/bin/env python3
"""
Fantasy Football Prediction Web App
Flask backend for serving player predictions from all three models
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os
import joblib

# Add current directory to path to import our models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our prediction models
from enhanced_rb_predictor import enhanced_prediction as rb_enhanced_prediction
from enhanced_qb_predictor import enhanced_prediction as qb_enhanced_prediction
from enhanced_wr_te_predictor import enhanced_prediction as wr_te_enhanced_prediction

app = Flask(__name__)

# Model artifacts directories
ARTIFACTS_DIR = {
    'rb': Path('./artifacts_rb2'),
    'qb': Path('./artifacts_qb'),
    'wr_te': Path('./artifacts_fixed')
}

def get_available_players():
    """Get list of all available players from all models"""
    players = set()
    
    # Get RB players
    try:
        rb_table = pd.read_parquet(ARTIFACTS_DIR['rb'] / 'rb_training_table.parquet')
        players.update(rb_table['PLAYER'].unique())
    except:
        pass
    
    # Get QB players
    try:
        qb_table = pd.read_parquet(ARTIFACTS_DIR['qb'] / 'qb_training_table.parquet')
        players.update(qb_table['PLAYER'].unique())
    except:
        pass
    
    # Get WR/TE players
    try:
        wr_te_table = pd.read_parquet(ARTIFACTS_DIR['wr_te'] / 'wrte_fixed_training_table.parquet')
        players.update(wr_te_table['PLAYER'].unique())
    except:
        pass
    
    return sorted(list(players))

def get_player_position(player_name):
    """Determine player position based on available data"""
    positions = []
    
    # Check RB
    try:
        rb_table = pd.read_parquet(ARTIFACTS_DIR['rb'] / 'rb_training_table.parquet')
        if player_name in rb_table['PLAYER'].values:
            positions.append('RB')
    except:
        pass
    
    # Check QB
    try:
        qb_table = pd.read_parquet(ARTIFACTS_DIR['qb'] / 'qb_training_table.parquet')
        if player_name in qb_table['PLAYER'].values:
            positions.append('QB')
    except:
        pass
    
    # Check WR/TE
    try:
        wr_te_table = pd.read_parquet(ARTIFACTS_DIR['wr_te'] / 'wrte_fixed_training_table.parquet')
        if player_name in wr_te_table['PLAYER'].values:
            positions.append('WR/TE')
    except:
        pass
    
    return positions

def get_base_prediction(player_name, position):
    """Get base prediction from the trained model"""
    try:
        if position == 'RB':
            model_dict = joblib.load(ARTIFACTS_DIR['rb'] / 'rb_model.joblib')
            table = pd.read_parquet(ARTIFACTS_DIR['rb'] / 'rb_training_table.parquet')
        elif position == 'QB':
            model_dict = joblib.load(ARTIFACTS_DIR['qb'] / 'qb_model.joblib')
            table = pd.read_parquet(ARTIFACTS_DIR['qb'] / 'qb_training_table.parquet')
        else:  # WR/TE
            model_dict = joblib.load(ARTIFACTS_DIR['wr_te'] / 'wrte_fixed_model.joblib')
            table = pd.read_parquet(ARTIFACTS_DIR['wr_te'] / 'wrte_fixed_training_table.parquet')
        
        model = model_dict['model']
        X_cols = model_dict['X_cols']
        
        # Use appropriate week column based on position
        week_col = 'WEEK' if position in ['WR/TE', 'WR', 'TE'] else 'GLOBAL_WEEK'
        
        if week_col not in table.columns:
            return None
            
        player_data = table[table['PLAYER'] == player_name].sort_values(week_col)
        if len(player_data) == 0:
            return None
        
        # Get latest data for prediction
        latest_data = player_data.iloc[-1]
        X_latest = latest_data[X_cols].values.reshape(1, -1)
        
        # Make prediction
        base_prediction = model.predict(X_latest)[0]
        return base_prediction
        
    except Exception as e:
        print(f"Error getting base prediction for {player_name} ({position}): {e}")
        return None

def get_recent_performance(player_name, position):
    """Get recent performance data for a player"""
    try:
        if position == 'RB':
            table = pd.read_parquet(ARTIFACTS_DIR['rb'] / 'rb_training_table.parquet')
        elif position == 'QB':
            table = pd.read_parquet(ARTIFACTS_DIR['qb'] / 'qb_training_table.parquet')
        else:  # WR/TE
            table = pd.read_parquet(ARTIFACTS_DIR['wr_te'] / 'wrte_fixed_training_table.parquet')
        
        # Use appropriate week column based on position
        week_col = 'WEEK' if position in ['WR/TE', 'WR', 'TE'] else 'GLOBAL_WEEK'
        
        if week_col not in table.columns:
            return None
            
        player_data = table[table['PLAYER'] == player_name].sort_values(week_col)
        if len(player_data) == 0:
            return None
        
        recent = player_data.tail(5)
        return {
            'weeks': recent[week_col].tolist(),
            'ppr': recent['PPR'].tolist(),
            'avg_ppr': recent['PPR'].mean(),
            'total_games': len(player_data)
        }
    except:
        return None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/api/players')
def api_players():
    """API endpoint to get all available players"""
    players = get_available_players()
    return jsonify(players)

@app.route('/api/search')
def api_search():
    """API endpoint to search for players"""
    query = request.args.get('q', '').lower()
    if not query:
        return jsonify([])
    
    all_players = get_available_players()
    matches = [p for p in all_players if query in p.lower()]
    return jsonify(matches[:10])  # Limit to 10 results

@app.route('/api/rankings/<position>')
def api_rankings(position):
    """API endpoint to get player rankings by position with pagination"""
    try:
        # Get page parameter (default to 1)
        page = int(request.args.get('page', 1))
        per_page = 10
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        if position.upper() == 'RB':
            artifacts_dir = ARTIFACTS_DIR['rb']
            model_file = 'rb_model.joblib'
            table_file = 'rb_training_table.parquet'
        elif position.upper() == 'QB':
            artifacts_dir = ARTIFACTS_DIR['qb']
            model_file = 'qb_model.joblib'
            table_file = 'qb_training_table.parquet'
        elif position.upper() in ['WR', 'TE', 'WR/TE', 'WRTE']:
            artifacts_dir = ARTIFACTS_DIR['wr_te']
            model_file = 'wrte_fixed_model.joblib'
            table_file = 'wrte_fixed_training_table.parquet'
        else:
            return jsonify({'error': 'Invalid position'}), 400
        
        # Load model and data
        model_dict = joblib.load(artifacts_dir / model_file)
        table = pd.read_parquet(artifacts_dir / table_file)
        model = model_dict['model']
        X_cols = model_dict['X_cols']
        
        # Get unique players and their latest data
        players = table['PLAYER'].unique()
        rankings = []
        
        # Use appropriate week column based on position
        week_col = 'WEEK' if position.upper() in ['WR', 'TE', 'WR/TE', 'WRTE'] else 'GLOBAL_WEEK'
        
        for player in players:
            if week_col not in table.columns:
                continue
                
            player_data = table[table['PLAYER'] == player].sort_values(week_col)
            if len(player_data) == 0:
                continue
            
            try:
                # Get latest data for prediction
                latest_data = player_data.iloc[-1]
                X_latest = latest_data[X_cols].values.reshape(1, -1)
                
                # Make base prediction
                base_prediction = model.predict(X_latest)[0]
                
                # Get enhanced prediction
                if position.upper() == 'RB':
                    # RB function returns tuple (prediction, details)
                    enhanced_pred, details = rb_enhanced_prediction(player, base_prediction, artifacts_dir)
                    enhanced = {
                        'enhanced_prediction': enhanced_pred,
                        'total_adjustment': enhanced_pred - base_prediction
                    }
                elif position.upper() == 'QB':
                    enhanced = qb_enhanced_prediction(player, base_prediction, artifacts_dir)
                else:  # WR/TE
                    enhanced = wr_te_enhanced_prediction(player, base_prediction, artifacts_dir)
                
                if 'error' not in enhanced:
                    rankings.append({
                        'player': player,
                        'base_prediction': base_prediction,
                        'enhanced_prediction': enhanced['enhanced_prediction'],
                        'total_adjustment': enhanced.get('total_adjustment', 0)
                    })
            except:
                continue
        
        # Sort by enhanced prediction
        rankings.sort(key=lambda x: x['enhanced_prediction'], reverse=True)
        
        # Apply pagination
        total_players = len(rankings)
        paginated_rankings = rankings[start_idx:end_idx]
        
        return jsonify({
            'players': paginated_rankings,
            'pagination': {
                'current_page': page,
                'per_page': per_page,
                'total_players': total_players,
                'total_pages': (total_players + per_page - 1) // per_page,
                'has_next': end_idx < total_players,
                'has_prev': page > 1
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<player_name>')
def api_predict(player_name):
    """API endpoint to get predictions for a player"""
    try:
        # Get player positions
        positions = get_player_position(player_name)
        
        if not positions:
            return jsonify({'error': f'Player "{player_name}" not found in any model'}), 404
        
        results = {
            'player': player_name,
            'positions': positions,
            'predictions': {}
        }
        
        # Get predictions for each position
        for position in positions:
            try:
                # Get base prediction first
                base_prediction = get_base_prediction(player_name, position)
                if base_prediction is None:
                    results['predictions'][position] = {'error': 'Could not get base prediction'}
                    continue
                
                # Get enhanced prediction
                if position == 'RB':
                    # RB function returns tuple (prediction, details)
                    enhanced_pred, details = rb_enhanced_prediction(player_name, base_prediction, ARTIFACTS_DIR['rb'])
                    prediction = {
                        'enhanced_prediction': enhanced_pred,
                        'base_prediction': base_prediction,
                        'adjustments': {
                            'minimax': details.get('minimax_prediction', 0) - base_prediction,
                            'markov': details.get('markov_prediction', 0) - base_prediction,
                            'performance_penalty': -details.get('performance_penalty', 0)
                        },
                        'total_adjustment': enhanced_pred - base_prediction
                    }
                elif position == 'QB':
                    prediction = qb_enhanced_prediction(player_name, base_prediction, ARTIFACTS_DIR['qb'])
                else:  # WR/TE
                    prediction = wr_te_enhanced_prediction(player_name, base_prediction, ARTIFACTS_DIR['wr_te'])
                
                if 'error' not in prediction:
                    # Get recent performance
                    recent_perf = get_recent_performance(player_name, position)
                    
                    results['predictions'][position] = {
                        'base_prediction': prediction['base_prediction'],
                        'enhanced_prediction': prediction['enhanced_prediction'],
                        'adjustments': prediction.get('adjustments', {}),
                        'total_adjustment': prediction.get('total_adjustment', 0),
                        'recent_performance': recent_perf
                    }
                else:
                    results['predictions'][position] = {'error': prediction['error']}
                    
            except Exception as e:
                results['predictions'][position] = {'error': str(e)}
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
