// Fantasy Football PPR Predictions - Frontend JavaScript

class FantasyApp {
    constructor() {
        this.searchInput = document.getElementById('playerSearch');
        this.searchResults = document.getElementById('searchResults');
        this.searchBtn = document.getElementById('searchBtn');
        this.loadingSection = document.getElementById('loadingSection');
        this.resultsSection = document.getElementById('resultsSection');
        this.errorSection = document.getElementById('errorSection');
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.loadPlayers();
        this.loadInitialRankings();
    }
    
    setupEventListeners() {
        // Search input events
        this.searchInput.addEventListener('input', (e) => {
            this.handleSearchInput(e.target.value);
        });
        
        this.searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                e.preventDefault();
                this.performSearch();
            }
        });
        
        // Search button
        this.searchBtn.addEventListener('click', () => {
            this.performSearch();
        });
        
        // Click outside to close search results
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.search-container')) {
                this.hideSearchResults();
            }
        });
        
        // Tab button events
        document.querySelectorAll('.tab-button').forEach(button => {
            button.addEventListener('click', () => {
                this.switchTab(button.dataset.position);
            });
        });
    }
    
    async loadPlayers() {
        try {
            const response = await fetch('/api/players');
            this.players = await response.json();
        } catch (error) {
            console.error('Error loading players:', error);
        }
    }
    
    handleSearchInput(query) {
        if (query.length < 2) {
            this.hideSearchResults();
            return;
        }
        
        const matches = this.players.filter(player => 
            player.toLowerCase().includes(query.toLowerCase())
        ).slice(0, 10);
        
        this.showSearchResults(matches);
    }
    
    showSearchResults(matches) {
        if (matches.length === 0) {
            this.hideSearchResults();
            return;
        }
        
        this.searchResults.innerHTML = matches.map(player => 
            `<div class="search-result-item" data-player="${player}">${player}</div>`
        ).join('');
        
        // Add click listeners to result items
        this.searchResults.querySelectorAll('.search-result-item').forEach(item => {
            item.addEventListener('click', () => {
                this.searchInput.value = item.dataset.player;
                this.hideSearchResults();
                this.performSearch();
            });
        });
        
        this.searchResults.style.display = 'block';
    }
    
    hideSearchResults() {
        this.searchResults.style.display = 'none';
    }
    
    async performSearch() {
        const playerName = this.searchInput.value.trim();
        if (!playerName) {
            this.showError('Please enter a player name');
            return;
        }
        
        this.showLoading();
        this.hideResults();
        this.hideError();
        
        try {
            const response = await fetch(`/api/predict/${encodeURIComponent(playerName)}`);
            const data = await response.json();
            
            if (response.ok) {
                this.showResults(data);
            } else {
                this.showError(data.error || 'Player not found');
            }
        } catch (error) {
            this.showError('Error fetching predictions: ' + error.message);
        }
    }
    
    showLoading() {
        this.loadingSection.style.display = 'block';
    }
    
    hideLoading() {
        this.loadingSection.style.display = 'none';
    }
    
    showResults(data) {
        this.hideLoading();
        
        // Update player header
        document.getElementById('playerName').textContent = data.player;
        
        const positionsContainer = document.getElementById('playerPositions');
        positionsContainer.innerHTML = data.positions.map(pos => 
            `<span class="position-badge">${pos}</span>`
        ).join('');
        
        // Generate predictions
        const predictionsContainer = document.getElementById('predictionsContainer');
        predictionsContainer.innerHTML = '';
        
        for (const [position, prediction] of Object.entries(data.predictions)) {
            if (prediction.error) {
                predictionsContainer.innerHTML += this.createErrorCard(position, prediction.error);
            } else {
                predictionsContainer.innerHTML += this.createPredictionCard(position, prediction);
            }
        }
        
        this.resultsSection.style.display = 'block';
    }
    
    createPredictionCard(position, prediction) {
        const positionClass = position.toLowerCase().replace('/', '_');
        const adjustments = prediction.adjustments || {};
        
        return `
            <div class="prediction-card ${positionClass}">
                <div class="prediction-header">
                    <div class="prediction-title">${position} Prediction</div>
                    <div class="prediction-value">${prediction.enhanced_prediction.toFixed(1)} PPR</div>
                </div>
                
                <div class="prediction-breakdown">
                    <div class="breakdown-item">
                        <div class="breakdown-label">Base Prediction</div>
                        <div class="breakdown-value neutral">${prediction.base_prediction.toFixed(1)} PPR</div>
                    </div>
                    <div class="breakdown-item">
                        <div class="breakdown-label">Total Adjustment</div>
                        <div class="breakdown-value ${prediction.total_adjustment >= 0 ? 'positive' : 'negative'}">
                            ${prediction.total_adjustment >= 0 ? '+' : ''}${prediction.total_adjustment.toFixed(1)} PPR
                        </div>
                    </div>
                    <div class="breakdown-item">
                        <div class="breakdown-label">Enhanced Prediction</div>
                        <div class="breakdown-value positive">${prediction.enhanced_prediction.toFixed(1)} PPR</div>
                    </div>
                </div>
                
                ${this.createAdjustmentsSection(adjustments)}
                ${this.createRecentPerformanceSection(prediction.recent_performance)}
            </div>
        `;
    }
    
    createAdjustmentsSection(adjustments) {
        const adjustmentItems = Object.entries(adjustments).map(([key, value]) => {
            const label = this.formatAdjustmentLabel(key);
            const valueClass = value >= 0 ? 'positive' : 'negative';
            const sign = value >= 0 ? '+' : '';
            
            return `
                <div class="adjustment-item">
                    <div class="adjustment-label">${label}</div>
                    <div class="adjustment-value ${valueClass}">${sign}${value.toFixed(1)}</div>
                </div>
            `;
        }).join('');
        
        if (adjustmentItems) {
            return `
                <div class="recent-performance">
                    <h4>Enhancement Adjustments</h4>
                    <div class="adjustments-grid">
                        ${adjustmentItems}
                    </div>
                </div>
            `;
        }
        
        return '';
    }
    
    createRecentPerformanceSection(recentPerf) {
        if (!recentPerf) return '';
        
        return `
            <div class="recent-performance">
                <h4>Recent Performance (Last 5 Games)</h4>
                <div class="recent-stats">
                    <div class="recent-stat">
                        <div class="recent-stat-label">Average PPR</div>
                        <div class="recent-stat-value">${recentPerf.avg_ppr.toFixed(1)}</div>
                    </div>
                    <div class="recent-stat">
                        <div class="recent-stat-label">Total Games</div>
                        <div class="recent-stat-value">${recentPerf.total_games}</div>
                    </div>
                    <div class="recent-stat">
                        <div class="recent-stat-label">Recent PPR</div>
                        <div class="recent-stat-value">${recentPerf.ppr.slice(-3).join(', ')}</div>
                    </div>
                </div>
            </div>
        `;
    }
    
    createErrorCard(position, error) {
        return `
            <div class="prediction-card ${position.toLowerCase().replace('/', '_')}">
                <div class="prediction-header">
                    <div class="prediction-title">${position} Prediction</div>
                    <div class="prediction-value" style="color: #dc3545;">Error</div>
                </div>
                <p style="color: #dc3545; margin-top: 15px;">${error}</p>
            </div>
        `;
    }
    
    formatAdjustmentLabel(key) {
        const labels = {
            'minimax': 'Minimax Theory',
            'markov': 'Markov Chain',
            'performance_penalty': 'Performance Penalty',
            'passing_consistency': 'Passing Consistency',
            'rushing_upside': 'Rushing Upside',
            'receiving_upside': 'Receiving Upside'
        };
        
        return labels[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
    }
    
    showError(message) {
        this.hideLoading();
        document.getElementById('errorMessage').textContent = message;
        this.errorSection.style.display = 'block';
    }
    
    hideError() {
        this.errorSection.style.display = 'none';
    }
    
    hideResults() {
        this.resultsSection.style.display = 'none';
    }
    
    async loadInitialRankings() {
        await this.loadRankings('RB');
    }
    
    async switchTab(position) {
        // Update active tab
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-position="${position}"]`).classList.add('active');
        
        // Load rankings for this position
        await this.loadRankings(position);
    }
    
    async loadRankings(position, page = 1) {
        const container = document.getElementById('rankingsContainer');
        container.innerHTML = `
            <div class="rankings-loading">
                <i class="fas fa-spinner fa-spin"></i>
                <p>Loading ${position} rankings...</p>
            </div>
        `;
        
        try {
            const response = await fetch(`/api/rankings/${position}?page=${page}`);
            const data = await response.json();
            
            if (response.ok) {
                this.displayRankings(data.players, position, data.pagination);
            } else {
                container.innerHTML = `
                    <div class="rankings-loading">
                        <i class="fas fa-exclamation-triangle"></i>
                        <p>Error loading rankings: ${data.error}</p>
                    </div>
                `;
            }
        } catch (error) {
            container.innerHTML = `
                <div class="rankings-loading">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error loading rankings: ${error.message}</p>
                </div>
            `;
        }
    }
    
    displayRankings(rankings, position, pagination) {
        const container = document.getElementById('rankingsContainer');
        const positionClass = position.toLowerCase().replace('/', '_').replace('te', '_te');
        const startRank = (pagination.current_page - 1) * pagination.per_page + 1;
        
        const rankingsHTML = rankings.map((player, index) => `
            <div class="ranking-item-compact ${positionClass}" onclick="this.searchPlayer('${player.player}')">
                <div class="ranking-position">${startRank + index}</div>
                <div class="ranking-player">
                    <div class="player-name">${player.player}</div>
                    <div class="player-prediction">${player.enhanced_prediction.toFixed(1)} PPR</div>
                </div>
                <div class="ranking-adjustment ${player.total_adjustment >= 0 ? 'positive' : 'negative'}">
                    ${player.total_adjustment >= 0 ? '+' : ''}${player.total_adjustment.toFixed(1)}
                </div>
            </div>
        `).join('');
        
        // Add pagination controls
        const paginationHTML = this.createPaginationHTML(pagination, position);
        
        container.innerHTML = `
            <div class="rankings-list">
                ${rankingsHTML}
            </div>
            ${paginationHTML}
        `;
        
        // Add click handlers to ranking items
        container.querySelectorAll('.ranking-item-compact').forEach((item, index) => {
            item.addEventListener('click', () => {
                const playerName = rankings[index].player;
                this.searchInput.value = playerName;
                this.performSearch();
            });
        });
        
        // Add click handlers to pagination buttons
        container.querySelectorAll('.pagination-btn').forEach(button => {
            button.addEventListener('click', (e) => {
                const position = e.target.closest('.pagination-btn').dataset.position;
                const page = parseInt(e.target.closest('.pagination-btn').dataset.page);
                this.loadRankings(position, page);
            });
        });
    }
    
    createPaginationHTML(pagination, position) {
        let paginationHTML = '<div class="pagination">';
        
        // Previous button
        if (pagination.has_prev) {
            paginationHTML += `
                <button class="pagination-btn prev-btn" data-position="${position}" data-page="${pagination.current_page - 1}">
                    <i class="fas fa-chevron-left"></i> Previous
                </button>
            `;
        }
        
        // Page info
        paginationHTML += `
            <span class="pagination-info">
                Page ${pagination.current_page} of ${pagination.total_pages} 
                (${pagination.total_players} players)
            </span>
        `;
        
        // Next button
        if (pagination.has_next) {
            paginationHTML += `
                <button class="pagination-btn next-btn" data-position="${position}" data-page="${pagination.current_page + 1}">
                    Next <i class="fas fa-chevron-right"></i>
                </button>
            `;
        }
        
        paginationHTML += '</div>';
        return paginationHTML;
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new FantasyApp();
});
