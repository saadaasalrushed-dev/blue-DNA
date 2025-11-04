/**
 * Blue DNA - AI Beach Guardian
 * Frontend JavaScript
 */

// Global app state
const AppState = {
    scanCount: 0,
    userName: null
};

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

/**
 * Initialize application
 */
function initializeApp() {
    // Load scan count from localStorage
    const savedCount = localStorage.getItem('scanCount');
    if (savedCount) {
        AppState.scanCount = parseInt(savedCount);
    }
    
    // Update dashboard scan count if on dashboard page
    if (document.getElementById('scanCount')) {
        document.getElementById('scanCount').textContent = `Scans: ${AppState.scanCount}`;
    }
    
    // Initialize page-specific features
    if (document.body.classList.contains('scanner-page')) {
        initScanner();
    }
    
    if (document.body.classList.contains('dashboard-page')) {
        initDashboard();
    }
}

/**
 * Initialize scanner page
 */
function initScanner() {
    console.log('Scanner page initialized');
    // Scanner-specific initialization is handled in scanner.html inline script
}

/**
 * Initialize dashboard page
 */
function initDashboard() {
    console.log('Dashboard page initialized');
    
    // Add click animations to cards
    const cards = document.querySelectorAll('.nav-card');
    cards.forEach(card => {
        card.addEventListener('click', function() {
            // Add ripple effect
            const ripple = document.createElement('div');
            ripple.style.cssText = `
                position: absolute;
                border-radius: 50%;
                background: rgba(255, 255, 255, 0.6);
                transform: scale(0);
                animation: ripple 0.6s linear;
                pointer-events: none;
            `;
            
            const rect = this.getBoundingClientRect();
            const size = Math.max(rect.width, rect.height);
            ripple.style.width = ripple.style.height = size + 'px';
            ripple.style.left = (event.clientX - rect.left - size / 2) + 'px';
            ripple.style.top = (event.clientY - rect.top - size / 2) + 'px';
            
            this.style.position = 'relative';
            this.appendChild(ripple);
            
            setTimeout(() => ripple.remove(), 600);
        });
    });
}

/**
 * Update scan count
 */
function updateScanCount() {
    AppState.scanCount++;
    localStorage.setItem('scanCount', AppState.scanCount);
    
    if (document.getElementById('scanCount')) {
        document.getElementById('scanCount').textContent = `Scans: ${AppState.scanCount}`;
    }
}

/**
 * Show notification
 */
function showNotification(message, type = 'info') {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.textContent = message;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? '#4caf50' : type === 'error' ? '#f44336' : '#0066CC'};
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
        z-index: 1000;
        animation: slideIn 0.3s ease-out;
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease-out';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * Format confidence percentage
 */
function formatConfidence(confidence) {
    return (confidence * 100).toFixed(1) + '%';
}

/**
 * Get color for result type
 */
function getResultColor(result) {
    const colors = {
        'Clean': '#4caf50',
        'Plastic': '#ff9800',
        'Oil': '#f44336'
    };
    return colors[result] || '#666';
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

