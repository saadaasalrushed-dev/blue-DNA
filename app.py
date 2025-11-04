"""
Blue DNA - AI Beach Guardian
Main Flask Application
Built for student competition (Grades 5-8)
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import io
import base64
from model import load_model, predict, preprocess_image

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'blue-dna-ai-beach-guardian-2025')
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Enable CORS for cross-origin requests
CORS(app)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load AI model lazily (only when needed) to save memory on free tier
# This prevents worker timeout issues on Render free tier
print("Model will be loaded on first request to save memory...")
model = None
_model_loading = False

def get_model():
    """Get model instance, loading it if needed (lazy loading)"""
    global model, _model_loading
    if model is not None:
        return model
    
    if _model_loading:
        # Another request is already loading, wait a bit
        import time
        time.sleep(1)
        return model
    
    _model_loading = True
    try:
        print("Loading AI model (lazy load)...")
        model = load_model()
        if model is None:
            print("WARNING: Model is None - using dummy model")
            model = load_model()  # Will create dummy model
        print("AI model loaded successfully!")
        return model
    except Exception as e:
        print(f"ERROR loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        model = None
        return None
    finally:
        _model_loading = False


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def ensure_upload_folder():
    """Create upload folder if it doesn't exist"""
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])


# Routes
@app.route('/')
def index():
    """Home page with login/name input"""
    return render_template('index.html')


@app.route('/login', methods=['POST'])
def login():
    """Handle user login/name input"""
    try:
        name = request.form.get('name', '').strip()
        if name:
            session['user_name'] = name
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        else:
            return jsonify({'success': False, 'error': 'Please enter your name'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/dashboard')
def dashboard():
    """Main dashboard with navigation cards"""
    if 'user_name' not in session:
        return redirect(url_for('index'))
    return render_template('dashboard.html', user_name=session.get('user_name', 'Guardian'))


@app.route('/scanner')
def scanner():
    """AI Scanner page"""
    if 'user_name' not in session:
        return redirect(url_for('index'))
    return render_template('scanner.html', user_name=session.get('user_name', 'Guardian'))


@app.route('/map')
def map():
    """Smart Map page with UAE coastline"""
    if 'user_name' not in session:
        return redirect(url_for('index'))
    return render_template('map.html', user_name=session.get('user_name', 'Guardian'))


@app.route('/info')
def info():
    """Marine Info educational page"""
    if 'user_name' not in session:
        return redirect(url_for('index'))
    return render_template('info.html', user_name=session.get('user_name', 'Guardian'))


@app.route('/api/classify', methods=['POST'])
def classify_image():
    """
    AI Image Classification Endpoint
    Accepts: image file (multipart/form-data) or base64 image
    Returns: JSON with result, confidence, advice
    """
    try:
        # Ensure upload folder exists
        ensure_upload_folder()
        
        # Check if image is uploaded as file
        if 'image' in request.files:
            file = request.files['image']
            if file.filename == '':
                return jsonify({'error': 'No image file provided'}), 400
            
            if not allowed_file(file.filename):
                return jsonify({'error': 'Invalid file type. Only JPG and PNG allowed.'}), 400
            
            # Read image
            image_data = file.read()
            image = Image.open(io.BytesIO(image_data))
            
        # Check if image is sent as base64
        elif 'image_data' in request.json:
            # Decode base64 image
            image_data = base64.b64decode(request.json['image_data'].split(',')[1])
            image = Image.open(io.BytesIO(image_data))
            
        else:
            return jsonify({'error': 'No image provided'}), 400
        
        # Get model (lazy loading)
        current_model = get_model()
        if current_model is None:
            return jsonify({
                'success': False,
                'error': 'AI model not loaded. Please check server logs.',
                'details': 'Model file may be missing or corrupted. Using dummy model.'
            }), 500
        
        # Preprocess image for AI model
        processed_image = preprocess_image(image)
        
        # Get AI prediction
        result, confidence = predict(current_model, processed_image)
        
        # Generate advice based on result
        advice = generate_advice(result, confidence)
        
        # Return JSON response
        return jsonify({
            'success': True,
            'result': result,
            'confidence': float(confidence),
            'advice': advice,
            'warning': confidence < 0.80  # Low confidence warning
        })
        
    except Exception as e:
        print(f"Error in classify_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': f'Error processing image: {str(e)}',
            'details': str(e)
        }), 500


@app.route('/api/report', methods=['POST'])
def submit_report():
    """
    Submit Pollution Report Endpoint
    Stores pollution reports (optional feature)
    """
    try:
        data = request.json
        location = data.get('location', '')
        pollution_type = data.get('pollution_type', '')
        description = data.get('description', '')
        user_name = session.get('user_name', 'Anonymous')
        
        # In a full implementation, you would save to database here
        # For now, we'll just return success
        
        return jsonify({
            'success': True,
            'message': 'Report submitted successfully!',
            'report_id': f'RPT-{hash(location + pollution_type)}'
        })
        
    except Exception as e:
        return jsonify({'error': f'Error submitting report: {str(e)}'}), 500


@app.route('/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect(url_for('index'))


def generate_advice(result, confidence):
    """Generate advice message based on classification result"""
    if result == 'Clean':
        return "Great news! This area appears clean. Keep it that way by disposing of trash properly and encouraging others to do the same."
    elif result == 'Plastic':
        return "⚠️ Plastic pollution detected! Please report this to local authorities. Consider organizing a beach cleanup. Every piece of plastic removed helps protect marine life."
    elif result == 'Oil':
        return "🚨 Oil spill detected! This is serious. Report immediately to environmental authorities. Do not touch the oil - it can be harmful. Contact: UAE Environmental Agency."
    else:
        return "Please take another photo for better analysis."


# Error Handlers
@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Page not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error. Please try again.'}), 500


@app.errorhandler(413)
def too_large(error):
    """Handle file too large errors"""
    return jsonify({'error': 'File too large. Maximum size is 10MB.'}), 413


if __name__ == '__main__':
    # Create necessary folders
    ensure_upload_folder()
    
    # Run Flask app
    # For production, use: app.run(host='0.0.0.0', port=5000)
    # For development, use:
    app.run(debug=True, host='0.0.0.0', port=5000)

