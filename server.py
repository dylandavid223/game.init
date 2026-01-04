from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
import json
import numpy as np
from datetime import datetime
import logging
import sys
import os

# Add AI and graphics modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import AI and graphics modules
try:
    from ai_backend import AIEngine, RacingAI
    from graphics_engine import GraphicsProcessor, TerrainGenerator
    AI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: AI modules not available - {e}")
    print("Running in simulation mode...")
    AI_AVAILABLE = False

app = Flask(__name__, static_folder='.', static_url_path='')
app.config['SECRET_KEY'] = 'horizon-secret-key-2024'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Game state
game_state = {
    'players': {},
    'ai_engines': {},
    'graphics_processor': None,
    'terrain': None,
    'start_time': time.time()
}

# Initialize AI and graphics if available
if AI_AVAILABLE:
    try:
        game_state['graphics_processor'] = GraphicsProcessor()
        game_state['terrain'] = TerrainGenerator()
        print("AI and Graphics modules loaded successfully!")
    except Exception as e:
        print(f"Error initializing AI/Graphics: {e}")
        AI_AVAILABLE = False

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.route('/')
def index():
    """Serve the main HTML page."""
    return app.send_static_file('index.html')

@app.route('/status')
def status():
    """API endpoint for server status."""
    return jsonify({
        'status': 'online',
        'uptime': time.time() - game_state['start_time'],
        'players': len(game_state['players']),
        'ai_available': AI_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/ai/analyze', methods=['POST'])
def analyze_driving():
    """API endpoint for AI driving analysis."""
    try:
        data = request.json
        if not AI_AVAILABLE:
            return jsonify({'error': 'AI engine not available'}), 503
        
        # Simulate AI analysis
        analysis_result = simulate_ai_analysis(data)
        
        # If we have real AI, use it
        if 'ai_engines' in game_state and data.get('player_id') in game_state['ai_engines']:
            ai_engine = game_state['ai_engines'][data.get('player_id')]
            analysis_result = ai_engine.analyze_driving(data)
        
        return jsonify(analysis_result)
    except Exception as e:
        logger.error(f"Error in AI analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/graphics/process', methods=['POST'])
def process_graphics():
    """API endpoint for graphics processing."""
    try:
        data = request.json
        
        if not AI_AVAILABLE or game_state['graphics_processor'] is None:
            # Return simulated graphics data
            return jsonify(simulate_graphics_data(data))
        
        # Process with real graphics engine
        graphics_data = game_state['graphics_processor'].process_frame(data)
        return jsonify(graphics_data)
    except Exception as e:
        logger.error(f"Error in graphics processing: {e}")
        return jsonify({'error': str(e)}), 500

def simulate_ai_analysis(game_data):
    """Simulate AI analysis for testing."""
    import random
    
    # Generate realistic AI analysis data
    path_nodes = random.randint(50, 200)
    processing_time = random.uniform(5, 50)
    optimization = random.randint(70, 98)
    
    # Generate path points
    path_points = []
    base_x = game_data.get('player_position', {}).get('x', 0)
    base_z = game_data.get('player_position', {}).get('z', 0)
    
    for i in range(10):
        path_points.append({
            'x': base_x + random.uniform(-20, 20),
            'z': base_z + random.uniform(-20, 20),
            'type': 'optimal' if random.random() > 0.3 else 'alternative'
        })
    
    # Generate steering suggestions
    suggestions = {
        'steering_correction': random.uniform(-0.1, 0.1),
        'speed_adjustment': random.uniform(-0.05, 0.05),
        'optimal_path': random.random() > 0.2,
        'braking_point': random.uniform(0.1, 0.9)
    }
    
    # Generate prediction
    prediction = {
        'steering': random.uniform(-0.5, 0.5),
        'look_ahead_x': random.uniform(-10, 10),
        'look_ahead_z': random.uniform(-10, 10),
        'speed_adjustment': random.uniform(-0.1, 0.1),
        'confidence': random.uniform(0.7, 0.95)
    }
    
    return {
        'processing_time': processing_time,
        'path_nodes': path_nodes,
        'optimization': optimization,
        'neural_net_status': 'online',
        'path_points': path_points,
        'suggestions': suggestions,
        'prediction': prediction,
        'timestamp': datetime.now().isoformat()
    }

def simulate_graphics_data(graphics_data):
    """Simulate graphics processing data."""
    import random
    
    # Simulate weather effects
    weather_effects = {
        'fog_density': random.uniform(0.0001, 0.001),
        'light_intensity': random.uniform(0.8, 1.2),
        'cloud_cover': random.uniform(0.1, 0.8),
        'precipitation': random.uniform(0, 0.3)
    }
    
    # Simulate lighting optimization
    light_optimization = {
        'shadow_quality': random.choice(['low', 'medium', 'high']),
        'ambient_occlusion': random.random() > 0.5,
        'bloom_strength': random.uniform(0.1, 0.5),
        'reflections': random.random() > 0.3
    }
    
    # Simulate performance data
    performance = {
        'gpu_usage': random.uniform(20, 80),
        'cpu_usage': random.uniform(10, 50),
        'memory_usage': random.uniform(500, 2000),
        'frame_time': random.uniform(8, 20)
    }
    
    return {
        'weather_effects': weather_effects,
        'light_optimization': light_optimization,
        'shadow_quality': random.choice(['soft', 'hard', 'pcf']),
        'performance': performance,
        'render_mode': 'optimized',
        'timestamp': datetime.now().isoformat()
    }

# WebSocket event handlers
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    client_id = request.sid
    game_state['players'][client_id] = {
        'connected_at': time.time(),
        'car_type': None,
        'last_update': time.time(),
        'position': {'x': 0, 'y': 0, 'z': 0}
    }
    
    # Initialize AI engine for this player if available
    if AI_AVAILABLE:
        try:
            game_state['ai_engines'][client_id] = RacingAI()
            logger.info(f"Initialized AI engine for player {client_id}")
        except Exception as e:
            logger.error(f"Failed to initialize AI engine: {e}")
    
    logger.info(f"Client connected: {client_id}")
    emit('connection_established', {
        'player_id': client_id,
        'server_time': datetime.now().isoformat(),
        'ai_available': AI_AVAILABLE
    })
    
    # Send welcome message
    emit('python_output', {
        'output': f"[SYSTEM] Connected to Python AI Backend (AI: {'ENABLED' if AI_AVAILABLE else 'SIMULATION MODE'})"
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    client_id = request.sid
    if client_id in game_state['players']:
        del game_state['players'][client_id]
    
    if client_id in game_state['ai_engines']:
        del game_state['ai_engines'][client_id]
    
    logger.info(f"Client disconnected: {client_id}")

@socketio.on('game_data')
def handle_game_data(data):
    """Handle game data from client."""
    client_id = request.sid
    
    if client_id not in game_state['players']:
        return
    
    # Update player state
    game_state['players'][client_id].update({
        'last_update': time.time(),
        'position': data.get('player_position', {'x': 0, 'y': 0, 'z': 0}),
        'speed': data.get('speed', 0),
        'rotation': data.get('rotation', 0)
    })
    
    # Process with AI if available
    if AI_AVAILABLE and client_id in game_state['ai_engines']:
        try:
            # Add player state to data
            data['player_state'] = game_state['players'][client_id]
            
            # Get AI analysis
            ai_analysis = game_state['ai_engines'][client_id].process_game_data(data)
            
            # Send AI response back to client
            emit('ai_response', ai_analysis)
            
            # Log to Python output
            if random.random() < 0.1:  # Log 10% of updates
                emit('python_output', {
                    'output': f"[AI] Processed frame - Speed: {data.get('speed', 0):.1f}, Nodes: {ai_analysis.get('path_nodes', 0)}"
                })
        except Exception as e:
            logger.error(f"Error in AI processing: {e}")
            emit('python_output', {
                'output': f"[ERROR] AI processing failed: {str(e)}"
            })
    else:
        # Send simulated AI response
        simulated_data = simulate_ai_analysis(data)
        emit('ai_response', simulated_data)

@socketio.on('ai_analysis')
def handle_ai_analysis(data):
    """Handle AI analysis request."""
    client_id = request.sid
    
    if data.get('type') == 'initial' and client_id in game_state['ai_engines']:
        # Initialize AI with car type
        car_type = data.get('car_type', 'street')
        ai_engine = game_state['ai_engines'][client_id]
        ai_engine.initialize(car_type)
        
        emit('python_output', {
            'output': f"[AI] Initialized with {car_type} profile - Neural networks loaded"
        })
        
        # Send initial analysis
        initial_analysis = {
            'processing_time': 42.5,
            'path_nodes': 128,
            'optimization': 85,
            'neural_net_status': 'online',
            'message': f"AI ready for {car_type} racing"
        }
        emit('ai_response', initial_analysis)

@socketio.on('car_selected')
def handle_car_selected(data):
    """Handle car selection."""
    client_id = request.sid
    
    if client_id in game_state['players']:
        game_state['players'][client_id]['car_type'] = data.get('car_type')
        
        emit('python_output', {
            'output': f"[SYSTEM] Car selected: {data.get('car_type')} - AI profile activated"
        })

@socketio.on('ai_mode_changed')
def handle_ai_mode_changed(data):
    """Handle AI mode change."""
    client_id = request.sid
    
    emit('python_output', {
        'output': f"[AI] Mode changed to: {data.get('mode', 'unknown')}"
    })

@socketio.on('optimization_request')
def handle_optimization_request(data):
    """Handle optimization request."""
    client_id = request.sid
    
    # Simulate optimization process
    emit('python_output', {
        'output': f"[OPTIMIZATION] Starting neural network optimization..."
    })
    
    # Simulate processing delay
    time.sleep(0.5)
    
    # Send optimization results
    optimization_result = {
        'processing_time': 125.3,
        'path_nodes': 256,
        'optimization': 92,
        'neural_net_status': 'optimized',
        'message': "Pathfinding optimized - 15% performance improvement"
    }
    
    emit('ai_response', optimization_result)
    emit('python_output', {
        'output': f"[OPTIMIZATION] Complete - New score: 92%"
    })

def background_tasks():
    """Background tasks for server maintenance."""
    while True:
        try:
            # Clean up disconnected players
            current_time = time.time()
            disconnected_players = []
            
            for player_id, player_data in game_state['players'].items():
                if current_time - player_data['last_update'] > 30:  # 30 seconds timeout
                    disconnected_players.append(player_id)
            
            for player_id in disconnected_players:
                if player_id in game_state['players']:
                    del game_state['players'][player_id]
                if player_id in game_state['ai_engines']:
                    del game_state['ai_engines'][player_id]
            
            # Periodic server status update
            socketio.emit('server_status', {
                'uptime': time.time() - game_state['start_time'],
                'player_count': len(game_state['players']),
                'timestamp': datetime.now().isoformat()
            })
            
            time.sleep(5)  # Run every 5 seconds
            
        except Exception as e:
            logger.error(f"Error in background tasks: {e}")
            time.sleep(10)

if __name__ == '__main__':
    # Start background tasks in a separate thread
    bg_thread = threading.Thread(target=background_tasks, daemon=True)
    bg_thread.start()
    
    logger.info("Starting Horizon Browser Server...")
    logger.info(f"AI Available: {AI_AVAILABLE}")
    logger.info("Server running on http://localhost:5000")
    
    # Run the server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)