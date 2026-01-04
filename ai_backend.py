import numpy as np
import random
import time
from datetime import datetime
import math
from typing import Dict, List, Tuple, Optional
import json

class NeuralNetwork:
    """Simple neural network for racing AI."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights with small random values
        self.weights1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.weights2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        
        # Biases
        self.bias1 = np.zeros((1, self.hidden_size))
        self.bias2 = np.zeros((1, self.output_size))
        
        self.learning_rate = 0.01
        self.trained_samples = 0
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, inputs):
        """Forward pass through the network."""
        # Ensure inputs are in the right shape
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        
        # Hidden layer
        self.hidden_input = np.dot(inputs, self.weights1) + self.bias1
        self.hidden_output = self.sigmoid(self.hidden_input)
        
        # Output layer
        self.final_input = np.dot(self.hidden_output, self.weights2) + self.bias2
        self.final_output = self.sigmoid(self.final_input)
        
        return self.final_output
    
    def backward(self, inputs, targets):
        """Backward pass for training."""
        if len(inputs.shape) == 1:
            inputs = inputs.reshape(1, -1)
        if len(targets.shape) == 1:
            targets = targets.reshape(1, -1)
        
        # Calculate output error
        output_error = targets - self.final_output
        output_delta = output_error * self.sigmoid_derivative(self.final_output)
        
        # Calculate hidden layer error
        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
        
        # Update weights and biases
        self.weights2 += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.weights1 += inputs.T.dot(hidden_delta) * self.learning_rate
        
        self.bias2 += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate
        self.bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate
        
        self.trained_samples += 1
        
        # Return mean squared error
        mse = np.mean(output_error ** 2)
        return mse
    
    def save_weights(self, filename):
        """Save network weights to file."""
        weights_data = {
            'weights1': self.weights1.tolist(),
            'weights2': self.weights2.tolist(),
            'bias1': self.bias1.tolist(),
            'bias2': self.bias2.tolist(),
            'trained_samples': self.trained_samples
        }
        
        with open(filename, 'w') as f:
            json.dump(weights_data, f)
    
    def load_weights(self, filename):
        """Load network weights from file."""
        try:
            with open(filename, 'r') as f:
                weights_data = json.load(f)
            
            self.weights1 = np.array(weights_data['weights1'])
            self.weights2 = np.array(weights_data['weights2'])
            self.bias1 = np.array(weights_data['bias1'])
            self.bias2 = np.array(weights_data['bias2'])
            self.trained_samples = weights_data.get('trained_samples', 0)
            
            return True
        except Exception as e:
            print(f"Error loading weights: {e}")
            return False

class Pathfinder:
    """Pathfinding algorithm for racing lines."""
    
    def __init__(self):
        self.waypoints = []
        self.optimal_path = []
        self.path_cache = {}
        
    def generate_waypoints(self, start_pos, end_pos, num_points=50):
        """Generate waypoints between start and end positions."""
        waypoints = []
        
        # Generate bezier curve for smooth racing line
        for i in range(num_points):
            t = i / (num_points - 1)
            
            # Cubic bezier with control points
            p0 = np.array([start_pos['x'], start_pos['z']])
            p1 = np.array([start_pos['x'] + 20, start_pos['z'] + 10])
            p2 = np.array([end_pos['x'] - 20, end_pos['z'] - 10])
            p3 = np.array([end_pos['x'], end_pos['z']])
            
            point = (1 - t)**3 * p0 + 3 * (1 - t)**2 * t * p1 + 3 * (1 - t) * t**2 * p2 + t**3 * p3
            
            waypoints.append({
                'x': float(point[0]),
                'z': float(point[1]),
                'type': 'optimal' if i % 5 == 0 else 'intermediate'
            })
        
        self.waypoints = waypoints
        return waypoints
    
    def find_optimal_path(self, current_pos, lookahead_distance=50):
        """Find optimal path from current position."""
        if not self.waypoints:
            return []
        
        # Find nearest waypoint
        distances = []
        for i, wp in enumerate(self.waypoints):
            dist = math.sqrt((wp['x'] - current_pos['x'])**2 + (wp['z'] - current_pos['z'])**2)
            distances.append((dist, i, wp))
        
        # Sort by distance
        distances.sort(key=lambda x: x[0])
        
        # Take waypoints ahead
        optimal_path = []
        for dist, idx, wp in distances[:10]:
            optimal_path.append(wp)
        
        self.optimal_path = optimal_path
        return optimal_path
    
    def calculate_steering_angle(self, current_pos, current_heading, next_waypoint):
        """Calculate optimal steering angle to reach next waypoint."""
        if not next_waypoint:
            return 0
        
        # Vector to waypoint
        dx = next_waypoint['x'] - current_pos['x']
        dz = next_waypoint['z'] - current_pos['z']
        
        # Desired heading
        desired_heading = math.atan2(dx, dz)
        
        # Current heading (assuming 0 is forward along z-axis)
        current_heading_rad = math.radians(current_heading)
        
        # Angle difference
        angle_diff = desired_heading - current_heading_rad
        
        # Normalize to [-pi, pi]
        while angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        while angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        
        # Scale to reasonable steering value
        steering = angle_diff / (math.pi / 4)  # Max 45 degree correction
        steering = max(-1, min(1, steering))
        
        return steering
    
    def optimize_path(self, current_speed, track_conditions):
        """Optimize path based on speed and conditions."""
        optimized_path = []
        
        for wp in self.optimal_path:
            # Adjust waypoint based on speed (faster = wider turns)
            speed_factor = min(1.0, current_speed / 100)
            
            optimized_wp = wp.copy()
            if wp['type'] == 'optimal':
                # Widen turns at high speed
                optimized_wp['x'] = wp['x'] * (1 + speed_factor * 0.1)
            
            optimized_path.append(optimized_wp)
        
        return optimized_path

class RacingAI:
    """Main racing AI controller."""
    
    def __init__(self):
        self.neural_net = NeuralNetwork()
        self.pathfinder = Pathfinder()
        self.car_type = None
        self.ai_mode = 'assisted'
        self.performance_stats = {
            'total_decisions': 0,
            'avg_processing_time': 0,
            'success_rate': 0.95,
            'last_update': time.time()
        }
        
        # Load pre-trained weights if available
        self.load_trained_model()
        
        # Initialize pathfinder with default waypoints
        self.pathfinder.generate_waypoints(
            {'x': -100, 'z': -100},
            {'x': 100, 'z': 100},
            100
        )
    
    def load_trained_model(self):
        """Load pre-trained neural network model."""
        try:
            # Try to load from file
            if self.neural_net.load_weights('ai_weights.json'):
                print("Loaded pre-trained AI model")
            else:
                print("Training new AI model...")
                self.train_initial_model()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.train_initial_model()
    
    def train_initial_model(self):
        """Train initial neural network with basic racing patterns."""
        print("Training initial racing AI model...")
        
        training_data = [
            # Straight line at high speed
            ([1.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]),
            # Gentle right turn
            ([0.8, 0.2, 0.0, 0.0, 80.0, 0.3, 0.0, 0.0, 0.0, 0.0], [0.1, 0.0, 0.9, 0.0]),
            # Sharp left turn
            ([0.6, -0.4, 0.0, 0.0, 60.0, -0.8, 0.0, 0.0, 0.0, 0.0], [-0.3, 0.0, 0.7, 0.0]),
            # Braking for corner
            ([0.0, 0.0, 0.8, 0.0, 120.0, 0.5, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]),
        ]
        
        for epoch in range(1000):
            total_error = 0
            for inputs, targets in training_data:
                # Forward pass
                self.neural_net.forward(np.array(inputs))
                # Backward pass
                error = self.neural_net.backward(np.array(inputs), np.array(targets))
                total_error += error
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Error: {total_error / len(training_data):.4f}")
        
        # Save trained weights
        self.neural_net.save_weights('ai_weights.json')
        print("Initial AI training complete!")
    
    def initialize(self, car_type):
        """Initialize AI for specific car type."""
        self.car_type = car_type
        
        # Adjust AI behavior based on car type
        if car_type == 'street':
            self.aggression = 0.7
            self.cornering = 0.8
            self.top_speed_focus = 0.6
        elif car_type == 'offroad':
            self.aggression = 0.4
            self.cornering = 0.6
            self.top_speed_focus = 0.3
        elif car_type == 'hyper':
            self.aggression = 0.9
            self.cornering = 0.9
            self.top_speed_focus = 0.8
        
        return {
            'status': 'initialized',
            'car_type': car_type,
            'ai_profile': {
                'aggression': self.aggression,
                'cornering': self.cornering,
                'top_speed_focus': self.top_speed_focus
            }
        }
    
    def process_game_data(self, game_data):
        """Process incoming game data and generate AI response."""
        start_time = time.time()
        
        # Extract data
        player_pos = game_data.get('player_position', {'x': 0, 'y': 0, 'z': 0})
        speed = game_data.get('speed', 0)
        rotation = game_data.get('rotation', 0)
        
        # Update pathfinder
        optimal_path = self.pathfinder.find_optimal_path(player_pos)
        
        # Calculate next waypoint
        next_waypoint = optimal_path[0] if optimal_path else None
        
        # Neural network decision
        nn_inputs = self.prepare_nn_inputs(game_data, next_waypoint)
        nn_output = self.neural_net.forward(nn_inputs)
        
        # Interpret neural network output
        steering_suggestion = float(nn_output[0][0])  # -1 to 1
        throttle_suggestion = float(nn_output[0][1])  # 0 to 1
        brake_suggestion = float(nn_output[0][2])     # 0 to 1
        confidence = float(nn_output[0][3])           # 0 to 1
        
        # Adjust based on car type and AI mode
        if self.ai_mode == 'assisted':
            # Blend player input with AI suggestions
            player_steering = game_data.get('rotation', 0) / math.pi  # Normalize
            steering_suggestion = steering_suggestion * 0.3 + player_steering * 0.7
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Update performance stats
        self.performance_stats['total_decisions'] += 1
        self.performance_stats['avg_processing_time'] = (
            self.performance_stats['avg_processing_time'] * 0.9 + 
            processing_time * 0.1
        )
        
        # Prepare response
        response = {
            'processing_time': processing_time,
            'path_nodes': len(optimal_path),
            'optimization': int(confidence * 100),
            'neural_net_status': 'online',
            'path_points': optimal_path[:10],  # Send first 10 waypoints
            'suggestions': {
                'steering_correction': steering_suggestion,
                'speed_adjustment': throttle_suggestion - brake_suggestion,
                'optimal_path': confidence > 0.7,
                'braking_point': brake_suggestion
            },
            'prediction': {
                'steering': steering_suggestion,
                'look_ahead_x': next_waypoint['x'] if next_waypoint else 0,
                'look_ahead_z': next_waypoint['z'] if next_waypoint else 0,
                'speed_adjustment': throttle_suggestion - brake_suggestion,
                'confidence': confidence
            },
            'performance': {
                'total_decisions': self.performance_stats['total_decisions'],
                'avg_processing_time': self.performance_stats['avg_processing_time'],
                'success_rate': self.performance_stats['success_rate']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def prepare_nn_inputs(self, game_data, next_waypoint):
        """Prepare inputs for neural network."""
        player_pos = game_data.get('player_position', {'x': 0, 'y': 0, 'z': 0})
        speed = game_data.get('speed', 0)
        rotation = game_data.get('rotation', 0)
        
        # Normalize inputs
        inputs = [
            # Player position (normalized)
            player_pos['x'] / 100,      # -1 to 1 range for ±100 units
            player_pos['z'] / 100,
            
            # Speed and direction
            speed / 200,                # 0 to 1 for 0-200 km/h
            math.sin(rotation) * 0.5 + 0.5,  # 0 to 1
            
            # Next waypoint relative position
            (next_waypoint['x'] - player_pos['x']) / 50 if next_waypoint else 0,
            (next_waypoint['z'] - player_pos['z']) / 50 if next_waypoint else 0,
            
            # Car type factors
            self.aggression,
            self.cornering,
            self.top_speed_focus,
            
            # AI mode (0=manual, 0.5=assisted, 1=full)
            0.5 if self.ai_mode == 'assisted' else 1.0
        ]
        
        return np.array(inputs)
    
    def analyze_driving(self, driving_data):
        """Analyze driving patterns for feedback and improvement."""
        # This would analyze driving patterns and provide feedback
        analysis = {
            'smoothness_score': random.uniform(70, 95),
            'cornering_efficiency': random.uniform(65, 90),
            'speed_consistency': random.uniform(75, 92),
            'suggestions': [
                'Try taking turns wider for better exit speed',
                'Brake 10m earlier for the hairpin turn',
                'Use more of the track width'
            ]
        }
        
        return analysis

class AIEngine:
    """Main AI engine orchestrator."""
    
    def __init__(self):
        self.racing_ai = RacingAI()
        self.active_players = {}
        self.frame_counter = 0
        
    def process_frame(self, frame_data):
        """Process a frame of game data."""
        self.frame_counter += 1
        
        # Update all active players
        for player_id, player_data in self.active_players.items():
            if player_id in frame_data:
                ai_response = self.racing_ai.process_game_data(frame_data[player_id])
                self.active_players[player_id]['last_response'] = ai_response
        
        # Return aggregated AI data
        return {
            'frame_number': self.frame_counter,
            'active_players': len(self.active_players),
            'ai_status': 'active',
            'timestamp': datetime.now().isoformat()
        }
    
    def add_player(self, player_id, car_type='street'):
        """Add a new player to AI tracking."""
        self.active_players[player_id] = {
            'car_type': car_type,
            'joined_at': time.time(),
            'last_update': time.time(),
            'ai_instance': RacingAI()
        }
        
        # Initialize AI for this player
        self.active_players[player_id]['ai_instance'].initialize(car_type)
    
    def remove_player(self, player_id):
        """Remove a player from AI tracking."""
        if player_id in self.active_players:
            del self.active_players[player_id]
    
    def get_ai_stats(self):
        """Get AI engine statistics."""
        return {
            'total_players': len(self.active_players),
            'total_frames': self.frame_counter,
            'avg_processing_time': 25.5,  # Simulated
            'memory_usage': 128,  # MB
            'neural_net_status': 'operational'
        }

# Utility functions
def simulate_ai_processing(game_state):
    """Simulate AI processing for testing."""
    time.sleep(0.01)  # Simulate 10ms processing time
    
    return {
        'steering_adjustment': random.uniform(-0.1, 0.1),
        'throttle_adjustment': random.uniform(-0.05, 0.05),
        'brake_suggestion': random.uniform(0, 0.2),
        'optimal_path': random.choice([True, False]),
        'processing_time': random.uniform(5, 25)
    }

if __name__ == "__main__":
    # Test the AI engine
    print("Testing Racing AI Engine...")
    
    ai = RacingAI()
    ai.initialize('street')
    
    test_data = {
        'player_position': {'x': 0, 'y': 0, 'z': 0},
        'speed': 80,
        'rotation': 0.5,
        'keys': ['w', 'd']
    }
    
    response = ai.process_game_data(test_data)
    print("AI Response:")
    print(json.dumps(response, indent=2))