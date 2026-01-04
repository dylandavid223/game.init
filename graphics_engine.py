import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import json

class Vector3:
    """Simple 3D vector class."""
    
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        length = self.length()
        if length > 0:
            return Vector3(self.x/length, self.y/length, self.z/length)
        return Vector3(0, 0, 0)
    
    def to_dict(self):
        return {'x': self.x, 'y': self.y, 'z': self.z}
    
    @classmethod
    def from_dict(cls, data):
        return cls(data.get('x', 0), data.get('y', 0), data.get('z', 0))

class TerrainGenerator:
    """Generates and manages terrain data."""
    
    def __init__(self, size=1000, resolution=512):
        self.size = size
        self.resolution = resolution
        self.height_map = None
        self.normal_map = None
        self.texture_map = None
        
        self.generate_terrain()
        self.calculate_normals()
        self.generate_texture_map()
    
    def generate_terrain(self):
        """Generate terrain using Perlin-like noise."""
        print("Generating terrain...")
        
        # Initialize height map
        self.height_map = np.zeros((self.resolution, self.resolution))
        
        # Generate multiple octaves of noise
        for octave in range(4):
            frequency = 2 ** octave
            amplitude = 1.0 / (frequency * 2)
            
            for i in range(self.resolution):
                for j in range(self.resolution):
                    # Simplex-like noise
                    nx = i / self.resolution * frequency
                    ny = j / self.resolution * frequency
                    
                    # Simple pseudo-random noise
                    noise = (math.sin(nx * 12.9898 + ny * 78.233) * 43758.5453) % 1
                    self.height_map[i][j] += noise * amplitude
        
        # Add mountains and valleys
        self.add_features()
        
        # Normalize height map
        min_height = np.min(self.height_map)
        max_height = np.max(self.height_map)
        if max_height > min_height:
            self.height_map = (self.height_map - min_height) / (max_height - min_height)
        
        # Scale to desired height range
        self.height_map = self.height_map * 100  # 0-100 units height
    
    def add_features(self):
        """Add terrain features like mountains and valleys."""
        center_x, center_y = self.resolution // 2, self.resolution // 2
        
        # Add central mountain
        for i in range(self.resolution):
            for j in range(self.resolution):
                dx = (i - center_x) / self.resolution
                dy = (j - center_y) / self.resolution
                distance = math.sqrt(dx*dx + dy*dy)
                
                # Mountain in center
                if distance < 0.2:
                    height = 1.0 - (distance / 0.2) * 0.5
                    self.height_map[i][j] += height * 0.5
                
                # Valley around mountain
                elif distance < 0.3:
                    height = (distance - 0.2) / 0.1 * 0.3
                    self.height_map[i][j] += height
    
    def calculate_normals(self):
        """Calculate surface normals for lighting."""
        print("Calculating terrain normals...")
        
        self.normal_map = np.zeros((self.resolution, self.resolution, 3))
        
        for i in range(1, self.resolution - 1):
            for j in range(1, self.resolution - 1):
                # Get heights of neighboring points
                h_left = self.height_map[i-1][j]
                h_right = self.height_map[i+1][j]
                h_up = self.height_map[i][j-1]
                h_down = self.height_map[i][j+1]
                
                # Calculate gradient
                dx = (h_right - h_left) / 2
                dy = (h_down - h_up) / 2
                
                # Normal vector (pointing up)
                normal = Vector3(-dx, 1, -dy)
                normal = normal.normalize()
                
                self.normal_map[i][j] = [normal.x, normal.y, normal.z]
        
        # Fill borders with nearest normals
        self.normal_map[0] = self.normal_map[1]
        self.normal_map[-1] = self.normal_map[-2]
        self.normal_map[:, 0] = self.normal_map[:, 1]
        self.normal_map[:, -1] = self.normal_map[:, -2]
    
    def generate_texture_map(self):
        """Generate texture coordinates based on terrain features."""
        print("Generating texture map...")
        
        self.texture_map = np.zeros((self.resolution, self.resolution, 2))
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                # Simple UV coordinates
                self.texture_map[i][j] = [i / self.resolution, j / self.resolution]
    
    def get_height(self, x, z):
        """Get terrain height at world coordinates."""
        # Convert world coordinates to height map indices
        i = int((x / self.size + 0.5) * self.resolution)
        j = int((z / self.size + 0.5) * self.resolution)
        
        # Clamp indices
        i = max(0, min(self.resolution - 1, i))
        j = max(0, min(self.resolution - 1, j))
        
        return float(self.height_map[i][j])
    
    def get_normal(self, x, z):
        """Get surface normal at world coordinates."""
        i = int((x / self.size + 0.5) * self.resolution)
        j = int((z / self.size + 0.5) * self.resolution)
        
        i = max(0, min(self.resolution - 1, i))
        j = max(0, min(self.resolution - 1, j))
        
        normal_data = self.normal_map[i][j]
        return Vector3(normal_data[0], normal_data[1], normal_data[2])
    
    def get_terrain_data(self, bounds=None):
        """Get terrain data for rendering."""
        if bounds is None:
            bounds = {
                'min_x': -self.size/2,
                'max_x': self.size/2,
                'min_z': -self.size/2,
                'max_z': self.size/2
            }
        
        # Calculate sampling resolution
        samples_x = 100
        samples_z = 100
        
        terrain_data = {
            'vertices': [],
            'normals': [],
            'textures': [],
            'indices': [],
            'bounds': bounds
        }
        
        # Generate grid of vertices
        for i in range(samples_x):
            for j in range(samples_z):
                x = bounds['min_x'] + (bounds['max_x'] - bounds['min_x']) * i / (samples_x - 1)
                z = bounds['min_z'] + (bounds['max_z'] - bounds['min_z']) * j / (samples_z - 1)
                y = self.get_height(x, z)
                
                terrain_data['vertices'].append([x, y, z])
                
                normal = self.get_normal(x, z)
                terrain_data['normals'].append([normal.x, normal.y, normal.z])
                
                # Texture coordinates
                u = (x - bounds['min_x']) / (bounds['max_x'] - bounds['min_x'])
                v = (z - bounds['min_z']) / (bounds['max_z'] - bounds['min_z'])
                terrain_data['textures'].append([u, v])
        
        # Generate triangle indices
        for i in range(samples_x - 1):
            for j in range(samples_z - 1):
                # Two triangles per quad
                a = i * samples_z + j
                b = (i + 1) * samples_z + j
                c = i * samples_z + (j + 1)
                d = (i + 1) * samples_z + (j + 1)
                
                terrain_data['indices'].extend([a, b, c, b, d, c])
        
        return terrain_data

class GraphicsProcessor:
    """Processes graphics and visual effects."""
    
    def __init__(self):
        self.terrain_generator = TerrainGenerator()
        self.light_sources = []
        self.particle_systems = []
        self.post_effects = []
        self.frame_counter = 0
        
        # Initialize lighting
        self.setup_lighting()
        
        # Initialize post-processing effects
        self.setup_post_effects()
    
    def setup_lighting(self):
        """Setup lighting system."""
        # Main directional light (sun)
        sun_light = {
            'type': 'directional',
            'direction': Vector3(-1, -1, -1).normalize(),
            'color': [1.0, 1.0, 0.9],
            'intensity': 1.0,
            'casts_shadows': True
        }
        
        # Ambient light
        ambient_light = {
            'type': 'ambient',
            'color': [0.3, 0.4, 0.5],
            'intensity': 0.3
        }
        
        self.light_sources = [sun_light, ambient_light]
    
    def setup_post_effects(self):
        """Setup post-processing effects."""
        self.post_effects = [
            {
                'name': 'bloom',
                'enabled': True,
                'strength': 0.5,
                'radius': 0.8,
                'threshold': 0.7
            },
            {
                'name': 'motion_blur',
                'enabled': True,
                'strength': 0.3,
                'samples': 8
            },
            {
                'name': 'color_grading',
                'enabled': True,
                'contrast': 1.1,
                'saturation': 1.2,
                'brightness': 1.0
            },
            {
                'name': 'vignette',
                'enabled': True,
                'strength': 0.3,
                'roundness': 0.8
            }
        ]
    
    def process_frame(self, frame_data):
        """Process a frame of graphics data."""
        start_time = datetime.now()
        self.frame_counter += 1
        
        # Extract data
        player_pos = Vector3.from_dict(frame_data.get('player_position', {}))
        camera_pos = Vector3.from_dict(frame_data.get('camera_position', {}))
        time_of_day = frame_data.get('time_of_day', 12.0)
        weather = frame_data.get('weather', 'clear')
        
        # Update lighting based on time of day
        self.update_lighting(time_of_day, weather)
        
        # Update post-processing effects
        self.update_post_effects(frame_data)
        
        # Generate terrain data for current view
        view_bounds = self.calculate_view_bounds(camera_pos)
        terrain_data = self.terrain_generator.get_terrain_data(view_bounds)
        
        # Generate particle effects
        particle_data = self.generate_particles(player_pos, weather)
        
        # Calculate performance metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Prepare response
        response = {
            'frame_number': self.frame_counter,
            'processing_time': processing_time,
            'terrain_data': terrain_data,
            'lighting': self.get_lighting_data(),
            'post_effects': self.post_effects,
            'particles': particle_data,
            'weather_effects': self.get_weather_effects(weather),
            'optimization': {
                'lod_level': self.calculate_lod(player_pos),
                'culling_applied': True,
                'texture_quality': 'high',
                'shadow_quality': 'soft',
                'draw_calls': len(terrain_data.get('indices', [])) // 3
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return response
    
    def calculate_view_bounds(self, camera_pos):
        """Calculate visible terrain bounds from camera position."""
        view_distance = 500
        bounds = {
            'min_x': camera_pos.x - view_distance,
            'max_x': camera_pos.x + view_distance,
            'min_z': camera_pos.z - view_distance,
            'max_z': camera_pos.z + view_distance
        }
        
        # Clamp to terrain bounds
        terrain_size = self.terrain_generator.size / 2
        bounds['min_x'] = max(-terrain_size, bounds['min_x'])
        bounds['max_x'] = min(terrain_size, bounds['max_x'])
        bounds['min_z'] = max(-terrain_size, bounds['min_z'])
        bounds['max_z'] = min(terrain_size, bounds['max_z'])
        
        return bounds
    
    def update_lighting(self, time_of_day, weather):
        """Update lighting based on time of day and weather."""
        # Convert time of day to sun angle
        sun_angle = (time_of_day - 6) / 12 * math.pi  # 6am = 0°, 6pm = π
        
        # Update sun direction
        sun_direction = Vector3(
            math.sin(sun_angle),
            -math.cos(sun_angle),
            0
        ).normalize()
        
        # Update sun color based on time
        if time_of_day < 6 or time_of_day > 18:  # Night
            sun_color = [0.1, 0.1, 0.3]
            sun_intensity = 0.1
        elif time_of_day < 8:  # Dawn
            sun_color = [1.0, 0.6, 0.4]
            sun_intensity = 0.5
        elif time_of_day > 16:  # Dusk
            sun_color = [1.0, 0.7, 0.5]
            sun_intensity = 0.6
        else:  # Day
            sun_color = [1.0, 1.0, 0.9]
            sun_intensity = 1.0
        
        # Adjust for weather
        if weather == 'rainy':
            sun_intensity *= 0.4
            sun_color = [s * 0.7 for s in sun_color]
        elif weather == 'stormy':
            sun_intensity *= 0.2
            sun_color = [s * 0.5 for s in sun_color]
        elif weather == 'cloudy':
            sun_intensity *= 0.7
            sun_color = [s * 0.8 for s in sun_color]
        
        # Update sun light
        for light in self.light_sources:
            if light['type'] == 'directional':
                light['direction'] = sun_direction.to_dict()
                light['color'] = sun_color
                light['intensity'] = sun_intensity
    
    def update_post_effects(self, frame_data):
        """Update post-processing effects based on game state."""
        speed = frame_data.get('speed', 0)
        weather = frame_data.get('weather', 'clear')
        
        # Adjust motion blur based on speed
        for effect in self.post_effects:
            if effect['name'] == 'motion_blur':
                effect['strength'] = min(0.5, speed / 200 * 0.3)
            
            # Adjust bloom for weather
            if effect['name'] == 'bloom' and weather == 'rainy':
                effect['strength'] = 0.7
                effect['threshold'] = 0.6
    
    def generate_particles(self, player_pos, weather):
        """Generate particle effects."""
        particles = []
        
        # Weather particles
        if weather == 'rainy':
            for _ in range(50):
                particle = {
                    'type': 'rain',
                    'position': [
                        player_pos.x + random.uniform(-50, 50),
                        player_pos.y + random.uniform(10, 50),
                        player_pos.z + random.uniform(-50, 50)
                    ],
                    'velocity': [0, -20, 0],
                    'size': random.uniform(0.1, 0.3),
                    'lifetime': 2.0
                }
                particles.append(particle)
        
        elif weather == 'stormy':
            for _ in range(100):
                particle = {
                    'type': 'rain',
                    'position': [
                        player_pos.x + random.uniform(-100, 100),
                        player_pos.y + random.uniform(20, 100),
                        player_pos.z + random.uniform(-100, 100)
                    ],
                    'velocity': [
                        random.uniform(-5, 5),
                        -30,
                        random.uniform(-5, 5)
                    ],
                    'size': random.uniform(0.2, 0.4),
                    'lifetime': 1.5
                }
                particles.append(particle)
        
        # Dust/smoke particles from car
        if random.random() < 0.3:
            for _ in range(10):
                particle = {
                    'type': 'dust',
                    'position': [
                        player_pos.x + random.uniform(-2, 2),
                        0.5,
                        player_pos.z + random.uniform(-2, 2)
                    ],
                    'velocity': [
                        random.uniform(-1, 1),
                        random.uniform(0.5, 2),
                        random.uniform(-1, 1)
                    ],
                    'size': random.uniform(0.5, 1.5),
                    'lifetime': random.uniform(1, 3)
                }
                particles.append(particle)
        
        return particles
    
    def get_weather_effects(self, weather):
        """Get weather-specific visual effects."""
        effects = {
            'fog_density': 0.001,
            'fog_color': [0.8, 0.8, 0.9],
            'cloud_cover': 0.3,
            'precipitation': 0.0,
            'wind_strength': 0.2
        }
        
        if weather == 'rainy':
            effects.update({
                'fog_density': 0.003,
                'fog_color': [0.6, 0.6, 0.7],
                'cloud_cover': 0.9,
                'precipitation': 0.8,
                'wind_strength': 0.8
            })
        elif weather == 'stormy':
            effects.update({
                'fog_density': 0.005,
                'fog_color': [0.4, 0.4, 0.5],
                'cloud_cover': 1.0,
                'precipitation': 1.0,
                'wind_strength': 1.0
            })
        elif weather == 'cloudy':
            effects.update({
                'fog_density': 0.002,
                'fog_color': [0.7, 0.7, 0.8],
                'cloud_cover': 0.7,
                'precipitation': 0.1,
                'wind_strength': 0.5
            })
        
        return effects
    
    def get_lighting_data(self):
        """Get current lighting data for rendering."""
        return {
            'lights': self.light_sources,
            'ambient_occlusion': True,
            'global_illumination': True,
            'shadow_map_size': 2048,
            'shadow_softness': 0.8
        }
    
    def calculate_lod(self, player_pos):
        """Calculate level of detail based on distance."""
        distance = player_pos.length()
        
        if distance < 100:
            return 'high'
        elif distance < 300:
            return 'medium'
        else:
            return 'low'
    
    def optimize_graphics(self, optimization_request):
        """Optimize graphics settings based on performance."""
        fps = optimization_request.get('fps', 60)
        gpu_usage = optimization_request.get('gpu_usage', 50)
        
        adjustments = {}
        
        # Adjust quality based on performance
        if fps < 30:
            adjustments['texture_quality'] = 'medium'
            adjustments['shadow_quality'] = 'low'
            adjustments['particle_count'] = 50
        elif fps < 45:
            adjustments['shadow_quality'] = 'medium'
            adjustments['particle_count'] = 100
        else:
            adjustments['texture_quality'] = 'high'
            adjustments['shadow_quality'] = 'high'
            adjustments['particle_count'] = 200
        
        # Adjust based on GPU usage
        if gpu_usage > 80:
            adjustments['post_effects'] = ['bloom', 'color_grading']  # Only essential effects
        elif gpu_usage > 60:
            adjustments['post_effects'] = ['bloom', 'color_grading', 'vignette']
        else:
            adjustments['post_effects'] = [effect['name'] for effect in self.post_effects]
        
        return adjustments
    
    def render_debug_info(self):
        """Render debug information for development."""
        return {
            'terrain_triangles': self.frame_counter * 1000,
            'particle_count': len(self.particle_systems) * 100,
            'light_count': len(self.light_sources),
            'memory_usage': 256,  # MB
            'gpu_memory': 512,    # MB
            'render_time': 16.7   # ms (60 FPS)
        }

class WeatherSystem:
    """Manages weather effects and transitions."""
    
    def __init__(self):
        self.current_weather = 'clear'
        self.target_weather = 'clear'
        self.transition_progress = 0
        self.transition_speed = 0.1
        
        self.weather_patterns = {
            'clear': {'duration': 300, 'intensity': 0.1},
            'cloudy': {'duration': 180, 'intensity': 0.3},
            'rainy': {'duration': 120, 'intensity': 0.7},
            'stormy': {'duration': 60, 'intensity': 1.0}
        }
        
        self.next_weather_change = 0
        self.update_next_change()
    
    def update(self, delta_time):
        """Update weather system."""
        # Update transition
        if self.transition_progress < 1:
            self.transition_progress = min(1, self.transition_progress + self.transition_speed * delta_time)
        
        # Check for weather change
        if self.next_weather_change <= 0:
            self.change_weather()
            self.update_next_change()
        else:
            self.next_weather_change -= delta_time
    
    def change_weather(self):
        """Change to a new weather pattern."""
        weather_types = list(self.weather_patterns.keys())
        # Weighted random selection
        weights = [0.4, 0.3, 0.2, 0.1]  # clear, cloudy, rainy, stormy
        
        self.target_weather = random.choices(weather_types, weights=weights, k=1)[0]
        self.transition_progress = 0
        
        print(f"Weather changing to: {self.target_weather}")
    
    def update_next_change(self):
        """Update when next weather change will occur."""
        current_pattern = self.weather_patterns.get(self.target_weather, self.weather_patterns['clear'])
        self.next_weather_change = current_pattern['duration'] * random.uniform(0.8, 1.2)
    
    def get_weather_data(self):
        """Get current weather data for rendering."""
        # Interpolate between current and target weather
        if self.transition_progress < 1:
            # Blend weather effects during transition
            current = self.weather_patterns[self.current_weather]
            target = self.weather_patterns[self.target_weather]
            
            intensity = current['intensity'] * (1 - self.transition_progress) + target['intensity'] * self.transition_progress
            
            if self.transition_progress >= 1:
                self.current_weather = self.target_weather
        else:
            intensity = self.weather_patterns[self.current_weather]['intensity']
        
        weather_data = {
            'type': self.current_weather,
            'intensity': intensity,
            'in_transition': self.transition_progress < 1,
            'transition_progress': self.transition_progress,
            'next_change': self.next_weather_change
        }
        
        # Add weather-specific effects
        if self.current_weather == 'rainy':
            weather_data.update({
                'rain_density': intensity,
                'rain_direction': [0, -1, 0],
                'puddle_amount': intensity * 0.8,
                'wetness': intensity * 0.9
            })
        elif self.current_weather == 'stormy':
            weather_data.update({
                'rain_density': intensity,
                'rain_direction': [random.uniform(-0.5, 0.5), -1, random.uniform(-0.5, 0.5)],
                'puddle_amount': intensity,
                'wetness': 1.0,
                'lightning_chance': 0.1
            })
        elif self.current_weather == 'cloudy':
            weather_data.update({
                'cloud_density': intensity,
                'cloud_speed': 0.5,
                'shadow_intensity': 1.0 - intensity * 0.5
            })
        
        return weather_data

if __name__ == "__main__":
    # Test the graphics engine
    print("Testing Graphics Engine...")
    
    graphics = GraphicsProcessor()
    weather = WeatherSystem()
    
    # Test frame processing
    test_frame = {
        'player_position': {'x': 0, 'y': 0, 'z': 0},
        'camera_position': {'x': 0, 'y': 15, 'z': 25},
        'time_of_day': 12.0,
        'weather': 'clear',
        'speed': 80
    }
    
    result = graphics.process_frame(test_frame)
    print(f"Graphics processing complete:")
    print(f"- Frame: {result['frame_number']}")
    print(f"- Processing time: {result['processing_time']:.2f}ms")
    print(f"- Terrain vertices: {len(result['terrain_data']['vertices'])}")
    print(f"- Draw calls: {result['optimization']['draw_calls']}")
    
    # Test weather system
    weather.update(1.0)
    weather_data = weather.get_weather_data()
    print(f"\nWeather: {weather_data['type']} (intensity: {weather_data['intensity']:.2f})")