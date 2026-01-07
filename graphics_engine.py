import numpy as np
import math
import random
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
from collections import deque
from dataclasses import dataclass
from enum import Enum

class WeatherType(Enum):
    CLEAR = "clear"
    CLOUDY = "cloudy"
    RAINY = "rainy"
    STORMY = "stormy"
    SNOWY = "snowy"
    FOGGY = "foggy"

@dataclass
class QualitySettings:
    """Graphics quality settings."""
    texture_quality: str = "high"
    shadow_quality: str = "high"
    water_quality: str = "high"
    particle_count: int = 200
    post_effects: List[str] = None
    anti_aliasing: str = "msaa4x"
    anisotropic_filtering: int = 16
    render_distance: float = 1000.0
    
    def __post_init__(self):
        if self.post_effects is None:
            self.post_effects = ["bloom", "color_grading", "ssao", "motion_blur"]

class AdvancedVector3(Vector3):
    """Enhanced 3D vector with more operations."""
    
    def __init__(self, x=0, y=0, z=0):
        super().__init__(x, y, z)
    
    def reflect(self, normal):
        """Calculate reflection vector."""
        return self - normal * (2 * self.dot(normal))
    
    def lerp(self, other, t):
        """Linear interpolation."""
        return AdvancedVector3(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
            self.z + (other.z - self.z) * t
        )
    
    def rotate(self, axis, angle):
        """Rotate vector around axis."""
        axis = axis.normalize()
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Rodrigues' rotation formula
        dot = self.dot(axis)
        cross = self.cross(axis)
        
        return self * cos_a + cross * sin_a + axis * dot * (1 - cos_a)

class AdvancedTerrainGenerator(TerrainGenerator):
    """Enhanced terrain generator with more features."""
    
    def __init__(self, size=1000, resolution=1024):
        super().__init__(size, resolution)
        self.biome_map = None
        self.water_level = 20.0
        self.generate_biomes()
        self.generate_rivers()
        self.generate_vegetation()
    
    def generate_biomes(self):
        """Generate different biomes based on height and moisture."""
        print("Generating biomes...")
        
        self.biome_map = np.zeros((self.resolution, self.resolution, 3))
        
        # Generate moisture map
        moisture = np.zeros((self.resolution, self.resolution))
        for octave in range(3):
            frequency = 3 ** octave
            amplitude = 1.0 / (frequency * 1.5)
            
            for i in range(self.resolution):
                for j in range(self.resolution):
                    nx = i / self.resolution * frequency
                    ny = j / self.resolution * frequency
                    noise = (math.sin(nx * 7.483 + ny * 34.156) * 19873.482) % 1
                    moisture[i][j] += noise * amplitude
        
        # Assign biomes based on height and moisture
        for i in range(self.resolution):
            for j in range(self.resolution):
                height = self.height_map[i][j]
                moist = moisture[i][j]
                
                if height < 10:
                    # Water
                    self.biome_map[i][j] = [0.2, 0.4, 0.8]
                elif height < 30:
                    if moist > 0.6:
                        # Swamp
                        self.biome_map[i][j] = [0.3, 0.5, 0.2]
                    else:
                        # Grassland
                        self.biome_map[i][j] = [0.4, 0.7, 0.3]
                elif height < 60:
                    if moist > 0.7:
                        # Forest
                        self.biome_map[i][j] = [0.2, 0.6, 0.2]
                    else:
                        # Plains
                        self.biome_map[i][j] = [0.6, 0.8, 0.4]
                else:
                    if moist > 0.8:
                        # Alpine
                        self.biome_map[i][j] = [0.8, 0.9, 1.0]
                    else:
                        # Mountain
                        self.biome_map[i][j] = [0.7, 0.6, 0.5]
    
    def generate_rivers(self):
        """Generate river networks."""
        print("Generating rivers...")
        
        # Simple river generation algorithm
        river_height = np.copy(self.height_map)
        
        # Find high points that could be river sources
        sources = []
        for i in range(10, self.resolution - 10, 20):
            for j in range(10, self.resolution - 10, 20):
                if self.height_map[i][j] > 50:  # High enough for rivers
                    sources.append((i, j))
        
        # Generate rivers from sources
        for source in sources[:5]:  # Limit number of rivers
            x, y = source
            path = []
            
            # Flow downhill
            for _ in range(100):
                if x <= 1 or x >= self.resolution - 2 or y <= 1 or y >= self.resolution - 2:
                    break
                
                path.append((x, y))
                
                # Find lowest neighbor
                min_height = self.height_map[x][y]
                next_x, next_y = x, y
                
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        
                        nx, ny = x + dx, y + dy
                        if self.height_map[nx][ny] < min_height:
                            min_height = self.height_map[nx][ny]
                            next_x, next_y = nx, ny
                
                if next_x == x and next_y == y:
                    break
                
                x, y = next_x, next_y
                
                # Lower height to create riverbed
                self.height_map[x][y] -= 5.0
        
        self.recalculate_normals()
    
    def generate_vegetation(self):
        """Generate vegetation density map."""
        print("Generating vegetation...")
        
        self.vegetation_map = np.zeros((self.resolution, self.resolution))
        
        for i in range(self.resolution):
            for j in range(self.resolution):
                height = self.height_map[i][j]
                biome = self.biome_map[i][j]
                
                # Vegetation based on biome and height
                if height < 10:  # Water
                    density = 0.0
                elif biome[1] > 0.5:  # Green areas
                    if height < 40:
                        density = random.uniform(0.7, 1.0)
                    else:
                        density = random.uniform(0.3, 0.6)
                else:
                    density = random.uniform(0.1, 0.3)
                
                self.vegetation_map[i][j] = density
    
    def recalculate_normals(self):
        """Recalculate normals after terrain modification."""
        super().calculate_normals()
    
    def add_detail_noise(self):
        """Add high-frequency detail noise."""
        print("Adding detail noise...")
        
        detail_map = np.zeros((self.resolution, self.resolution))
        
        for octave in range(5):
            frequency = 8 ** octave
            amplitude = 1.0 / (frequency * 3)
            
            for i in range(self.resolution):
                for j in range(self.resolution):
                    nx = i / self.resolution * frequency
                    ny = j / self.resolution * frequency
                    
                    noise1 = math.sin(nx * 32.145 + ny * 17.483) * 43758.5453
                    noise2 = math.cos(nx * 21.789 + ny * 46.123) * 28745.982
                    noise = ((noise1 + noise2) % 1 - 0.5) * 2
                    
                    detail_map[i][j] += noise * amplitude
        
        # Apply detail noise
        self.height_map += detail_map * 2
    
    def generate_terrain_data(self, bounds=None, lod_level='high'):
        """Generate terrain data with LOD support."""
        if bounds is None:
            bounds = {
                'min_x': -self.size/2,
                'max_x': self.size/2,
                'min_z': -self.size/2,
                'max_z': self.size/2
            }
        
        # Determine sample density based on LOD
        if lod_level == 'high':
            samples_x = 200
            samples_z = 200
        elif lod_level == 'medium':
            samples_x = 100
            samples_z = 100
        else:  # low
            samples_x = 50
            samples_z = 50
        
        terrain_data = {
            'vertices': [],
            'normals': [],
            'textures': [],
            'colors': [],  # Biome colors
            'indices': [],
            'bounds': bounds,
            'lod': lod_level
        }
        
        # Generate grid
        for i in range(samples_x):
            for j in range(samples_z):
                x = bounds['min_x'] + (bounds['max_x'] - bounds['min_x']) * i / (samples_x - 1)
                z = bounds['min_z'] + (bounds['max_z'] - bounds['min_z']) * j / (samples_z - 1)
                y = self.get_height(x, z)
                
                terrain_data['vertices'].append([x, y, z])
                
                normal = self.get_normal(x, z)
                terrain_data['normals'].append([normal.x, normal.y, normal.z])
                
                # Get biome color
                terrain_data['colors'].append(self.get_biome_color(x, z))
                
                # Texture coordinates with detail tiling
                u = (x - bounds['min_x']) / (bounds['max_x'] - bounds['min_x']) * 10
                v = (z - bounds['min_z']) / (bounds['max_z'] - bounds['min_z']) * 10
                terrain_data['textures'].append([u, v])
        
        # Generate indices
        for i in range(samples_x - 1):
            for j in range(samples_z - 1):
                a = i * samples_z + j
                b = (i + 1) * samples_z + j
                c = i * samples_z + (j + 1)
                d = (i + 1) * samples_z + (j + 1)
                
                terrain_data['indices'].extend([a, b, c, b, d, c])
        
        return terrain_data
    
    def get_biome_color(self, x, z):
        """Get biome color at world coordinates."""
        i = int((x / self.size + 0.5) * self.resolution)
        j = int((z / self.size + 0.5) * self.resolution)
        
        i = max(0, min(self.resolution - 1, i))
        j = max(0, min(self.resolution - 1, j))
        
        return list(self.biome_map[i][j])

class WaterRenderer:
    """Advanced water rendering system."""
    
    def __init__(self, terrain_generator):
        self.terrain = terrain_generator
        self.water_level = terrain_generator.water_level
        self.wave_speed = 0.5
        self.wave_amplitude = 0.3
        self.water_color = [0.2, 0.5, 0.8, 0.7]
        self.refraction_strength = 0.1
        self.reflection_strength = 0.8
        
        # Generate water mesh
        self.generate_water_mesh()
    
    def generate_water_mesh(self):
        """Generate water surface mesh."""
        self.water_vertices = []
        self.water_normals = []
        self.water_indices = []
        
        grid_size = 50
        grid_spacing = self.terrain.size / grid_size
        
        # Create grid
        for i in range(grid_size + 1):
            for j in range(grid_size + 1):
                x = -self.terrain.size/2 + i * grid_spacing
                z = -self.terrain.size/2 + j * grid_spacing
                
                # Basic wave function
                time = datetime.now().timestamp() * self.wave_speed
                wave = math.sin(x * 0.1 + time) * math.cos(z * 0.1 + time) * self.wave_amplitude
                
                y = self.water_level + wave
                
                self.water_vertices.append([x, y, z])
                
                # Calculate normal for waves
                dx = math.cos(x * 0.1 + time) * math.cos(z * 0.1 + time) * 0.1
                dz = math.sin(x * 0.1 + time) * -math.sin(z * 0.1 + time) * 0.1
                normal = AdvancedVector3(-dx, 1, -dz).normalize()
                self.water_normals.append([normal.x, normal.y, normal.z])
        
        # Create indices
        for i in range(grid_size):
            for j in range(grid_size):
                a = i * (grid_size + 1) + j
                b = (i + 1) * (grid_size + 1) + j
                c = i * (grid_size + 1) + (j + 1)
                d = (i + 1) * (grid_size + 1) + (j + 1)
                
                self.water_indices.extend([a, b, c, b, d, c])
    
    def update(self, delta_time):
        """Update water animation."""
        # Update wave animation
        pass  # Animation is time-based in generate_water_mesh
    
    def get_water_data(self):
        """Get water rendering data."""
        # Regenerate mesh with updated time
        self.generate_water_mesh()
        
        return {
            'vertices': self.water_vertices,
            'normals': self.water_normals,
            'indices': self.water_indices,
            'color': self.water_color,
            'refraction_strength': self.refraction_strength,
            'reflection_strength': self.reflection_strength,
            'wave_amplitude': self.wave_amplitude
        }

class AdvancedGraphicsProcessor(GraphicsProcessor):
    """Enhanced graphics processor with more features."""
    
    def __init__(self):
        super().__init__()
        
        # Replace with advanced terrain
        self.terrain_generator = AdvancedTerrainGenerator(size=2000, resolution=2048)
        
        # Add new systems
        self.water_renderer = WaterRenderer(self.terrain_generator)
        self.quality_settings = QualitySettings()
        self.shadow_map = None
        self.ssao_map = None
        self.reflection_map = None
        self.frame_history = deque(maxlen=60)
        
        # Initialize advanced features
        self.setup_shadows()
        self.setup_ssao()
        self.setup_reflections()
        
        # Performance monitoring
        self.performance_stats = {
            'fps': 60,
            'gpu_memory': 0,
            'cpu_time': 0,
            'gpu_time': 0,
            'triangle_count': 0
        }
    
    def setup_shadows(self):
        """Setup shadow mapping system."""
        self.shadow_map = {
            'resolution': 4096,
            'bias': 0.001,
            'softness': 0.5,
            'cascades': 4,
            'max_distance': 1000
        }
    
    def setup_ssao(self):
        """Setup Screen Space Ambient Occlusion."""
        self.ssao_map = {
            'radius': 0.5,
            'strength': 1.0,
            'samples': 16,
            'noise_size': 4
        }
    
    def setup_reflections(self):
        """Setup reflection/refraction system."""
        self.reflection_map = {
            'resolution': 1024,
            'fresnel_power': 5.0,
            'reflection_blur': 0.5,
            'refraction_blur': 0.3
        }
    
    def process_frame(self, frame_data):
        """Process a frame with advanced features."""
        start_time = datetime.now()
        self.frame_counter += 1
        
        # Extract data
        player_pos = AdvancedVector3.from_dict(frame_data.get('player_position', {}))
        camera_pos = AdvancedVector3.from_dict(frame_data.get('camera_position', {}))
        camera_rotation = frame_data.get('camera_rotation', {'x': 0, 'y': 0, 'z': 0})
        time_of_day = frame_data.get('time_of_day', 12.0)
        weather = frame_data.get('weather', 'clear')
        delta_time = frame_data.get('delta_time', 0.016)
        
        # Update systems
        self.water_renderer.update(delta_time)
        
        # Update lighting with HDR support
        self.update_hdr_lighting(time_of_day, weather)
        
        # Calculate LOD based on distance and performance
        lod_level = self.calculate_adaptive_lod(player_pos)
        
        # Generate terrain with LOD
        view_bounds = self.calculate_view_bounds(camera_pos)
        terrain_data = self.terrain_generator.generate_terrain_data(view_bounds, lod_level)
        
        # Generate water data
        water_data = self.water_renderer.get_water_data()
        
        # Generate advanced particle effects
        particle_data = self.generate_advanced_particles(player_pos, weather, time_of_day)
        
        # Generate atmospheric effects
        atmosphere_data = self.generate_atmospheric_effects(camera_pos, time_of_day)
        
        # Generate procedural vegetation
        vegetation_data = self.generate_vegetation_data(view_bounds)
        
        # Calculate performance metrics
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Update performance stats
        self.update_performance_stats(terrain_data, processing_time)
        
        # Prepare advanced response
        response = {
            'frame_number': self.frame_counter,
            'processing_time': processing_time,
            'terrain_data': terrain_data,
            'water_data': water_data,
            'lighting': self.get_advanced_lighting_data(),
            'post_effects': self.get_advanced_post_effects(),
            'particles': particle_data,
            'weather_effects': self.get_advanced_weather_effects(weather),
            'atmosphere': atmosphere_data,
            'vegetation': vegetation_data,
            'optimization': {
                'lod_level': lod_level,
                'culling_applied': True,
                'occlusion_culling': True,
                'frustum_culling': True,
                'quality_settings': self.quality_settings.__dict__,
                'triangle_count': len(terrain_data.get('indices', [])) // 3,
                'draw_calls': self.calculate_draw_calls(terrain_data, vegetation_data)
            },
            'shadows': self.shadow_map,
            'ssao': self.ssao_map,
            'reflections': self.reflection_map,
            'performance': self.performance_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Store frame for motion blur
        self.frame_history.append(response)
        
        return response
    
    def update_hdr_lighting(self, time_of_day, weather):
        """Update HDR lighting with physical-based parameters."""
        # Physical sun model
        solar_altitude = (time_of_day - 12) * 15  # degrees
        
        # Convert to radians
        sun_angle = math.radians(solar_altitude)
        
        # Calculate sun position
        sun_direction = AdvancedVector3(
            math.cos(sun_angle),
            math.sin(sun_angle),
            0.2  # Small Z component for depth
        ).normalize()
        
        # Calculate sun color based on atmospheric scattering
        if solar_altitude > 10:  # Day
            sun_color = [1.0, 1.0, 0.95]
            sun_intensity = 100000.0  # lux
            exposure = 1.0
        elif solar_altitude > 0:  # Morning/Evening
            sun_color = [1.0, 0.8, 0.6]
            sun_intensity = 30000.0
            exposure = 0.8
        else:  # Night
            sun_color = [0.1, 0.1, 0.3]
            sun_intensity = 1000.0
            exposure = 0.3
        
        # Apply weather effects
        if weather == 'rainy':
            sun_intensity *= 0.3
            exposure *= 0.6
        elif weather == 'stormy':
            sun_intensity *= 0.1
            exposure *= 0.4
        elif weather == 'cloudy':
            sun_intensity *= 0.6
            exposure *= 0.7
        
        # Update sun light
        for light in self.light_sources:
            if light['type'] == 'directional':
                light['direction'] = sun_direction.to_dict()
                light['color'] = sun_color
                light['intensity'] = sun_intensity
                light['exposure'] = exposure
        
        # Update ambient based on sun position
        for light in self.light_sources:
            if light['type'] == 'ambient':
                ambient_intensity = max(0.1, math.sin(sun_angle + math.pi/2) * 0.3)
                light['intensity'] = ambient_intensity
    
    def generate_advanced_particles(self, player_pos, weather, time_of_day):
        """Generate advanced particle effects with physics."""
        particles = []
        
        # Weather particles with physics
        if weather in ['rainy', 'stormy']:
            rain_density = 100 if weather == 'rainy' else 200
            
            for _ in range(rain_density):
                # Rain drops with size variation
                size = random.uniform(0.1, 0.3)
                speed = random.uniform(15, 25) if weather == 'rainy' else random.uniform(25, 40)
                
                particle = {
                    'type': 'rain',
                    'position': [
                        player_pos.x + random.uniform(-100, 100),
                        player_pos.y + random.uniform(20, 100),
                        player_pos.z + random.uniform(-100, 100)
                    ],
                    'velocity': [
                        random.uniform(-3, 3) if weather == 'stormy' else 0,
                        -speed,
                        random.uniform(-3, 3) if weather == 'stormy' else 0
                    ],
                    'size': size,
                    'lifetime': random.uniform(1.5, 3.0),
                    'color': [0.7, 0.8, 1.0, 0.8],
                    'gravity': 30.0,
                    'wind_effect': True
                }
                particles.append(particle)
        
        # Snow particles
        elif weather == 'snowy':
            for _ in range(150):
                particle = {
                    'type': 'snow',
                    'position': [
                        player_pos.x + random.uniform(-150, 150),
                        player_pos.y + random.uniform(50, 200),
                        player_pos.z + random.uniform(-150, 150)
                    ],
                    'velocity': [
                        random.uniform(-2, 2),
                        random.uniform(-3, -1),
                        random.uniform(-2, 2)
                    ],
                    'size': random.uniform(0.3, 0.8),
                    'lifetime': random.uniform(5, 10),
                    'color': [1.0, 1.0, 1.0, 0.9],
                    'gravity': 5.0,
                    'swirl': True
                }
                particles.append(particle)
        
        # Dust particles with wind
        for _ in range(20):
            if random.random() < 0.1:
                particle = {
                    'type': 'dust',
                    'position': [
                        player_pos.x + random.uniform(-5, 5),
                        0.2,
                        player_pos.z + random.uniform(-5, 5)
                    ],
                    'velocity': [
                        random.uniform(-0.5, 0.5),
                        random.uniform(0.1, 0.5),
                        random.uniform(-0.5, 0.5)
                    ],
                    'size': random.uniform(0.2, 1.0),
                    'lifetime': random.uniform(2, 5),
                    'color': [0.8, 0.7, 0.5, 0.3],
                    'fade': True
                }
                particles.append(particle)
        
        # Fireflies at night
        if time_of_day > 18 or time_of_day < 6:
            for _ in range(30):
                particle = {
                    'type': 'firefly',
                    'position': [
                        player_pos.x + random.uniform(-20, 20),
                        random.uniform(1, 3),
                        player_pos.z + random.uniform(-20, 20)
                    ],
                    'velocity': [
                        random.uniform(-0.1, 0.1),
                        0,
                        random.uniform(-0.1, 0.1)
                    ],
                    'size': random.uniform(0.1, 0.2),
                    'lifetime': random.uniform(3, 6),
                    'color': [1.0, 1.0, 0.5, 0.8],
                    'pulse': True,
                    'pulse_speed': random.uniform(2, 4)
                }
                particles.append(particle)
        
        return particles
    
    def generate_atmospheric_effects(self, camera_pos, time_of_day):
        """Generate atmospheric scattering and fog."""
        # Rayleigh scattering simulation
        sun_angle = (time_of_day - 12) / 12 * math.pi
        
        # Base atmospheric color
        if time_of_day < 6 or time_of_day > 18:  # Night
            sky_color = [0.05, 0.05, 0.15]
            horizon_color = [0.1, 0.1, 0.3]
        elif time_of_day < 8:  # Dawn
            sky_color = [0.3, 0.2, 0.4]
            horizon_color = [0.8, 0.5, 0.3]
        elif time_of_day > 16:  # Dusk
            sky_color = [0.2, 0.2, 0.4]
            horizon_color = [0.9, 0.6, 0.4]
        else:  # Day
            sky_color = [0.5, 0.7, 1.0]
            horizon_color = [0.8, 0.9, 1.0]
        
        # Volumetric fog
        fog_density = 0.001 + (math.sin(time_of_day * math.pi / 12) * 0.002)
        fog_height = 50.0
        fog_falloff = 0.5
        
        return {
            'sky_color': sky_color,
            'horizon_color': horizon_color,
            'sun_color': [1.0, 0.9, 0.8],
            'moon_color': [0.8, 0.9, 1.0],
            'star_intensity': max(0, (18 - time_of_day) / 6) if time_of_day > 18 else max(0, (6 - time_of_day) / 6),
            'cloud_coverage': 0.3 + math.sin(time_of_day * 0.1) * 0.2,
            'fog': {
                'density': fog_density,
                'color': horizon_color,
                'height': fog_height,
                'falloff': fog_falloff,
                'scattering': 0.8
            },
            'god_rays': {
                'enabled': True,
                'intensity': 0.3,
                'samples': 16,
                'decay': 0.95
            }
        }
    
    def generate_vegetation_data(self, bounds):
        """Generate procedural vegetation placement."""
        vegetation = {
            'trees': [],
            'grass': [],
            'rocks': [],
            'flowers': []
        }
        
        # Generate trees
        tree_count = 100
        for _ in range(tree_count):
            x = random.uniform(bounds['min_x'], bounds['max_x'])
            z = random.uniform(bounds['min_z'], bounds['max_z'])
            y = self.terrain_generator.get_height(x, z)
            
            # Check if suitable for trees (not water, not too steep)
            normal = self.terrain_generator.get_normal(x, z)
            slope = 1.0 - normal.y  # 0 = flat, 1 = vertical
            
            if y > self.terrain_generator.water_level + 5 and slope < 0.5:
                tree_type = random.choice(['oak', 'pine', 'birch'])
                height = random.uniform(5, 15)
                radius = random.uniform(0.5, 1.5)
                
                vegetation['trees'].append({
                    'position': [x, y, z],
                    'type': tree_type,
                    'height': height,
                    'radius': radius,
                    'variation': random.uniform(0.8, 1.2)
                })
        
        # Generate grass patches
        grass_count = 500
        for _ in range(grass_count):
            x = random.uniform(bounds['min_x'], bounds['max_x'])
            z = random.uniform(bounds['min_z'], bounds['max_z'])
            y = self.terrain_generator.get_height(x, z)
            
            if y > self.terrain_generator.water_level + 2:
                vegetation['grass'].append({
                    'position': [x, y, z],
                    'density': random.uniform(0.5, 1.0),
                    'height': random.uniform(0.3, 1.0),
                    'wind_effect': True
                })
        
        return vegetation
    
    def calculate_adaptive_lod(self, player_pos):
        """Calculate adaptive LOD based on distance and performance."""
        distance = player_pos.length()
        
        # Base LOD on distance
        if distance < 100:
            base_lod = 'high'
        elif distance < 300:
            base_lod = 'medium'
        else:
            base_lod = 'low'
        
        # Adjust based on performance
        if self.performance_stats['fps'] < 30:
            # Downgrade LOD if performance is poor
            if base_lod == 'high':
                base_lod = 'medium'
            elif base_lod == 'medium':
                base_lod = 'low'
        
        return base_lod
    
    def calculate_draw_calls(self, terrain_data, vegetation_data):
        """Calculate total draw calls for the frame."""
        # Terrain draw calls
        terrain_triangles = len(terrain_data.get('indices', [])) // 3
        terrain_draw_calls = max(1, terrain_triangles // 1000)  # Batch triangles
        
        # Vegetation draw calls (instanced)
        tree_draw_calls = len(vegetation_data.get('trees', [])) // 50  # Batch trees
        grass_draw_calls = len(vegetation_data.get('grass', [])) // 100  # Batch grass
        
        # Particle draw calls
        particle_draw_calls = 3  # One per particle system
        
        # Total draw calls
        total = (terrain_draw_calls + tree_draw_calls + 
                grass_draw_calls + particle_draw_calls + 5)  # +5 for water, sky, etc.
        
        return total
    
    def update_performance_stats(self, terrain_data, processing_time):
        """Update performance statistics."""
        triangle_count = len(terrain_data.get('indices', [])) // 3
        
        # Calculate FPS
        if processing_time > 0:
            current_fps = 1000 / processing_time
            # Smooth FPS calculation
            self.performance_stats['fps'] = self.performance_stats['fps'] * 0.9 + current_fps * 0.1
        
        self.performance_stats.update({
            'triangle_count': triangle_count,
            'cpu_time': processing_time,
            'gpu_time': processing_time * 0.7,  # Estimate
            'memory_usage_mb': triangle_count * 0.01 + 100  # Estimate
        })
    
    def get_advanced_lighting_data(self):
        """Get advanced lighting data with HDR support."""
        return {
            'lights': self.light_sources,
            'ambient_occlusion': {
                'enabled': True,
                'radius': self.ssao_map['radius'],
                'strength': self.ssao_map['strength']
            },
            'global_illumination': {
                'enabled': True,
                'bounces': 2,
                'intensity': 0.5
            },
            'shadows': self.shadow_map,
            'hdr': {
                'enabled': True,
                'exposure': 1.0,
                'bloom_threshold': 0.8,
                'bloom_strength': 0.5
            },
            'volumetric_lighting': {
                'enabled': True,
                'samples': 32,
                'scattering': 0.8
            }
        }
    
    def get_advanced_post_effects(self):
        """Get advanced post-processing effects."""
        return [
            {
                'name': 'bloom',
                'enabled': True,
                'strength': 0.5,
                'radius': 0.8,
                'threshold': 0.7,
                'high_quality': True
            },
            {
                'name': 'motion_blur',
                'enabled': True,
                'strength': 0.3,
                'samples': 16,
                'velocity_scale': 1.0
            },
            {
                'name': 'color_grading',
                'enabled': True,
                'lut_texture': 'filmic',
                'temperature': 6500,
                'tint': 0.0,
                'contrast': 1.1,
                'saturation': 1.2,
                'brightness': 1.0
            },
            {
                'name': 'vignette',
                'enabled': True,
                'strength': 0.3,
                'roundness': 0.8,
                'smoothness': 0.5
            },
            {
                'name': 'chromatic_aberration',
                'enabled': False,
                'strength': 0.1,
                'samples': 3
            },
            {
                'name': 'film_grain',
                'enabled': False,
                'strength': 0.05,
                'size': 1.0
            },
            {
                'name': 'depth_of_field',
                'enabled': True,
                'focus_distance': 50.0,
                'aperture': 2.8,
                'focal_length': 50.0
            }
        ]
    
    def get_advanced_weather_effects(self, weather):
        """Get advanced weather-specific visual effects."""
        effects = {
            'fog': {
                'density': 0.001,
                'height': 50,
                'falloff': 0.5,
                'scattering': 0.8,
                'color': [0.8, 0.8, 0.9]
            },
            'clouds': {
                'coverage': 0.3,
                'thickness': 0.5,
                'speed': 0.5,
                'shadow_intensity': 0.7
            },
            'precipitation': {
                'intensity': 0.0,
                'type': 'none',
                'wind_influence': 0.0
            },
            'wind': {
                'strength': 0.2,
                'direction': [1, 0, 0],
                'gustiness': 0.3
            },
            'wetness': {
                'amount': 0.0,
                'puddles': False,
                'reflection_strength': 0.0
            }
        }
        
        if weather == 'rainy':
            effects.update({
                'fog': {
                    'density': 0.003,
                    'height': 30,
                    'falloff': 0.7,
                    'scattering': 0.9,
                    'color': [0.6, 0.6, 0.7]
                },
                'clouds': {
                    'coverage': 0.9,
                    'thickness': 0.8,
                    'speed': 1.0,
                    'shadow_intensity': 0.9
                },
                'precipitation': {
                    'intensity': 0.8,
                    'type': 'rain',
                    'wind_influence': 0.5
                },
                'wind': {
                    'strength': 0.8,
                    'direction': [1, 0, 0.5],
                    'gustiness': 0.5
                },
                'wetness': {
                    'amount': 0.8,
                    'puddles': True,
                    'reflection_strength': 0.6
                }
            })
        elif weather == 'stormy':
            effects.update({
                'fog': {
                    'density': 0.005,
                    'height': 20,
                    'falloff': 0.8,
                    'scattering': 1.0,
                    'color': [0.4, 0.4, 0.5]
                },
                'clouds': {
                    'coverage': 1.0,
                    'thickness': 1.0,
                    'speed': 2.0,
                    'shadow_intensity': 1.0
                },
                'precipitation': {
                    'intensity': 1.0,
                    'type': 'heavy_rain',
                    'wind_influence': 1.0
                },
                'wind': {
                    'strength': 1.2,
                    'direction': [random.uniform(-1, 1), 0, random.uniform(-1, 1)],
                    'gustiness': 0.8
                },
                'wetness': {
                    'amount': 1.0,
                    'puddles': True,
                    'reflection_strength': 0.8
                },
                'lightning': {
                    'enabled': True,
                    'frequency': 0.1,
                    'intensity': 1.0
                }
            })
        elif weather == 'snowy':
            effects.update({
                'fog': {
                    'density': 0.002,
                    'height': 100,
                    'falloff': 0.3,
                    'scattering': 0.7,
                    'color': [0.9, 0.9, 1.0]
                },
                'clouds': {
                    'coverage': 0.8,
                    'thickness': 0.6,
                    'speed': 0.3,
                    'shadow_intensity': 0.6
                },
                'precipitation': {
                    'intensity': 0.6,
                    'type': 'snow',
                    'wind_influence': 0.3
                },
                'wind': {
                    'strength': 0.4,
                    'direction': [0.5, 0, 0.5],
                    'gustiness': 0.2
                },
                'wetness': {
                    'amount': 0.0,
                    'puddles': False,
                    'reflection_strength': 0.0
                },
                'snow_accumulation': {
                    'enabled': True,
                    'rate': 0.1,
                    'max_depth': 0.5
                }
            })
        
        return effects
    
    def apply_dynamic_resolution(self, target_fps=60):
        """Apply dynamic resolution scaling to maintain target FPS."""
        current_fps = self.performance_stats['fps']
        
        if current_fps < target_fps * 0.9:
            # Reduce resolution
            scale_factor = 0.9
            print(f"Reducing resolution scale to {scale_factor}")
        elif current_fps > target_fps * 1.1:
            # Increase resolution
            scale_factor = min(1.0, self.quality_settings.render_distance * 1.1)
            print(f"Increasing resolution scale to {scale_factor}")
        else:
            scale_factor = 1.0
        
        return {
            'scale_factor': scale_factor,
            'target_fps': target_fps,
            'current_fps': current_fps
        }
    
    def generate_debug_visualization(self):
        """Generate debug visualization data."""
        return {
            'wireframe': False,
            'show_normals': False,
            'show_bounding_boxes': True,
            'show_frustum': True,
            'show_light_sources': True,
            'show_collision_volumes': False,
            'performance_overlay': True,
            'statistics': {
                'frame_time': self.performance_stats['cpu_time'],
                'fps': self.performance_stats['fps'],
                'triangle_count': self.performance_stats['triangle_count'],
                'draw_calls': 0,  # Would be calculated
                'memory_usage': f"{self.performance_stats['memory_usage_mb']:.1f} MB",
                'gpu_memory': "512 MB",
                'vram_usage': "75%"
            }
        }

class GraphicsEngine:
    """Main graphics engine class."""
    
    def __init__(self):
        self.graphics_processor = AdvancedGraphicsProcessor()
        self.weather_system = WeatherSystem ()
        self.current_frame = 0
        self.is_paused = False
        self.debug_mode = False
        
        print("Advanced Graphics Engine initialized!")
        print(f"Terrain size: {self.graphics_processor.terrain_generator.size}")
        print(f"Resolution: {self.graphics_processor.terrain_generator.resolution}")
    
    def render_frame(self, game_state):
        """Render a complete frame."""
        if self.is_paused:
            return self.get_last_frame()
        
        self.current_frame += 1
        
        # Update weather
        self.weather_system.update(game_state.get('delta_time', 0.016))
        weather_data = self.weather_system.get_weather_data()
        
        # Prepare frame data
        frame_data = {
            'player_position': game_state.get('player_position', {'x': 0, 'y': 0, 'z': 0}),
            'camera_position': game_state.get('camera_position', {'x': 0, 'y': 15, 'z': 25}),
            'camera_rotation': game_state.get('camera_rotation', {'x': -20, 'y': 0, 'z': 0}),
            'time_of_day': game_state.get('time_of_day', 12.0),
            'weather': weather_data['type'],
            'speed': game_state.get('speed', 0),
            'delta_time': game_state.get('delta_time', 0.016),
            'performance_metrics': game_state.get('performance_metrics', {})
        }
        
        # Process frame
        result = self.graphics_processor.process_frame(frame_data)
        
        # Add weather data
        result['weather_data'] = weather_data
        
        # Add debug info if enabled
        if self.debug_mode:
            result['debug'] = self.graphics_processor.generate_debug_visualization()
        
        return result
    
    def get_last_frame(self):
        """Get the last rendered frame (when paused)."""
        return {
            'frame_number': self.current_frame,
            'paused': True,
            'message': 'Graphics engine is paused'
        }
    
    def toggle_pause(self):
        """Toggle pause state."""
        self.is_paused = not self.is_paused
        return self.is_paused
    
    def toggle_debug(self):
        """Toggle debug mode."""
        self.debug_mode = not self.debug_mode
        self.graphics_processor.debug_mode = self.debug_mode
        return self.debug_mode
    
    def set_graphics_settings(self, settings):
        """Update graphics settings."""
        for key, value in settings.items():
            if hasattr(self.graphics_processor.quality_settings, key):
                setattr(self.graphics_processor.quality_settings, key, value)
        
        return self.graphics_processor.quality_settings.__dict__

# Test the enhanced graphics engine
if __name__ == "__main__":
    print("Testing Advanced Graphics Engine...")
    
    engine = GraphicsEngine()
    
    # Test different scenarios
    test_scenarios = [
        {
            'name': 'Daytime Clear',
            'player_position': {'x': 0, 'y': 0, 'z': 0},
            'camera_position': {'x': 0, 'y': 20, 'z': 40},
            'time_of_day': 14.0,
            'speed': 60
        },
        {
            'name': 'Evening Storm',
            'player_position': {'x': 100, 'y': 0, 'z': -50},
            'camera_position': {'x': 100, 'y': 25, 'z': -25},
            'time_of_day': 19.0,
            'weather': 'stormy',
            'speed': 120
        },
        {
            'name': 'Night Snow',
            'player_position': {'x': -200, 'y': 0, 'z': 100},
            'camera_position': {'x': -200, 'y': 30, 'z': 125},
            'time_of_day': 22.0,
            'weather': 'snowy',
            'speed': 30
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting: {scenario['name']}")
        result = engine.render_frame(scenario)
        
        print(f"Frame: {result['frame_number']}")
        print(f"Processing time: {result['processing_time']:.2f}ms")
        print(f"Terrain triangles: {result['optimization']['triangle_count']}")
        print(f"Draw calls: {result['optimization']['draw_calls']}")
        print(f"Trees: {len(result['vegetation']['trees'])}")
        print(f"Grass patches: {len(result['vegetation']['grass'])}")
        
        # Test dynamic resolution
        if scenario['name'] == 'Evening Storm':
            dyn_res = engine.graphics_processor.apply_dynamic_resolution()
            print(f"Dynamic resolution: {dyn_res['scale_factor']:.2f}")
    
    print("\nGraphics engine test complete!")