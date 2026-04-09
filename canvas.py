import cv2
import numpy as np
from datetime import datetime


class DrawingCanvas:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.drawing_color = (0, 255, 0)
        self.stroke_width = 5
        self.prev_point = None
        
        self.undo_stack = []
        self.redo_stack = []
        self.max_history = 20
        
        self.mode = "draw"
        self.eraser_width = 20
        
        self.shape_tool = None
        self.shape_start = None
        self.shapes = []
        
        self.point_buffer = []
        self.buffer_size = 15
        self.smoothing_factor = 0.75
        self.min_draw_distance = 3
        
        # New features
        self.background_type = "blank"  # blank, grid, lined
        self.brush_type = "solid"  # solid, spray
        self.symmetry_mode = False
        self.symmetry_axis = "vertical"  # vertical or horizontal
        self.rainbow_mode = False
        self.rainbow_hue = 0
        self.spray_density = 50
        self.spray_radius = 20
        
    def save_state(self):
        if len(self.undo_stack) >= self.max_history:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.canvas.copy())
        self.redo_stack.clear()
    
    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(self.canvas.copy())
            self.canvas = self.undo_stack.pop()
            self.prev_point = None
            return True
        return False
    
    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(self.canvas.copy())
            self.canvas = self.redo_stack.pop()
            self.prev_point = None
            return True
        return False
    
    def _smooth_point(self, point):
        self.point_buffer.append(point)
        if len(self.point_buffer) > self.buffer_size:
            self.point_buffer.pop(0)
        
        if len(self.point_buffer) < 3:
            return point
        
        # Weighted moving average - more weight to recent points
        weights = [i + 1 for i in range(len(self.point_buffer))]
        total_weight = sum(weights)
        avg_x = sum(p[0] * w for p, w in zip(self.point_buffer, weights)) / total_weight
        avg_y = sum(p[1] * w for p, w in zip(self.point_buffer, weights)) / total_weight
        
        # Exponential smoothing with previous point
        if self.prev_point:
            smooth_x = self.smoothing_factor * avg_x + (1 - self.smoothing_factor) * self.prev_point[0]
            smooth_y = self.smoothing_factor * avg_y + (1 - self.smoothing_factor) * self.prev_point[1]
            return (int(smooth_x), int(smooth_y))
        
        return (int(avg_x), int(avg_y))
    
    def draw_line(self, point):
        if point is None:
            return
        
        smoothed = self._smooth_point(point)
        
        if self.prev_point is not None:
            # Distance check - only draw if moved enough (reduces clustering)
            distance = np.sqrt((smoothed[0] - self.prev_point[0])**2 + (smoothed[1] - self.prev_point[1])**2)
            
            if distance >= self.min_draw_distance:
                color = (0, 0, 0) if self.mode == "eraser" else self.drawing_color
                width = self.eraser_width if self.mode == "eraser" else self.stroke_width
                
                # Get color (rainbow mode overrides)
                if self.rainbow_mode and self.mode == "draw":
                    color = self._get_rainbow_color()
                
                # Draw based on brush type
                if self.brush_type == "spray":
                    self._draw_spray(smoothed, color)
                else:
                    cv2.line(self.canvas, self.prev_point, smoothed, color, width, cv2.LINE_AA)
                
                # Draw symmetry if enabled
                if self.symmetry_mode:
                    self._draw_symmetry(self.prev_point, smoothed, color, width)
                
                self.prev_point = smoothed
        else:
            self.prev_point = smoothed
    
    def reset_prev_point(self):
        self.prev_point = None
        self.point_buffer.clear()
    
    def clear_canvas(self):
        self.save_state()
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.prev_point = None
    
    def set_mode(self, mode):
        self.mode = mode
        self.prev_point = None
    
    def toggle_mode(self):
        self.mode = "eraser" if self.mode == "draw" else "draw"
        self.prev_point = None
        return self.mode
    
    def set_shape_tool(self, tool):
        self.shape_tool = tool
        self.shape_start = None
    
    def start_shape(self, point):
        self.save_state()
        self.shape_start = point
    
    def draw_shape(self, end_point, preview=False):
        if self.shape_start is None or self.shape_tool is None:
            return
        
        if preview:
            temp_canvas = self.canvas.copy()
        else:
            temp_canvas = self.canvas
        
        color = self.drawing_color
        
        if self.shape_tool == "circle":
            radius = int(np.sqrt((end_point[0] - self.shape_start[0])**2 + 
                               (end_point[1] - self.shape_start[1])**2))
            cv2.circle(temp_canvas, self.shape_start, radius, color, self.stroke_width)
        elif self.shape_tool == "rectangle":
            cv2.rectangle(temp_canvas, self.shape_start, end_point, color, self.stroke_width)
        elif self.shape_tool == "line":
            cv2.line(temp_canvas, self.shape_start, end_point, color, self.stroke_width)
        
        if not preview:
            self.shape_start = None
        
        return temp_canvas if preview else None
    
    def set_color(self, color):
        self.drawing_color = color
    
    def set_background(self, bg_type):
        """Set background: blank, grid, or lined"""
        self.background_type = bg_type
    
    def set_brush_type(self, brush):
        """Set brush: solid or spray"""
        self.brush_type = brush
    
    def toggle_symmetry(self):
        """Toggle symmetry mode on/off"""
        self.symmetry_mode = not self.symmetry_mode
        return self.symmetry_mode
    
    def toggle_rainbow_mode(self):
        """Toggle rainbow color cycling"""
        self.rainbow_mode = not self.rainbow_mode
        return self.rainbow_mode
    
    def _get_rainbow_color(self):
        """Get next color in rainbow cycle"""
        import colorsys
        self.rainbow_hue = (self.rainbow_hue + 5) % 360
        rgb = colorsys.hsv_to_rgb(self.rainbow_hue / 360, 1.0, 1.0)
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))  # BGR format
    
    def set_stroke_width(self, width):
        self.stroke_width = width
    
    def get_canvas(self):
        return self.canvas.copy()
    
    def _draw_spray(self, point, color):
        """Draw spray paint effect at point"""
        for _ in range(self.spray_density):
            # Random point within spray radius
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, self.spray_radius)
            x = int(point[0] + radius * np.cos(angle))
            y = int(point[1] + radius * np.sin(angle))
            
            # Check bounds
            if 0 <= x < self.width and 0 <= y < self.height:
                # Vary opacity based on distance from center
                distance_ratio = radius / self.spray_radius
                if np.random.random() > distance_ratio * 0.5:  # More dots near center
                    cv2.circle(self.canvas, (x, y), 1, color, -1, cv2.LINE_AA)
    
    def _draw_symmetry(self, p1, p2, color, width):
        """Draw mirrored line based on symmetry axis"""
        if self.symmetry_axis == "vertical":
            # Mirror across vertical center
            center_x = self.width // 2
            p1_mirror = (2 * center_x - p1[0], p1[1])
            p2_mirror = (2 * center_x - p2[0], p2[1])
        else:
            # Mirror across horizontal center
            center_y = self.height // 2
            p1_mirror = (p1[0], 2 * center_y - p1[1])
            p2_mirror = (p2[0], 2 * center_y - p2[1])
        
        if self.brush_type == "spray":
            self._draw_spray(p2_mirror, color)
        else:
            cv2.line(self.canvas, p1_mirror, p2_mirror, color, width, cv2.LINE_AA)
    
    def get_background(self):
        """Generate background image based on type"""
        bg = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if self.background_type == "grid":
            # Draw grid
            grid_color = (40, 40, 40)
            grid_spacing = 50
            for x in range(0, self.width, grid_spacing):
                cv2.line(bg, (x, 0), (x, self.height), grid_color, 1)
            for y in range(0, self.height, grid_spacing):
                cv2.line(bg, (0, y), (self.width, y), grid_color, 1)
        
        elif self.background_type == "lined":
            # Draw lined paper
            line_color = (50, 50, 80)
            line_spacing = 40
            for y in range(80, self.height, line_spacing):
                cv2.line(bg, (0, y), (self.width, y), line_color, 1)
            # Add margin line
            cv2.line(bg, (60, 0), (60, self.height), (60, 60, 100), 2)
        
        return bg
    
    def overlay_on_frame(self, frame, alpha=0.5):
        # Apply background
        background = self.get_background()
        # Combine background with canvas
        combined = cv2.addWeighted(background, 0.3, self.canvas, 1, 0)
        return cv2.addWeighted(frame, 1, combined, alpha, 0)
    
    def save_drawing(self, filepath=None):
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"air_doodle_{timestamp}.png"
        cv2.imwrite(filepath, self.canvas)
        return filepath
    
    def load_drawing(self, filepath):
        loaded = cv2.imread(filepath)
        if loaded is not None:
            self.save_state()
            self.canvas = cv2.resize(loaded, (self.width, self.height))
            return True
        return False
