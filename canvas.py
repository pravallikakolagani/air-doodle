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
                
                # Use LINE_AA for anti-aliased smoother lines
                cv2.line(self.canvas, self.prev_point, smoothed, color, width, cv2.LINE_AA)
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
    
    def set_stroke_width(self, width):
        self.stroke_width = width
    
    def get_canvas(self):
        return self.canvas.copy()
    
    def overlay_on_frame(self, frame, alpha=0.5):
        return cv2.addWeighted(frame, 1, self.canvas, alpha, 0)
    
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
