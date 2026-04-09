import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp
from mediapipe.tasks.python import vision
import os

# Page config
st.set_page_config(
    page_title="Air Doodle - Web",
    page_icon="✏️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #1a1a2e;
    }
    .stButton button {
        background-color: #4a4a8a;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
    }
    .stButton button:hover {
        background-color: #6a6aaa;
    }
    h1 {
        color: #fff;
        text-align: center;
    }
    .control-panel {
        background-color: #16213e;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

class HandTrackerWeb:
    def __init__(self):
        self.model_path = self._get_model_path()
        base_options = mp.tasks.BaseOptions(model_asset_path=self.model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
    
    def _get_model_path(self):
        model_path = "hand_landmarker.task"
        if not os.path.exists(model_path):
            import urllib.request
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            st.info("Downloading hand tracking model...")
            urllib.request.urlretrieve(url, model_path)
        return model_path
    
    def process_frame(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        
        landmarks = []
        if results.hand_landmarks:
            h, w = frame.shape[:2]
            for hand_landmarks in results.hand_landmarks:
                pts = {}
                for id, lm in enumerate(hand_landmarks):
                    pts[id] = (int(lm.x * w), int(lm.y * h))
                landmarks.append(pts)
        
        return landmarks

class DrawingCanvasWeb:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.canvas = np.zeros((height, width, 3), dtype=np.uint8)
        self.prev_point = None
        self.drawing_color = (0, 255, 0)
        self.stroke_width = 5
        self.rainbow_hue = 0
        self.point_buffer = []
        self.buffer_size = 10
    
    def _smooth_point(self, point):
        self.point_buffer.append(point)
        if len(self.point_buffer) > self.buffer_size:
            self.point_buffer.pop(0)
        
        if len(self.point_buffer) < 3:
            return point
        
        avg_x = sum(p[0] for p in self.point_buffer) / len(self.point_buffer)
        avg_y = sum(p[1] for p in self.point_buffer) / len(self.point_buffer)
        return (int(avg_x), int(avg_y))
    
    def draw_line(self, point, color=None, width=None, rainbow_mode=False):
        if point is None:
            return
        
        if rainbow_mode:
            import colorsys
            self.rainbow_hue = (self.rainbow_hue + 5) % 360
            rgb = colorsys.hsv_to_rgb(self.rainbow_hue / 360, 1.0, 1.0)
            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
        else:
            color = color or self.drawing_color
        
        width = width or self.stroke_width
        smoothed = self._smooth_point(point)
        
        if self.prev_point is not None:
            cv2.line(self.canvas, self.prev_point, smoothed, color, width, cv2.LINE_AA)
        
        self.prev_point = smoothed
    
    def reset(self):
        self.prev_point = None
        self.point_buffer.clear()
    
    def clear(self):
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.reset()
    
    def get_image(self):
        return self.canvas

def main():
    st.title("✏️ Air Doodle - Web Version")
    st.markdown("*Draw in the air with your finger!*")
    
    # Initialize session state
    if 'canvas' not in st.session_state:
        st.session_state.canvas = DrawingCanvasWeb(640, 480)
    if 'tracker' not in st.session_state:
        st.session_state.tracker = HandTrackerWeb()
    if 'is_drawing' not in st.session_state:
        st.session_state.is_drawing = False
    
    # Sidebar controls
    with st.sidebar:
        st.header("🎨 Controls")
        
        # Color picker
        color_hex = st.color_picker("Drawing Color", "#00FF00")
        rgb = tuple(int(color_hex[i:i+2], 16) for i in (5, 3, 1))  # Convert to BGR
        st.session_state.canvas.drawing_color = rgb
        
        # Stroke width
        st.session_state.canvas.stroke_width = st.slider("Stroke Width", 1, 20, 5)
        
        # Modes
        rainbow_mode = st.toggle("🌈 Rainbow Mode", False)
        spray_mode = st.toggle("💨 Spray Mode", False)
        
        st.divider()
        
        # Actions
        if st.button("🗑️ Clear Canvas", use_container_width=True):
            st.session_state.canvas.clear()
            st.rerun()
        
        if st.button("💾 Save Drawing", use_container_width=True):
            img = Image.fromarray(cv2.cvtColor(st.session_state.canvas.get_image(), cv2.COLOR_BGR2RGB))
            img.save("doodle.png")
            st.success("Saved to doodle.png!")
        
        st.divider()
        st.markdown("### Instructions")
        st.markdown("""
        1. Upload an image or use the demo
        2. Your hand tracking data will be processed
        3. Draw with your index finger
        """)
    
    # Main area - Webcam or file upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📷 Input")
        
        # Option for file upload (since we can't easily access webcam in browser)
        uploaded_file = st.file_uploader("Upload image/video", type=['jpg', 'jpeg', 'png', 'mp4', 'avi'])
        
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if frame is not None:
                st.session_state.last_frame = frame
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                # Process frame for hand tracking
                landmarks_list = st.session_state.tracker.process_frame(frame)
                
                if landmarks_list:
                    landmarks = landmarks_list[0]
                    
                    # Check if index finger is extended
                    if 8 in landmarks and 6 in landmarks:
                        index_tip = landmarks[8]
                        index_pip = landmarks[6]
                        
                        # Simple check: is index finger up?
                        if index_tip[1] < index_pip[1]:
                            st.session_state.is_drawing = True
                            
                            # Draw with spray or solid
                            if spray_mode:
                                # Simple spray effect
                                for _ in range(20):
                                    angle = np.random.uniform(0, 2 * np.pi)
                                    radius = np.random.uniform(0, 15)
                                    x = int(index_tip[0] + radius * np.cos(angle))
                                    y = int(index_tip[1] + radius * np.sin(angle))
                                    if 0 <= x < 640 and 0 <= y < 480:
                                        color = st.session_state.canvas.drawing_color
                                        if rainbow_mode:
                                            import colorsys
                                            st.session_state.canvas.rainbow_hue = (st.session_state.canvas.rainbow_hue + 5) % 360
                                            rgb = colorsys.hsv_to_rgb(st.session_state.canvas.rainbow_hue / 360, 1.0, 1.0)
                                            color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
                                        cv2.circle(st.session_state.canvas.canvas, (x, y), 1, color, -1)
                            else:
                                st.session_state.canvas.draw_line(
                                    index_tip, 
                                    rainbow_mode=rainbow_mode
                                )
                            
                            # Draw cursor
                            cv2.circle(frame, index_tip, 8, (0, 255, 255), 2)
                        else:
                            st.session_state.canvas.reset()
                        
                        # Draw landmarks
                        for idx, pt in landmarks.items():
                            cv2.circle(frame, pt, 3, (0, 255, 0), -1)
                else:
                    st.session_state.canvas.reset()
                
                st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), caption="Tracked", use_container_width=True)
    
    with col2:
        st.subheader("🎨 Canvas")
        
        # Combine canvas with optional background
        canvas_img = st.session_state.canvas.get_image()
        
        # Display canvas
        display_img = Image.fromarray(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB))
        st.image(display_img, use_container_width=True)
        
        # Status
        if st.session_state.is_drawing:
            st.success("✏️ Drawing...")
        else:
            st.info("👆 Show hand to draw")
    
    # Demo section
    with st.expander("📚 About Air Doodle Web"):
        st.markdown("""
        This is the web version of Air Doodle. Due to browser security restrictions,
        real-time webcam access requires additional setup.
        
        **Current Features:**
        - Upload images/videos for hand tracking
        - Draw with index finger detection
        - Rainbow color mode
        - Spray paint effect
        - Save drawings
        
        **Desktop App Features:**
        - Real-time webcam drawing
        - Video recording & playback
        - Symmetry mode
        - All brush effects
        """)

if __name__ == "__main__":
    main()
