import cv2
import numpy as np
import os
from hand_tracker import HandTracker
from canvas import DrawingCanvas


def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ret, frame = cap.read()
    if not ret:
        print("Failed to access webcam")
        return
    
    height, width = frame.shape[:2]
    hand_tracker = HandTracker()
    canvas = DrawingCanvas(width, height)
    
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (255, 255, 255) # White
    ]
    color_index = 0
    canvas.set_color(colors[color_index])
    
    is_drawing = False
    show_help = True
    drawing_started = False
    single_finger_mode = True  # Default to single finger drawing
    mode_indicator = "FINGER"  # Initialize status indicator
    os.makedirs("saves", exist_ok=True)
    
    print("Air Doodle Started!")
    print("Controls:")
    print("  - EXTEND INDEX FINGER ONLY to draw (single finger mode)")
    print("  - Or pinch thumb + index to draw (toggle with 'p')")
    print("  - Press 'c' to change color")
    print("  - Press '+' to increase stroke width")
    print("  - Press '-' to decrease stroke width")
    print("  - Press 'e' to toggle eraser mode")
    print("  - Press 'p' to toggle pinch/finger drawing mode")
    print("  - Press 'z' to undo, 'y' to redo")
    print("  - Press '1' circle, '2' rectangle, '3' line (shape tools)")
    print("  - Press '0' to exit shape mode")
    print("  - Press 's' to save drawing")
    print("  - Press 'l' to load drawing")
    print("  - Press 'x' to clear canvas")
    print("  - Press 'h' to toggle help")
    print("  - Press 'q' or ESC to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        frame = hand_tracker.find_hands(frame, draw=True)
        landmark_positions = hand_tracker.get_landmark_positions(frame)
        
        if landmark_positions:
            landmarks = landmark_positions[0]
            index_tip = hand_tracker.get_index_finger_tip(landmarks)
            
            # Determine drawing trigger based on mode
            if single_finger_mode:
                draw_trigger = hand_tracker.is_index_finger_extended(landmarks)
                mode_indicator = "FINGER"
            else:
                draw_trigger = hand_tracker.is_pinching(landmarks)
                mode_indicator = "PINCH"
            
            if draw_trigger:
                is_drawing = True
                if canvas.shape_tool:
                    if not drawing_started:
                        canvas.start_shape(index_tip)
                        drawing_started = True
                    else:
                        preview_canvas = canvas.draw_shape(index_tip, preview=True)
                        if preview_canvas is not None:
                            display_frame = cv2.addWeighted(frame, 1, preview_canvas, 0.5, 0)
                else:
                    if not drawing_started:
                        canvas.save_state()
                        drawing_started = True
                    canvas.draw_line(index_tip)
                
                cursor_color = (0, 0, 0) if canvas.mode == "eraser" else (0, 255, 255)
                cv2.circle(frame, index_tip, 10, cursor_color, -1)
            else:
                if drawing_started and canvas.shape_tool:
                    canvas.draw_shape(index_tip, preview=False)
                drawing_started = False
                if is_drawing:
                    canvas.reset_prev_point()
                is_drawing = False
            
            if index_tip:
                brush_color = (128, 128, 128) if canvas.mode == "eraser" else canvas.drawing_color
                cv2.circle(frame, index_tip, 8, brush_color, 2)
        else:
            drawing_started = False
            if is_drawing:
                canvas.reset_prev_point()
            is_drawing = False
        
        if canvas.shape_tool and canvas.shape_start and index_tip:
            preview_canvas = canvas.draw_shape(index_tip, preview=True)
            if preview_canvas is not None:
                display_frame = cv2.addWeighted(frame, 1, preview_canvas, 0.5, 0)
        else:
            display_frame = canvas.overlay_on_frame(frame)
        
        if show_help:
            mode_str = "ERASER" if canvas.mode == "eraser" else "DRAW"
            shape_str = canvas.shape_tool.upper() if canvas.shape_tool else "FREE"
            trigger_str = "FINGER" if single_finger_mode else "PINCH"
            help_texts = [
                "Air Doodle Controls:",
                f"{trigger_str}: draw | 'c' color | +/- width | 'e' eraser | 'p' mode",
                "'z' undo | 'y' redo | '1' circle | '2' rect | '3' line | '0' free",
                "'s' save | 'l' load | 'x' clear | 'h' hide | 'q' quit",
                f"Mode: {mode_str} | Tool: {shape_str} | Color: {color_index+1}/{len(colors)} | Width: {canvas.stroke_width}"
            ]
            y_offset = 30
            for i, text in enumerate(help_texts):
                color = (255, 255, 255) if i == 0 else (200, 200, 200)
                cv2.putText(display_frame, text, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        status_text = f"{mode_indicator} DRAWING" if is_drawing else f"{mode_indicator} READY"
        status_color = (0, 255, 0) if is_drawing else (0, 165, 255)
        if canvas.mode == "eraser":
            status_text = f"{mode_indicator} ERASING" if is_drawing else "ERASER READY"
            status_color = (128, 128, 128) if is_drawing else (64, 64, 64)
        cv2.putText(display_frame, status_text, (width - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        
        cv2.imshow("Air Doodle", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('c'):
            color_index = (color_index + 1) % len(colors)
            canvas.set_color(colors[color_index])
        elif key == ord('+') or key == ord('='):
            canvas.set_stroke_width(min(canvas.stroke_width + 2, 20))
        elif key == ord('-') or key == ord('_'):
            canvas.set_stroke_width(max(canvas.stroke_width - 2, 2))
        elif key == ord('x'):
            canvas.clear_canvas()
        elif key == ord('h'):
            show_help = not show_help
        elif key == ord('e'):
            new_mode = canvas.toggle_mode()
            print(f"Mode: {new_mode.upper()}")
        elif key == ord('z'):
            if canvas.undo():
                print("Undo")
        elif key == ord('y'):
            if canvas.redo():
                print("Redo")
        elif key == ord('1'):
            canvas.set_shape_tool("circle")
            print("Shape: Circle")
        elif key == ord('2'):
            canvas.set_shape_tool("rectangle")
            print("Shape: Rectangle")
        elif key == ord('3'):
            canvas.set_shape_tool("line")
            print("Shape: Line")
        elif key == ord('0'):
            canvas.set_shape_tool(None)
            print("Shape: Free draw")
        elif key == ord('s'):
            filepath = canvas.save_drawing("saves/doodle.png")
            print(f"Saved to: {filepath}")
        elif key == ord('l'):
            if canvas.load_drawing("saves/doodle.png"):
                print("Loaded saves/doodle.png")
            else:
                print("No saved drawing found")
        elif key == ord('p'):
            single_finger_mode = not single_finger_mode
            mode_name = "FINGER" if single_finger_mode else "PINCH"
            print(f"Drawing mode: {mode_name}")
    
    hand_tracker.release()
    cap.release()
    cv2.destroyAllWindows()
    print("Air Doodle closed.")


if __name__ == "__main__":
    main()
