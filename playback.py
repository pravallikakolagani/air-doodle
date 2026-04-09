import cv2
import os
import sys


def play_recording(filepath, loop=False):
    """Play back a recorded video"""
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return
    
    cap = cv2.VideoCapture(filepath)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video - {filepath}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_delay = int(1000 / fps) if fps > 0 else 33
    
    print(f"Playing: {filepath}")
    print(f"FPS: {fps:.1f}")
    print("Controls: SPACE=pause, LEFT/RIGHT=seek, q=quit")
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                if loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    break
        
        # Get current frame info
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Add progress bar overlay
        display_frame = frame.copy()
        bar_width = 400
        bar_height = 10
        bar_x = (display_frame.shape[1] - bar_width) // 2
        bar_y = display_frame.shape[0] - 30
        
        # Background bar
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress
        progress = int((current_frame / total_frames) * bar_width) if total_frames > 0 else 0
        cv2.rectangle(display_frame, (bar_x, bar_y), (bar_x + progress, bar_y + bar_height), (0, 255, 0), -1)
        
        # Frame counter
        text = f"{current_frame}/{total_frames} | {'PAUSED' if paused else 'PLAYING'}"
        cv2.putText(display_frame, text, (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Air Doodle Playback", display_frame)
        
        wait_time = 0 if paused else frame_delay
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == 81:  # Left arrow
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, current_frame - 30))
            paused = False
        elif key == 83:  # Right arrow
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(total_frames, current_frame + 30))
            paused = False
    
    cap.release()
    cv2.destroyAllWindows()
    print("Playback finished.")


def list_recordings():
    """List all available recordings"""
    saves_dir = "saves"
    if not os.path.exists(saves_dir):
        print("No saves directory found.")
        return []
    
    recordings = [f for f in os.listdir(saves_dir) if f.endswith('.avi')]
    return recordings


def main():
    if len(sys.argv) > 1:
        # Play specific file
        filepath = sys.argv[1]
        loop = '--loop' in sys.argv
        play_recording(filepath, loop)
    else:
        # Show list and let user choose
        recordings = list_recordings()
        
        if not recordings:
            print("No recordings found in saves/ directory.")
            print("\nUsage: python playback.py <recording_file.avi>")
            print("       python playback.py --loop  (to loop)")
            return
        
        print("\nAvailable recordings:")
        for i, rec in enumerate(recordings, 1):
            print(f"  {i}. {rec}")
        
        try:
            choice = input("\nEnter number to play (or q to quit): ").strip()
            if choice.lower() == 'q':
                return
            
            idx = int(choice) - 1
            if 0 <= idx < len(recordings):
                filepath = os.path.join("saves", recordings[idx])
                play_recording(filepath)
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")


if __name__ == "__main__":
    main()
