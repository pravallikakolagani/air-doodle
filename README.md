✏️ Air Doodle

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-FF6F00?style=for-the-badge" alt="MediaPipe">
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License">
</p><h3 align="center">🖐️ Draw in the Air. Create Without Touch.</h3><p align="center">
  <strong>No touchscreen. No stylus. No mouse. Just your hand.</strong>
</p><p align="center">
  <a href="https://air-doodle-topaz.vercel.app/">
    <strong>🚀 Try Air Doodle Live →</strong>
  </a>
</p>---

🌐 Live Demo

🚀 "Open Air Doodle" (https://air-doodle-topaz.vercel.app/)

Experience the Air Doodle interface directly in your browser.

GitHub:
https://github.com/pravallikakolagani/air-doodle

---

🎨 What is Air Doodle?

Air Doodle is a touchless digital drawing experience that lets you draw using your index finger as a virtual brush.

Instead of touching a screen or using a physical stylus, Air Doodle uses computer vision and hand tracking to translate your hand movements into digital strokes.

        🖐️ YOUR HAND
              │
              ▼
        📷 CAMERA
              │
              ▼
     🤖 HAND TRACKING
              │
              ▼
      ☝️ INDEX FINGER
              │
              ▼
       🎨 DRAWING ENGINE
              │
              ▼
       🖼️ DIGITAL CANVAS

«Your hand is the stylus.
The air is the canvas.»

---

✨ Features

Feature| Description
🖐️ Hand Tracking| Track hand movement using computer vision
✏️ Air Drawing| Draw using your index finger
🎨 Smart Brushes| Multiple brush styles for creative drawing
🌈 Rainbow Mode| Create dynamic rainbow-colored strokes
🪞 Symmetry Mode| Create mirrored artwork
🧹 Eraser| Remove unwanted strokes
↩️ Undo / Redo| Easily correct your artwork
🔷 Shape Tools| Create predefined shapes
🎬 Recording| Record drawing sessions
⏱️ Timelapse| Replay the creative process
💾 Save / Load| Save and restore drawings
🎨 Backgrounds| Change the drawing environment

---

🧠 How It Works

Air Doodle converts real-world hand movement into digital artwork through a computer-vision pipeline.

1. 📷 Capture

The camera captures the user's hand in real time.

2. 🖐️ Detect

The hand-tracking system identifies the hand and its landmarks.

3. ☝️ Track

The application identifies the index fingertip and continuously tracks its position.

4. 📍 Map

The fingertip coordinates are converted into canvas coordinates.

5. 🎨 Render

The drawing engine converts the movement into strokes based on the selected brush.

6. 🖼️ Display

The resulting artwork is rendered on the digital canvas.

---

🏗️ Architecture

┌─────────────────────┐
│       Webcam        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    OpenCV Input     │
│   Frame Processing  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Hand Tracking     │
│      MediaPipe      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Landmark Extraction │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Index Finger        │
│ Position Detection  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Drawing Engine    │
│ Brushes / Shapes    │
│ Colors / Symmetry   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│    Digital Canvas   │
└─────────────────────┘

---

🎮 Controls

Key / Gesture| Action
☝️ Index Finger| Draw
"C"| Change color
"+"| Increase stroke width
"-"| Decrease stroke width
"E"| Eraser mode
"B"| Change background
"V"| Change brush type
"M"| Toggle symmetry
"N"| Rainbow mode
"R"| Start / Stop recording
"T"| Timelapse
"Z"| Undo
"Y"| Redo
"1"| Shape 1
"2"| Shape 2
"3"| Shape 3
"S"| Save drawing
"L"| Load drawing

---

🖌️ Brush System

Air Doodle provides multiple creative drawing modes.

🖊️ Solid Brush

Clean and direct strokes for normal drawing.

💨 Spray Brush

Creates a spray-paint style effect.

🌈 Rainbow Mode

Automatically produces colorful strokes while drawing.

🪞 Symmetry Mode

Mirrors your movement to create symmetrical artwork.

🧹 Eraser

Remove unwanted parts of your drawing.

---

🎬 Record & Replay

Air Doodle can capture the drawing process so users can revisit their creative sessions.

This can be useful for:

- 🎨 Digital art demonstrations
- 🎥 Content creation
- 📚 Tutorials
- 🧑‍🏫 Teaching
- 🏆 Project demonstrations

---

↩️ Undo & Redo

Experiment freely without worrying about mistakes.

Z → Undo
Y → Redo

---

🛠️ Tech Stack

Core Technologies

- 🐍 Python
- 👁️ OpenCV
- 🖐️ MediaPipe

Components

Python
  │
  ├── OpenCV
  │     └── Camera + Image Processing
  │
  ├── MediaPipe
  │     └── Hand Tracking
  │
  └── Drawing Engine
        ├── Brushes
        ├── Shapes
        ├── Colors
        ├── Symmetry
        └── Canvas

---

📁 Project Structure

air-doodle/
│
├── api/
│   └── ...
│
├── air_doodle.py
├── app.py
├── canvas.py
├── hand_tracker.py
├── playback.py
│
├── hand_landmarker.task
├── index.html
│
├── requirements.txt
├── vercel.json
├── .vercelignore
│
└── README.md

---

🚀 Quick Start

Prerequisites

Make sure you have:

- Python 3.x
- A working webcam
- Git
- pip

---

1️⃣ Clone the Repository

git clone https://github.com/pravallikakolagani/air-doodle.git

cd air-doodle

---

2️⃣ Install Dependencies

pip install -r requirements.txt

---

3️⃣ Run Air Doodle

python air_doodle.py

Allow camera access when requested.

---

🌐 Web Version

The project also has a deployed web interface:

🚀 "air-doodle-topaz.vercel.app" (https://air-doodle-topaz.vercel.app/)

The repository includes web/deployment components such as:

app.py
index.html
api/
vercel.json

---

🎯 Use Cases

Air Doodle can be used as more than a drawing application.

🎨 Digital Art

Create artwork without touching a physical surface.

🏫 Education

Draw diagrams and explain concepts interactively.

🎤 Presentations

Explore gesture-based interaction for presentations.

♿ Accessibility

Touchless interaction can provide an alternative input method for some users.

🎮 Interactive Applications

Use hand tracking as the foundation for gesture-controlled experiences.

🥽 AR / VR

The same interaction concept can be extended into immersive environments.

---

💡 Innovation

Traditional digital drawing:

Hand → Mouse / Stylus → Screen → Drawing

Air Doodle:

Hand → Camera → AI Hand Tracking → Drawing

The physical controller disappears.

Instead of interacting with the screen, the user interacts through movement.

---

🔮 Future Roadmap

🎨 Creative

- [ ] Custom brush creation
- [ ] More brush effects
- [ ] Advanced geometric shapes
- [ ] PNG export
- [ ] SVG export
- [ ] Advanced color palette

🧠 AI

- [ ] AI drawing recognition
- [ ] Gesture recognition
- [ ] AI-assisted drawing
- [ ] Smart shape correction
- [ ] Voice commands

🖐️ Interaction

- [ ] Multi-hand support
- [ ] Gesture-based menus
- [ ] Custom gestures
- [ ] Touchless UI navigation

🌎 Platform

- [ ] Mobile support
- [ ] AR integration
- [ ] VR integration
- [ ] Collaborative drawing
- [ ] Cloud-based artwork sharing

---

🧪 Future Vision

Air Doodle could evolve from a drawing application into a general-purpose gesture interaction platform.

Imagine:

             🖐️
              │
              ▼
       Gesture Detection
              │
              ▼
       AI Interpretation
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
     Draw   Control  Create
       │      │      │
       ▼      ▼      ▼
     Canvas   UI     AI

The same technology could potentially power:

- Touchless interfaces
- Smart classrooms
- Interactive installations
- Presentation systems
- Creative AI tools
- AR/VR experiences
- Accessibility-focused interfaces

---

🤝 Contributing

Contributions are welcome!

Fork the repository

git clone https://github.com/pravallikakolagani/air-doodle.git

Create a branch

git checkout -b feature/amazing-feature

Make your changes

git add .

Commit

git commit -m "Add amazing feature"

Push

git push origin feature/amazing-feature

Then open a Pull Request.

---

🐛 Issues & Suggestions

Found a bug or have an idea?

Open an issue in the repository:

👉 https://github.com/pravallikakolagani/air-doodle/issues

Ideas, improvements, and creative experiments are always welcome.

---

⭐ Support the Project

If you like Air Doodle:

⭐ Star the repository
🍴 Fork it
🐛 Report bugs
💡 Suggest features
🤝 Contribute

Every star and contribution helps the project grow.

---

👩‍💻 Creator

<p align="center">Made with 💚 by <strong>@pravallikakolagani</strong>

Built with
<strong>Python · OpenCV · MediaPipe · Computer Vision</strong>

</p>---

<p align="center">✨ Draw Beyond the Screen.

🖐️ Your Hand Is the Stylus.

🌌 The Air Is the Canvas.

<br>🚀 <a href="https://air-doodle-topaz.vercel.app/"><strong>TRY AIR DOODLE →</strong></a>

</p>---

<p align="center">
  <sub>© 2026 Air Doodle · Built for creativity, experimentation & innovation.</sub>
</p>
