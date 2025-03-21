# Strike AI Pose 🎯

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.24%2B-FF4B4B.svg)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-brightgreen.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A real-time pose matching game that uses AI to detect and match poses. Challenge yourself to match reference poses and compete for high scores!

![Game Demo](static/demo.gif)

## 🌟 Features

- 🎮 Real-time pose detection and matching
- 📊 Live similarity scoring
- 🏆 Leaderboard system
- ⚙️ Configurable game settings
- 🎨 Clean, modern UI
- 🔄 Support for custom reference poses

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone git@gitlab.com:latentai/showcases/gtc_2025_pose_demo.git
cd gtc_2025_pose_demo
```

2. Create and activate a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the game:
```bash
python3 -m streamlit run app.py
```

## 🎮 How to Play

1. Enter your name and email to begin
2. Match the pose shown in the reference image
3. Hold the pose for 1 second to score
4. Match as many poses as you can in 50 seconds!
5. Try to beat your high score!

## 🛠️ Configuration

The game can be configured through `settings.json`:

```json
{
    "game": {
        "duration_seconds": 50,
        "pose_hold_time_seconds": 1
    },
    "pose_matcher": {
        "similarity_threshold": 0.85
    },
    "pose_detector": {
        "confidence_threshold": 0.5
    }
}
```

## 📁 Project Structure

```
pose_game/
├── app.py              # Main Streamlit application
├── pose_detector.py    # YOLOv8 pose detection
├── pose_matcher.py     # Pose similarity computation
├── depth_detector.py   # Depth detection utilities
├── settings.json       # Game configuration
├── requirements.txt    # Project dependencies
└── static/
    ├── poses2/         # Reference pose images
    └── logos/         # UI assets
```

## 🔧 Requirements

- Python 3.8 or higher
- Webcam
- Modern web browser
- See `requirements.txt` for Python package dependencies

## 💡 Tips for Best Results

1. **Lighting**: Ensure good, even lighting
2. **Distance**: Stand 6-8 feet from camera
3. **Background**: Use a plain, contrasting background
4. **Clothing**: Wear fitted clothing that contrasts with background
5. **Space**: Ensure enough room to move freely

## 🔍 Troubleshooting

### Common Issues

1. **Webcam Not Detected**
   - Check webcam permissions
   - Ensure no other app is using the webcam
   - Try restarting the application

2. **Poor Pose Detection**
   - Improve lighting conditions
   - Adjust distance from camera
   - Ensure clear contrast with background

3. **Performance Issues**
   - Close unnecessary applications
   - Check system meets minimum requirements
   - Ensure adequate GPU/CPU resources

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [Streamlit](https://streamlit.io/) for the web framework
- [OpenCV](https://opencv.org/) for image processing

---

Made with ❤️ by LatentAI
# spotit
