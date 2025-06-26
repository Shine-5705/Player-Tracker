# Football Player Re-Identification System ⚽

A robust computer vision system for tracking and re-identifying football players across video frames, designed to maintain consistent player identities even when players temporarily disappear from view.

## 🎥 Demo Video

Check out our system in action! The output video demonstrates successful player tracking and re-identification:

https://github.com/Shine-5705/Player-Tracker/blob/main/distance_based_tracked_video.mp4

*The demo shows players maintaining consistent IDs even when moving out of frame and reappearing during the goal event.*

## 🎯 Project Overview

This project implements a sophisticated player tracking system that combines object detection with advanced re-identification techniques. The system is specifically designed to handle the challenges of football player tracking, including occlusions, similar appearances, and players moving in and out of the frame.

**🎯 Challenge Objective**: Given a 15-second football video, identify each player and ensure that players who go out of frame and reappear are assigned the same identity as before - simulating real-time re-identification and player tracking.

### Key Features

- **Distance-Based Tracking**: Primary tracking using position-based similarity matching
- **Multi-Modal Re-identification**: Combines appearance features, jersey numbers, and spatial information
- **Real-time Processing**: Optimized for live video analysis
- **Robust Jersey Recognition**: OCR-based jersey number detection with blur tolerance
- **Role Classification**: Automatic detection of goalkeepers, players, and referees
- **Adaptive Thresholding**: Dynamic distance thresholds based on player velocity

## 🚀 Demo

The system successfully tracks players across a 15-second football clip, maintaining consistent IDs even when players:
- Move out of frame and reappear
- Are temporarily occluded by other players
- Change their orientation or posture
- Experience lighting variations

## 📋 Technical Requirements

### Dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install easyocr
pip install scipy
pip install numpy
pip install torch
pip install pickle5
```

### System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- 8GB+ RAM
- OpenCV 4.5+

## 📁 Project Structure

```
Player-Tracker/
├── Football_Player_Player.ipynb    # Main implementation notebook
├── best.pt                         # Pre-trained YOLOv11 model
├── 15sec_input_720p.mp4           # Input video file
├── distance_based_tracked_video.mp4 # Output video with tracking
├── distance_based_tracking_data.pkl # Tracking data
├── distance_tracking_stats.csv     # Performance statistics
├── requirements.txt                # Python dependencies
└── README.md                      # This file
```

1. **Clone the repository**
```bash
git clone https://github.com/Shine-5705/Player-Tracker.git
cd Player-Tracker
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Download the pre-trained model**
- Download the YOLOv11 model from: [Google Drive Link](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
- Place the model file (`best.pt`) in the project directory

4. **Prepare your video**
- Place your input video (e.g., `15sec_input_720p.mp4`) in the project directory

## 🛠️ Installation

### Running the Jupyter Notebook

The main implementation is provided in a Jupyter notebook for easy experimentation:

```bash
# Start Jupyter Notebook
jupyter notebook

# Open the main notebook
# Navigate to: Football_Player_Player.ipynb
```

### Basic Usage

```python
# Inside Football_Player_Player.ipynb or as a standalone script
from player_tracking import process_video_distance_based

# Configure paths
MODEL_PATH = "best.pt"
VIDEO_PATH = "15sec_input_720p.mp4"
OUTPUT_PATH = "tracked_output.mp4"

# Optional: Define field boundary to filter detections
FIELD_BOUNDARY = (20, 20, 1900, 1060)  # (x1, y1, x2, y2)

# Process video
tracker = process_video_distance_based(
    video_path=VIDEO_PATH,
    model_path=MODEL_PATH,
    output_path=OUTPUT_PATH,
    field_boundary=FIELD_BOUNDARY
)
```

### Advanced Configuration

```python
from player_tracking import DistanceBasedPlayerReIdentification

# Initialize tracker with custom parameters
tracker = DistanceBasedPlayerReIdentification(
    model_path="best.pt",
    max_disappeared=30,      # Frames before considering player lost
    min_bbox_area=1500,      # Minimum detection area
    field_boundary=None      # Optional field boundary
)

# Process single frame
cap = cv2.VideoCapture("video.mp4")
ret, frame = cap.read()
tracked_players = tracker.process_frame(frame)
```

### Analyzing Results

```python
# Load and analyze tracking data
from analysis import load_pickle_file, summarize_player_data

data = load_pickle_file("distance_based_tracking_data.pkl")
summarize_player_data(data)
```

## 📖 Usage

### Core Components

1. **Object Detection**: YOLOv11-based player detection
2. **Feature Extraction**: Multi-modal feature extraction including:
   - Color histograms (HSV-based)
   - Dominant color analysis
   - Jersey number recognition (OCR)
   - Bounding box dimensions
3. **Similarity Matching**: Weighted combination of:
   - Spatial distance (70%)
   - Appearance similarity (20%)
   - Size consistency (10%)
4. **Tracking Logic**: Hungarian algorithm for optimal assignment
5. **Re-identification**: Persistent identity management across frames

### Key Algorithms

- **Distance-Based Assignment**: Primary tracking using Euclidean distance
- **Adaptive Thresholding**: Dynamic distance thresholds based on player velocity
- **Multi-Modal Scoring**: Combines spatial, appearance, and size features
- **Jersey Number Locking**: Persistent jersey-to-player associations
- **Role Classification**: Automatic goalkeeper/player/referee detection

## 📊 Performance Metrics

The system tracks several performance indicators:

- **Active Players**: Number of currently tracked players
- **Detection Accuracy**: Frame-by-frame detection success rate
- **Jersey Recognition**: Success rate of jersey number detection
- **Tracking Stability**: Consistency of player ID assignments
- **Re-identification Success**: Accuracy when players re-enter frame

## 🔧 Configuration Options

### Tracking Parameters

```python
# Distance thresholds
base_distance_threshold = 120        # Base distance for matching
blur_distance_multiplier = 2.5       # Multiplier for high-velocity players
max_distance_threshold = 300         # Maximum allowed distance

# Matching weights
matching_weights = {
    'distance': 0.70,    # Spatial distance weight
    'appearance': 0.20,  # Visual similarity weight
    'size': 0.10        # Size consistency weight
}

# Detection filters
min_bbox_area = 1500                 # Minimum detection area
max_players = 25                     # Maximum tracked players
max_disappeared = 30                 # Frames before cleanup
```

### Visual Features

```python
# Color analysis
color_bins_hue = 12                  # HSV hue histogram bins
color_bins_saturation = 8            # HSV saturation histogram bins

# Jersey recognition
jersey_confidence_threshold = 0.4    # Minimum OCR confidence
jersey_roi = (0.1, 0.6, 0.2, 0.8)  # Chest region for OCR
```

## 📈 Output Files

The system generates several output files:

1. **Tracked Video**: Visual output with player annotations
2. **Tracking Data**: Pickle file with complete tracking history
3. **Statistics CSV**: Frame-by-frame tracking statistics
4. **Analysis JSON**: Human-readable tracking summary

## 🔍 Troubleshooting

### Common Issues

**Low Detection Accuracy**
- Adjust `conf` parameter in YOLO detection
- Modify `min_bbox_area` for your video resolution
- Check `field_boundary` settings

**Identity Switching**
- Increase appearance weight in `matching_weights`
- Adjust `base_distance_threshold`
- Improve jersey number recognition accuracy

**Performance Issues**
- Reduce video resolution
- Disable jersey OCR for faster processing
- Adjust `max_players` limit

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv11 framework
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for jersey number recognition
- [SciPy](https://scipy.org/) for optimization algorithms

## 📞 Contact

For questions or support, please [open an issue](https://github.com/Shine-5705/Player-Tracker/issues) or contact [guptashine5002@gmail.com](mailto:guptashine5002@gmail.com).

---

**⭐ If you find this project helpful, please consider giving it a star!**
