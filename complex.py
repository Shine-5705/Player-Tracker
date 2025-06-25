import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import pickle
import os
import cv2


class PlayerReIdentification:
    def __init__(self, model_path, similarity_threshold=0.6, max_disappeared=150, player_class_id=None):
        """
        Initialize the Player Re-Identification system
        
        Args:
            model_path: Path to the YOLO model
            similarity_threshold: Minimum similarity score for matching players
            max_disappeared: Maximum frames a player can be absent (increased for permanent tracking)
            player_class_id: Specific class ID for players (if None, will auto-detect)
        """
        self.model = YOLO(model_path)
        self.similarity_threshold = similarity_threshold
        self.max_disappeared = max_disappeared
        self.player_class_id = player_class_id
        
        # Get class names from model
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        print(f"Model classes: {self.class_names}")
        
        # Auto-detect player class ID if not provided
        if self.player_class_id is None:
            self.player_class_id = self._find_player_class_id()
        
        print(f"Using class ID {self.player_class_id} for players")
        
        # Player tracking data - PERMANENT throughout video
        self.next_player_id = 1
        self.player_features = {}  # player_id -> feature_history (NEVER deleted)
        self.player_positions = {}  # player_id -> position_history (NEVER deleted)
        self.player_colors = {}    # player_id -> color_history (NEVER deleted)
        self.disappeared_count = defaultdict(int)
        self.active_players = set()  # Currently visible players
        self.all_seen_players = set()  # All players ever seen (PERMANENT)
        self.player_last_seen = {}  # player_id -> frame_number
        self.current_frame = 0
        
        # Feature extraction setup
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standard person re-id size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _find_player_class_id(self):
        """Auto-detect the class ID for players based on class names"""
        player_keywords = ['player', 'person', 'people', 'human']
        
        for class_id, class_name in self.class_names.items():
            class_name_lower = class_name.lower()
            if any(keyword in class_name_lower for keyword in player_keywords):
                return class_id
        
        # If no match found, assume class 0 (most common)
        print("Warning: Could not auto-detect player class. Using class 0.")
        return 0
        
        # Feature extraction setup
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 64)),  # Standard person re-id size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract_visual_features(self, player_crop):
        """Extract visual features from player crop"""
        if player_crop.size == 0:
            return np.array([])
        
        # Resize to standard size
        resized = cv2.resize(player_crop, (64, 128))
        
        # Extract color histogram features
        color_features = self.extract_color_histogram(resized)
        
        # Extract texture features using LBP (Local Binary Patterns)
        texture_features = self.extract_texture_features(resized)
        
        # Combine features
        features = np.concatenate([color_features, texture_features])
        
        return features
    
    def extract_color_histogram(self, image, bins=32):
        """Extract color histogram features"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms for each channel
        h_hist = cv2.calcHist([hsv], [0], None, [bins], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [bins], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [bins], [0, 256])
        
        # Normalize histograms
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        
        return np.concatenate([h_hist, s_hist, v_hist])
    
    def extract_texture_features(self, image, radius=1, n_points=8):
        """Extract texture features using Local Binary Patterns"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Simple LBP implementation
        h, w = gray.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray[i, j]
                code = 0
                for k, (di, dj) in enumerate([(-1,-1), (-1,0), (-1,1), (0,1), 
                                            (1,1), (1,0), (1,-1), (0,-1)]):
                    if gray[i + di, j + dj] >= center:
                        code |= (1 << k)
                lbp[i, j] = code
        
        # Calculate LBP histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(float)
        hist /= (hist.sum() + 1e-6)  # Normalize
        
        return hist
    
    def extract_dominant_colors(self, image, k=3):
        """Extract dominant colors from player crop"""
        # Reshape image to be a list of pixels
        data = image.reshape((-1, 3))
        data = np.float32(data)
        
        # Apply k-means clustering
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8 and return dominant colors
        centers = np.uint8(centers)
        return centers
    
    def calculate_position_similarity(self, pos1, pos2):
        """Calculate position similarity between two detections"""
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        # Normalize by image dimensions (assuming 1920x1080)
        normalized_distance = distance / np.sqrt(1920**2 + 1080**2)
        return max(0, 1 - normalized_distance * 5)  # Scale factor for position importance
    
    def calculate_feature_similarity(self, features1, features2):
        """Calculate cosine similarity between feature vectors"""
        if len(features1) == 0 or len(features2) == 0:
            return 0
        
        # Reshape for cosine similarity calculation
        f1 = features1.reshape(1, -1)
        f2 = features2.reshape(1, -1)
        
        similarity = cosine_similarity(f1, f2)[0][0]
        return max(0, similarity)
    
    def match_players(self, current_detections):
        """Match current detections with existing players - PERMANENT ID SYSTEM"""
        if not self.all_seen_players:
            # First frame - create new IDs
            assignments = {}
            for i, detection in enumerate(current_detections):
                player_id = self.next_player_id
                assignments[i] = player_id
                self.next_player_id += 1
                self.active_players.add(player_id)
                self.all_seen_players.add(player_id)
                self.player_last_seen[player_id] = self.current_frame
            return assignments
        
        # Calculate similarity matrix against ALL EVER SEEN players (not just active)
        all_player_list = list(self.all_seen_players)
        similarity_matrix = np.zeros((len(current_detections), len(all_player_list)))
        
        for i, detection in enumerate(current_detections):
            det_features = detection['features']
            det_position = detection['center']
            
            for j, player_id in enumerate(all_player_list):
                # Get best matching features from history
                if player_id in self.player_features and len(self.player_features[player_id]) > 0:
                    # Calculate similarity with multiple historical features
                    feature_similarities = []
                    position_similarities = []
                    
                    # Check against last few appearances
                    recent_count = min(5, len(self.player_features[player_id]))
                    for k in range(recent_count):
                        hist_features = self.player_features[player_id][-(k+1)]
                        hist_position = self.player_positions[player_id][-(k+1)]
                        
                        feature_sim = self.calculate_feature_similarity(det_features, hist_features)
                        position_sim = self.calculate_position_similarity(det_position, hist_position)
                        
                        feature_similarities.append(feature_sim)
                        position_similarities.append(position_sim)
                    
                    # Use maximum similarity (best match from history)
                    best_feature_sim = max(feature_similarities) if feature_similarities else 0
                    best_position_sim = max(position_similarities) if position_similarities else 0
                    
                    # Give higher weight to recently seen players
                    frames_since_seen = self.current_frame - self.player_last_seen.get(player_id, 0)
                    recency_weight = max(0.5, 1.0 - (frames_since_seen / 500.0))  # Decay over time
                    
                    # Combined similarity with recency weighting
                    total_similarity = (0.7 * best_feature_sim + 0.3 * best_position_sim) * recency_weight
                    similarity_matrix[i, j] = total_similarity
        
        # Use Hungarian algorithm for optimal assignment
        if len(current_detections) > 0 and len(all_player_list) > 0:
            detection_indices, player_indices = linear_sum_assignment(-similarity_matrix)
        else:
            detection_indices, player_indices = [], []
        
        assignments = {}
        unmatched_detections = set(range(len(current_detections)))
        matched_players = set()
        
        for det_idx, player_idx in zip(detection_indices, player_indices):
            if similarity_matrix[det_idx, player_idx] > self.similarity_threshold:
                player_id = all_player_list[player_idx]
                assignments[det_idx] = player_id
                unmatched_detections.remove(det_idx)
                matched_players.add(player_id)
                
                # Reactivate player if they were inactive
                self.active_players.add(player_id)
                self.disappeared_count[player_id] = 0
                self.player_last_seen[player_id] = self.current_frame
        
        # Create NEW IDs for truly unmatched detections
        for det_idx in unmatched_detections:
            player_id = self.next_player_id
            assignments[det_idx] = player_id
            self.next_player_id += 1
            self.active_players.add(player_id)
            self.all_seen_players.add(player_id)
            matched_players.add(player_id)
            self.player_last_seen[player_id] = self.current_frame
        
        # Update disappeared count for unmatched active players (but NEVER delete them)
        for player_id in self.active_players.copy():
            if player_id not in matched_players:
                self.disappeared_count[player_id] += 1
                if self.disappeared_count[player_id] > self.max_disappeared:
                    self.active_players.remove(player_id)  # Remove from active, but keep in all_seen_players
                    # NEVER delete player data - keep it permanently for future matching
        
        return assignments
    
    def update_player_history(self, player_id, features, position, colors):
        """Update player history with new detection - PERMANENT STORAGE"""
        max_history = 20  # Keep more history for better matching
        
        # Update features history (NEVER delete player data)
        if player_id not in self.player_features:
            self.player_features[player_id] = deque(maxlen=max_history)
        self.player_features[player_id].append(features)
        
        # Update position history
        if player_id not in self.player_positions:
            self.player_positions[player_id] = deque(maxlen=max_history)
        self.player_positions[player_id].append(position)
        
        # Update color history
        if player_id not in self.player_colors:
            self.player_colors[player_id] = deque(maxlen=max_history)
        self.player_colors[player_id].append(colors)
    
    def process_frame(self, frame):
        """Process a single frame and return tracked players with PERMANENT IDs"""
        self.current_frame += 1
        
        # Run YOLO detection
        results = self.model(frame)
        
        current_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Filter for players only (not ball or referees)
                    if confidence > 0.5 and class_id == self.player_class_id:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Extract player crop
                        player_crop = frame[y1:y2, x1:x2]
                        
                        if player_crop.size > 0:
                            # Extract features
                            features = self.extract_visual_features(player_crop)
                            
                            # Extract dominant colors
                            colors = self.extract_dominant_colors(player_crop)
                            
                            # Calculate center position
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            
                            detection = {
                                'bbox': (x1, y1, x2, y2),
                                'center': center,
                                'features': features,
                                'colors': colors,
                                'confidence': confidence,
                                'crop': player_crop
                            }
                            
                            current_detections.append(detection)
        
        # Match detections with existing players (including previously seen ones)
        assignments = self.match_players(current_detections)
        
        # Update player histories
        tracked_players = []
        for det_idx, player_id in assignments.items():
            detection = current_detections[det_idx]
            
            # Update history (permanent storage)
            self.update_player_history(
                player_id, 
                detection['features'], 
                detection['center'], 
                detection['colors']
            )
            
            # Add to tracked players with status
            status = "ACTIVE"
            if player_id in self.disappeared_count and self.disappeared_count[player_id] > 0:
                status = "RETURNED"  # Player returned after disappearing
            
            tracked_player = {
                'player_id': player_id,
                'bbox': detection['bbox'],
                'center': detection['center'],
                'confidence': detection['confidence'],
                'status': status,
                'frames_since_first_seen': self.current_frame
            }
            tracked_players.append(tracked_player)
        
        return tracked_players
    
    def save_tracking_data(self, filepath):
        """Save tracking data to file"""
        data = {
            'next_player_id': self.next_player_id,
            'player_features': dict(self.player_features),
            'player_positions': dict(self.player_positions),
            'player_colors': dict(self.player_colors),
            'active_players': self.active_players
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_tracking_data(self, filepath):
        """Load tracking data from file"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                
            self.next_player_id = data['next_player_id']
            self.player_features = defaultdict(deque, data['player_features'])
            self.player_positions = defaultdict(deque, data['player_positions'])
            self.player_colors = defaultdict(deque, data['player_colors'])
            self.active_players = data['active_players']


def debug_model_classes(model_path, video_path, num_frames=5):
    """Debug function to identify class IDs and their meanings"""
    print("=== DEBUGGING MODEL CLASSES ===")
    
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    
    # Print model class names
    if hasattr(model, 'names'):
        print(f"Model class names: {model.names}")
    
    frame_count = 0
    class_detections = defaultdict(int)
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)
        
        print(f"\nFrame {frame_count + 1}:")
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    
                    class_name = model.names.get(class_id, f"class_{class_id}") if hasattr(model, 'names') else f"class_{class_id}"
                    
                    print(f"  - Class ID: {class_id}, Name: '{class_name}', Confidence: {confidence:.2f}")
                    class_detections[class_id] += 1
        
        frame_count += 1
    
    cap.release()
    
    print(f"\n=== SUMMARY ===")
    print(f"Total detections per class:")
    for class_id, count in class_detections.items():
        class_name = model.names.get(class_id, f"class_{class_id}") if hasattr(model, 'names') else f"class_{class_id}"
        print(f"  Class {class_id} ('{class_name}'): {count} detections")
    
    print(f"\nRecommendation:")
    print(f"- Use the class ID that corresponds to 'players' for player_class_id parameter")
    print(f"- Exclude ball and referee class IDs")


def process_video(video_path, model_path, output_path=None, debug_classes=False):
    """Process entire video with PERMANENT player re-identification"""
    
    # Debug classes first if requested
    if debug_classes:
        debug_model_classes(model_path, video_path)
        return None
    
    # Initialize the re-identification system
    reid_system = PlayerReIdentification(model_path)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if output_path:
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    # Color palette for different players (up to 20 different colors)
    colors = [
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Yellow
        (128, 0, 128),  # Purple
        (255, 165, 0),  # Orange
        (0, 128, 128),  # Teal
        (128, 128, 0),  # Olive
        (255, 192, 203), # Pink
        (0, 191, 255),  # Deep Sky Blue
        (50, 205, 50),  # Lime Green
        (255, 20, 147), # Deep Pink
        (0, 250, 154),  # Medium Spring Green
        (255, 69, 0),   # Red Orange
        (138, 43, 226), # Blue Violet
        (255, 215, 0),  # Gold
        (220, 20, 60),  # Crimson
        (32, 178, 170)  # Light Sea Green
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        tracked_players = reid_system.process_frame(frame)
        
        # Draw enhanced bounding boxes and IDs
        for player in tracked_players:
            x1, y1, x2, y2 = player['bbox']
            player_id = player['player_id']
            confidence = player['confidence']
            status = player['status']
            
            # Select color based on player ID
            color = colors[(player_id - 1) % len(colors)]
            
            # Draw thicker bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Create background for text
            label = f"ID: {player_id}"
            status_label = f"[{status}]" if status == "RETURNED" else ""
            
            # Calculate text size
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            (status_width, status_height), _ = cv2.getTextSize(status_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            
            # Draw background rectangle for text
            cv2.rectangle(frame, (x1, y1 - text_height - 10), 
                         (x1 + max(text_width, status_width) + 10, y1), color, -1)
            
            # Draw player ID
            cv2.putText(frame, label, (x1 + 5, y1 - text_height + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Draw status if returned
            if status == "RETURNED":
                cv2.putText(frame, status_label, (x1 + 5, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw confidence score
            conf_label = f"{confidence:.2f}"
            cv2.putText(frame, conf_label, (x2 - 50, y2 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center point
            center_x, center_y = player['center']
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        
        # Draw comprehensive frame info
        info_bg_height = 120
        cv2.rectangle(frame, (10, 10), (400, info_bg_height), (0, 0, 0), -1)
        
        # Frame number
        cv2.putText(frame, f"Frame: {frame_count}", (15, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Currently visible players
        cv2.putText(frame, f"Visible Players: {len(tracked_players)}", (15, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Total unique players ever seen
        cv2.putText(frame, f"Total Players Seen: {len(reid_system.all_seen_players)}", (15, 85), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Active vs inactive breakdown
        active_count = len(reid_system.active_players)
        total_count = len(reid_system.all_seen_players)
        inactive_count = total_count - active_count
        cv2.putText(frame, f"Active: {active_count} | Inactive: {inactive_count}", (15, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if output_path:
            out.write(frame)
        
        # Display frame (optional)
        cv2.imshow('Player Re-Identification - PERMANENT IDs', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
            print(f"Total unique players: {len(reid_system.all_seen_players)}")
            print(f"Currently active: {len(reid_system.active_players)}")
    
    # Cleanup
    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\n=== VIDEO PROCESSING COMPLETE ===")
    print(f"Total frames processed: {frame_count}")
    print(f"Total unique players detected: {len(reid_system.all_seen_players)}")
    print(f"Player IDs assigned: {list(sorted(reid_system.all_seen_players))}")
    print(f"Final active players: {len(reid_system.active_players)}")
    
    return reid_system


# Example usage
if __name__ == "__main__":
    # Update these paths according to your setup
    MODEL_PATH = "best.pt"  # Your YOLO model path
    VIDEO_PATH = "15sec_input_720p.mp4"
    OUTPUT_PATH = "output_tracked_video.mp4"
    
    # STEP 1: Debug your model classes first
    print("Step 1: Debugging model classes...")
    debug_model_classes(MODEL_PATH, VIDEO_PATH)
    
    # STEP 2: Process the video with correct class ID
    # After debugging, you might need to specify the correct player_class_id
    # For example: reid_system = PlayerReIdentification(MODEL_PATH, player_class_id=1)
    print("\nStep 2: Processing video...")
    reid_system = process_video(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH)
    
    if reid_system:
        # Save tracking data for future use
        reid_system.save_tracking_data("tracking_data.pkl")