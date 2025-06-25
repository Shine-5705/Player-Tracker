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
import easyocr

class PlayerReIdentification:
    def __init__(self, model_path, similarity_threshold=0.4, max_disappeared=200, player_class_id=None):
        self.model = YOLO(model_path)
        self.similarity_threshold = similarity_threshold
        self.max_disappeared = max_disappeared
        self.player_class_id = player_class_id
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}
        if self.player_class_id is None:
            self.player_class_id = self._find_player_class_id()
        self.next_player_id = 1
        self.player_features = {}
        self.player_positions = {}
        self.player_colors = {}
        self.player_jerseys = {}
        self.player_movements = {}
        self.player_body_ratios = {}
        self.disappeared_count = defaultdict(int)
        self.active_players = set()
        self.all_seen_players = set()
        self.player_last_seen = {}
        self.current_frame = 0
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((128, 64)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    def _find_player_class_id(self):
        player_keywords = ['player', 'person', 'people', 'human']
        for class_id, class_name in self.class_names.items():
            if any(keyword in class_name.lower() for keyword in player_keywords):
                return class_id
        return 0
    
    def extract_jersey_number(self, player_crop):
        try:
            h, w = player_crop.shape[:2]
            chest_region = player_crop[int(h*0.2):int(h*0.6), int(w*0.2):int(w*0.8)]
            results = self.ocr_reader.readtext(chest_region, allowlist='0123456789', width_ths=0.4, height_ths=0.4)
            if results:
                numbers = [result[1] for result in results if result[1].isdigit() and len(result[1]) <= 2]
                if numbers: return int(numbers[0])
        except: pass
        return None
    
    def extract_dominant_jersey_colors(self, player_crop):
        h, w = player_crop.shape[:2]
        torso_region = player_crop[int(h*0.2):int(h*0.7), int(w*0.1):int(w*0.9)]
        data = torso_region.reshape((-1, 3)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        _, counts = np.unique(labels, return_counts=True)
        dominant_colors = centers[np.argsort(counts)[::-1]]
        return np.uint8(dominant_colors)
    
    def extract_body_ratio(self, bbox):
        x1, y1, x2, y2 = bbox
        width, height = x2 - x1, y2 - y1
        return height / width if width > 0 else 0
    
    def extract_visual_features(self, player_crop):
        if player_crop.size == 0: return np.array([])
        resized = cv2.resize(player_crop, (64, 128))
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        s_hist = cv2.calcHist([hsv], [1], None, [16], [0, 256])
        v_hist = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        v_hist = cv2.normalize(v_hist, v_hist).flatten()
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_hist, _ = np.histogram(edges.ravel(), bins=16, range=(0, 255))
        edge_hist = edge_hist.astype(float) / (edge_hist.sum() + 1e-6)
        return np.concatenate([h_hist, s_hist, v_hist, edge_hist])
    
    def calculate_jersey_similarity(self, jersey1, jersey2):
        if jersey1 is None or jersey2 is None: return 0
        return 1.0 if jersey1 == jersey2 else 0
    
    def calculate_color_similarity(self, colors1, colors2):
        if colors1 is None or colors2 is None: return 0
        similarities = []
        for c1 in colors1:
            for c2 in colors2:
                diff = np.linalg.norm(c1.astype(float) - c2.astype(float))
                similarities.append(max(0, 1 - diff / 441.67))
        return max(similarities) if similarities else 0
    
    def calculate_movement_similarity(self, movements1, movements2):
        if not movements1 or not movements2: return 0
        recent1 = movements1[-min(5, len(movements1)):]
        recent2 = movements2[-min(5, len(movements2)):]
        if not recent1 or not recent2: return 0
        avg1 = np.mean([np.linalg.norm(m) for m in recent1])
        avg2 = np.mean([np.linalg.norm(m) for m in recent2])
        return max(0, 1 - abs(avg1 - avg2) / 100.0)
    
    def calculate_trajectory_similarity(self, current_pos, current_movement, player_id):
        if player_id not in self.player_positions or len(self.player_positions[player_id]) < 2: return 0.5
        last_positions = list(self.player_positions[player_id])[-min(3, len(self.player_positions[player_id])):]
        if len(last_positions) < 2: return 0.5
        predicted_pos = (last_positions[-1][0] + (last_positions[-1][0] - last_positions[-2][0]), last_positions[-1][1] + (last_positions[-1][1] - last_positions[-2][1]))
        distance_to_predicted = np.sqrt((current_pos[0] - predicted_pos[0])**2 + (current_pos[1] - predicted_pos[1])**2)
        return max(0, 1 - distance_to_predicted / 150.0)
    
    def calculate_motion_consistency(self, player_id, current_movement):
        if player_id not in self.player_movements or len(self.player_movements[player_id]) < 2: return 0.5
        recent_movements = list(self.player_movements[player_id])[-min(3, len(self.player_movements[player_id])):]
        if len(recent_movements) < 2: return 0.5
        avg_movement = np.mean(recent_movements, axis=0)
        consistency = 1 - (np.linalg.norm(current_movement - avg_movement) / 100.0)
        return max(0, min(1, consistency))
    
    def calculate_position_similarity(self, pos1, pos2):
        distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        normalized_distance = distance / np.sqrt(1920**2 + 1080**2)
        return max(0, 1 - normalized_distance * 3)
    
    def calculate_feature_similarity(self, features1, features2):
        if len(features1) == 0 or len(features2) == 0: return 0
        f1, f2 = features1.reshape(1, -1), features2.reshape(1, -1)
        return max(0, cosine_similarity(f1, f2)[0][0])
    
    def calculate_body_ratio_similarity(self, ratio1, ratio2):
        if ratio1 == 0 or ratio2 == 0: return 0
        return max(0, 1 - abs(ratio1 - ratio2) / max(ratio1, ratio2))
    
    def match_players(self, current_detections):
        if not self.all_seen_players:
            assignments = {}
            for i, detection in enumerate(current_detections):
                player_id = self.next_player_id
                assignments[i] = player_id
                self.next_player_id += 1
                self.active_players.add(player_id)
                self.all_seen_players.add(player_id)
                self.player_last_seen[player_id] = self.current_frame
            return assignments
        
        all_player_list = list(self.all_seen_players)
        similarity_matrix = np.zeros((len(current_detections), len(all_player_list)))
        
        for i, detection in enumerate(current_detections):
            det_features = detection['features']
            det_position = detection['center']
            det_jersey = detection['jersey_number']
            det_colors = detection['colors']
            det_ratio = detection['body_ratio']
            
            for j, player_id in enumerate(all_player_list):
                if player_id in self.player_features and len(self.player_features[player_id]) > 0:
                    jersey_similarities, color_similarities, feature_similarities = [], [], []
                    position_similarities, movement_similarities, ratio_similarities = [], [], []
                    
                    recent_count = min(8, len(self.player_features[player_id]))
                    for k in range(recent_count):
                        hist_features = self.player_features[player_id][-(k+1)]
                        hist_position = self.player_positions[player_id][-(k+1)]
                        hist_jersey = self.player_jerseys[player_id][-(k+1)] if player_id in self.player_jerseys and len(self.player_jerseys[player_id]) > k else None
                        hist_colors = self.player_colors[player_id][-(k+1)]
                        hist_ratio = self.player_body_ratios[player_id][-(k+1)] if player_id in self.player_body_ratios and len(self.player_body_ratios[player_id]) > k else 0
                        
                        jersey_similarities.append(self.calculate_jersey_similarity(det_jersey, hist_jersey))
                        color_similarities.append(self.calculate_color_similarity(det_colors, hist_colors))
                        feature_similarities.append(self.calculate_feature_similarity(det_features, hist_features))
                        position_similarities.append(self.calculate_position_similarity(det_position, hist_position))
                        ratio_similarities.append(self.calculate_body_ratio_similarity(det_ratio, hist_ratio))
                        
                        if player_id in self.player_movements and len(self.player_movements[player_id]) > 0:
                            movement_similarities.append(self.calculate_movement_similarity([detection.get('movement', np.array([0, 0]))], [self.player_movements[player_id][-(k+1)]]))
                    
                    best_jersey_sim = max(jersey_similarities) if jersey_similarities else 0
                    best_color_sim = max(color_similarities) if color_similarities else 0
                    best_feature_sim = max(feature_similarities) if feature_similarities else 0
                    best_position_sim = max(position_similarities) if position_similarities else 0
                    best_movement_sim = max(movement_similarities) if movement_similarities else 0
                    best_ratio_sim = max(ratio_similarities) if ratio_similarities else 0
                    
                    frames_since_seen = self.current_frame - self.player_last_seen.get(player_id, 0)
                    recency_weight = max(0.3, 1.0 - (frames_since_seen / 300.0))
                    
                    trajectory_sim = self.calculate_trajectory_similarity(det_position, detection.get('movement', np.array([0, 0])), player_id)
                    motion_consistency = self.calculate_motion_consistency(player_id, detection.get('movement', np.array([0, 0])))
                    
                    in_motion_bonus = 0
                    if np.linalg.norm(detection.get('movement', np.array([0, 0]))) > 5:
                        in_motion_bonus = 0.3 * (trajectory_sim + motion_consistency) / 2
                    
                    jersey_weight = 0.35 if best_jersey_sim > 0 else 0
                    color_weight = 0.2
                    feature_weight = 0.1
                    position_weight = 0.15
                    movement_weight = 0.1
                    ratio_weight = 0.05
                    trajectory_weight = 0.05
                    
                    total_weight = jersey_weight + color_weight + feature_weight + position_weight + movement_weight + ratio_weight
                    if total_weight > 0:
                        jersey_weight /= total_weight
                        color_weight /= total_weight
                        feature_weight /= total_weight
                        position_weight /= total_weight
                        movement_weight /= total_weight
                        ratio_weight /= total_weight
                    
                    total_similarity = (jersey_weight * best_jersey_sim + color_weight * best_color_sim + feature_weight * best_feature_sim + position_weight * best_position_sim + movement_weight * best_movement_sim + ratio_weight * best_ratio_sim) * recency_weight
                    similarity_matrix[i, j] = total_similarity
        
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
                self.active_players.add(player_id)
                self.disappeared_count[player_id] = 0
                self.player_last_seen[player_id] = self.current_frame
        
        for det_idx in unmatched_detections:
            player_id = self.next_player_id
            assignments[det_idx] = player_id
            self.next_player_id += 1
            self.active_players.add(player_id)
            self.all_seen_players.add(player_id)
            matched_players.add(player_id)
            self.player_last_seen[player_id] = self.current_frame
        
        for player_id in self.active_players.copy():
            if player_id not in matched_players:
                self.disappeared_count[player_id] += 1
                if self.disappeared_count[player_id] > self.max_disappeared:
                    self.active_players.remove(player_id)
        
        return assignments
    
    def update_player_history(self, player_id, features, position, colors, jersey_number, body_ratio, movement):
        max_history = 25
        if player_id not in self.player_features:
            self.player_features[player_id] = deque(maxlen=max_history)
            self.player_positions[player_id] = deque(maxlen=max_history)
            self.player_colors[player_id] = deque(maxlen=max_history)
            self.player_jerseys[player_id] = deque(maxlen=max_history)
            self.player_movements[player_id] = deque(maxlen=max_history)
            self.player_body_ratios[player_id] = deque(maxlen=max_history)
        self.player_features[player_id].append(features)
        self.player_positions[player_id].append(position)
        self.player_colors[player_id].append(colors)
        self.player_jerseys[player_id].append(jersey_number)
        self.player_movements[player_id].append(movement)
        self.player_body_ratios[player_id].append(body_ratio)
    
    def process_frame(self, frame):
        self.current_frame += 1
        results = self.model(frame)
        current_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    if confidence > 0.5 and class_id == self.player_class_id:
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        player_crop = frame[y1:y2, x1:x2]
                        
                        if player_crop.size > 0:
                            features = self.extract_visual_features(player_crop)
                            colors = self.extract_dominant_jersey_colors(player_crop)
                            jersey_number = self.extract_jersey_number(player_crop)
                            center = ((x1 + x2) // 2, (y1 + y2) // 2)
                            body_ratio = self.extract_body_ratio((x1, y1, x2, y2))
                            
                            detection = {
                                'bbox': (x1, y1, x2, y2),
                                'center': center,
                                'features': features,
                                'colors': colors,
                                'jersey_number': jersey_number,
                                'body_ratio': body_ratio,
                                'confidence': confidence,
                                'crop': player_crop,
                                'movement': np.array([0, 0])
                            }
                            current_detections.append(detection)
        
        assignments = self.match_players(current_detections)
        tracked_players = []
        
        for det_idx, player_id in assignments.items():
            detection = current_detections[det_idx]
            movement = np.array([0, 0])
            
            if player_id in self.player_positions and len(self.player_positions[player_id]) > 0:
                last_pos = self.player_positions[player_id][-1]
                current_pos = detection['center']
                movement = np.array([current_pos[0] - last_pos[0], current_pos[1] - last_pos[1]])
            
            detection['movement'] = movement
            
            self.update_player_history(player_id, detection['features'], detection['center'], detection['colors'], detection['jersey_number'], detection['body_ratio'], movement)
            
            status = "ACTIVE"
            if player_id in self.disappeared_count and self.disappeared_count[player_id] > 0:
                status = "RETURNED"
            
            tracked_player = {
                'player_id': player_id,
                'bbox': detection['bbox'],
                'center': detection['center'],
                'confidence': detection['confidence'],
                'jersey_number': detection['jersey_number'],
                'status': status,
                'frames_since_first_seen': self.current_frame
            }
            tracked_players.append(tracked_player)
        
        return tracked_players
    
    def save_tracking_data(self, filepath):
        data = {'next_player_id': self.next_player_id, 'player_features': dict(self.player_features), 'player_positions': dict(self.player_positions), 'player_colors': dict(self.player_colors), 'player_jerseys': dict(self.player_jerseys), 'player_movements': dict(self.player_movements), 'player_body_ratios': dict(self.player_body_ratios), 'active_players': self.active_players, 'all_seen_players': self.all_seen_players}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_tracking_data(self, filepath):
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            self.next_player_id = data['next_player_id']
            self.player_features = defaultdict(deque, data['player_features'])
            self.player_positions = defaultdict(deque, data['player_positions'])
            self.player_colors = defaultdict(deque, data['player_colors'])
            self.player_jerseys = defaultdict(deque, data.get('player_jerseys', {}))
            self.player_movements = defaultdict(deque, data.get('player_movements', {}))
            self.player_body_ratios = defaultdict(deque, data.get('player_body_ratios', {}))
            self.active_players = data['active_players']
            self.all_seen_players = data.get('all_seen_players', self.active_players)

def debug_model_classes(model_path, video_path, num_frames=5):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    if hasattr(model, 'names'):
        print(f"Model class names: {model.names}")
    frame_count = 0
    class_detections = defaultdict(int)
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret: break
        results = model(frame)
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    class_id = int(box.cls[0].cpu().numpy())
                    confidence = box.conf[0].cpu().numpy()
                    class_name = model.names.get(class_id, f"class_{class_id}") if hasattr(model, 'names') else f"class_{class_id}"
                    class_detections[class_id] += 1
        frame_count += 1
    cap.release()
    for class_id, count in class_detections.items():
        class_name = model.names.get(class_id, f"class_{class_id}") if hasattr(model, 'names') else f"class_{class_id}"
        print(f"Class {class_id} ('{class_name}'): {count} detections")

def process_video(video_path, model_path, output_path=None, debug_classes=False):
    if debug_classes:
        debug_model_classes(model_path, video_path)
        return None
    reid_system = PlayerReIdentification(model_path)
    cap = cv2.VideoCapture(video_path)
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    frame_count = 0
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0), (255, 192, 203), (0, 191, 255), (50, 205, 50), (255, 20, 147), (0, 250, 154), (255, 69, 0), (138, 43, 226), (255, 215, 0), (220, 20, 60), (32, 178, 170)]
    while True:
        ret, frame = cap.read()
        if not ret: break
        tracked_players = reid_system.process_frame(frame)
        for player in tracked_players:
            x1, y1, x2, y2 = player['bbox']
            player_id = player['player_id']
            confidence = player['confidence']
            jersey_number = player['jersey_number']
            status = player['status']
            color = colors[(player_id - 1) % len(colors)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            label = f"ID:{player_id}"
            if jersey_number: label += f" #{jersey_number}"
            status_label = f"[{status}]" if status == "RETURNED" else ""
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
            cv2.putText(frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            if status == "RETURNED":
                cv2.putText(frame, status_label, (x1 + 5, y1 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            center_x, center_y = player['center']
            cv2.circle(frame, (center_x, center_y), 3, color, -1)
        cv2.rectangle(frame, (10, 10), (400, 120), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_count}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Visible: {len(tracked_players)}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Total: {len(reid_system.all_seen_players)}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        active_count = len(reid_system.active_players)
        total_count = len(reid_system.all_seen_players)
        cv2.putText(frame, f"Active: {active_count} | Inactive: {total_count - active_count}", (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if output_path: out.write(frame)
        cv2.imshow('Enhanced Player Re-ID', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames, Total players: {len(reid_system.all_seen_players)}")
    cap.release()
    if output_path: out.release()
    cv2.destroyAllWindows()
    print(f"Complete: {frame_count} frames, {len(reid_system.all_seen_players)} unique players")
    return reid_system

if __name__ == "__main__":
    MODEL_PATH = "best.pt"
    VIDEO_PATH = "15sec_input_720p.mp4"
    OUTPUT_PATH = "output_tracked_video.mp4"
    reid_system = process_video(VIDEO_PATH, MODEL_PATH, OUTPUT_PATH)
    if reid_system: reid_system.save_tracking_data("tracking_data.pkl")