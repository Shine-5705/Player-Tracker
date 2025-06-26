import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import torch
import pickle
import cv2
import easyocr
import sys
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import euclidean

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    from google.colab.patches import cv2_imshow

class DistanceBasedPlayerReIdentification:
    def __init__(self, model_path, max_disappeared=30, min_bbox_area=1500, field_boundary=None):
        self.model = YOLO(model_path)
        self.max_disappeared = max_disappeared
        self.min_bbox_area = min_bbox_area
        self.field_boundary = field_boundary
        self.class_names = self.model.names if hasattr(self.model, 'names') else {}

        self.base_distance_threshold = 120
        self.blur_distance_multiplier = 2.5
        self.max_distance_threshold = 300

        self.matching_weights = {
            'distance': 0.70,
            'appearance': 0.20,
            'size': 0.10
        }

        self.max_players = 25
        self.player_class_id = 0
        self._initialize_class_ids()

        self.next_player_id = 1
        self.active_players = {}
        self.disappeared_players = {}
        self.current_frame = 0
        self.ocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

        self.last_positions = {}
        self.position_history = defaultdict(lambda: deque(maxlen=5))
        self.velocity_estimates = {}
        self.appearance_cache = {}
        self.size_history = defaultdict(lambda: deque(maxlen=3))

        self.jersey_assignments = {}
        self.player_jerseys = {}

        self.goal_areas = [
            (0, 150, 120, 570),
            (1800, 150, 1920, 570)
        ]

        self.tracking_stats = []

    def _initialize_class_ids(self):
        logging.info(f"Available classes: {self.class_names}")

        for class_id, class_name in self.class_names.items():
            name_lower = class_name.lower()
            if any(keyword in name_lower for keyword in ['player', 'person', 'human']):
                self.player_class_id = class_id
                break

        logging.info(f"Using class ID {self.player_class_id} for detection")

    def extract_appearance_features(self, player_crop, bbox):
        if player_crop.size == 0:
            return {}

        try:
            h, w = player_crop.shape[:2]
            torso_region = player_crop[int(h*0.25):int(h*0.75), int(w*0.2):int(w*0.8)]
            color_hist = self._extract_color_histogram(torso_region)
            dominant_color = self._get_dominant_color(torso_region)
            jersey_num, jersey_conf = self._extract_jersey_number_blur_tolerant(player_crop)

            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1

            return {
                'color_histogram': color_hist,
                'dominant_color': dominant_color,
                'jersey_number': jersey_num,
                'jersey_confidence': jersey_conf,
                'size': (width, height),
                'area': width * height
            }

        except Exception as e:
            logging.warning(f"Appearance feature extraction failed: {e}")
            return {}

    def _extract_color_histogram(self, region):
        if region.size == 0:
            return np.array([])

        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [12], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])

            hist_h = cv2.normalize(hist_h, hist_h).flatten()
            hist_s = cv2.normalize(hist_s, hist_s).flatten()

            return np.concatenate([hist_h, hist_s])
        except:
            return np.array([])

    def _get_dominant_color(self, region):
        if region.size == 0:
            return 'unknown'

        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            h_mean = np.mean(hsv[:, :, 0])
            s_mean = np.mean(hsv[:, :, 1])
            v_mean = np.mean(hsv[:, :, 2])

            if s_mean < 60:
                if v_mean > 180:
                    return 'white'
                elif v_mean < 80:
                    return 'black'
                else:
                    return 'gray'

            if 0 <= h_mean <= 15 or 165 <= h_mean <= 180:
                return 'red'
            elif 100 <= h_mean <= 130:
                return 'blue'
            elif 15 <= h_mean <= 35:
                return 'yellow'
            elif 35 <= h_mean <= 85:
                return 'green'
            else:
                return 'other'
        except:
            return 'unknown'

    def _extract_jersey_number_blur_tolerant(self, player_crop):
        if player_crop.size == 0:
            return None, 0.0

        try:
            h, w = player_crop.shape[:2]
            chest_region = player_crop[int(h*0.1):int(h*0.6), int(w*0.2):int(w*0.8)]

            if chest_region.size == 0:
                return None, 0.0

            gray = cv2.cvtColor(chest_region, cv2.COLOR_BGR2GRAY)

            enhanced_images = [
                cv2.equalizeHist(gray),
                cv2.GaussianBlur(gray, (3, 3), 0),
                cv2.bilateralFilter(gray, 9, 75, 75),
                cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
            ]

            best_number = None
            best_confidence = 0.0

            for enhanced in enhanced_images:
                try:
                    results = self.ocr_reader.readtext(
                        enhanced,
                        allowlist='0123456789',
                        width_ths=0.3,
                        height_ths=0.3,
                        detail=1
                    )

                    for (bbox, text, confidence) in results:
                        clean_text = ''.join(filter(str.isdigit, text))
                        if clean_text and len(clean_text) <= 2:
                            try:
                                number = int(clean_text)
                                if 1 <= number <= 99 and confidence > best_confidence:
                                    best_number = number
                                    best_confidence = confidence
                            except:
                                continue
                except Exception:
                    continue

            return best_number, best_confidence
        except Exception:
            return None, 0.0

    def calculate_adaptive_distance_threshold(self, player_id):
        base_threshold = self.base_distance_threshold

        if player_id in self.velocity_estimates:
            recent_velocity = self.velocity_estimates[player_id]
            adaptive_threshold = base_threshold + (recent_velocity * self.blur_distance_multiplier)
            return min(adaptive_threshold, self.max_distance_threshold)

        return base_threshold

    def calculate_distance_score(self, detection_center, player_id):
        if player_id not in self.last_positions:
            return 0.0

        last_pos = self.last_positions[player_id]
        distance = euclidean(detection_center, last_pos)
        threshold = self.calculate_adaptive_distance_threshold(player_id)

        if distance <= threshold:
            return 1.0 - (distance / threshold)
        else:
            return 0.0

    def calculate_appearance_score(self, detection_features, player_id):
        if player_id not in self.appearance_cache:
            return 0.0

        cached_features = self.appearance_cache[player_id]
        total_score = 0.0
        score_count = 0

        if (len(detection_features['color_histogram']) > 0 and
            len(cached_features.get('color_histogram', [])) > 0):
            try:
                corr = cv2.compareHist(
                    detection_features['color_histogram'].astype(np.float32),
                    cached_features['color_histogram'].astype(np.float32),
                    cv2.HISTCMP_CORREL
                )
                total_score += max(0, corr)
                score_count += 1
            except:
                pass

        if (detection_features['dominant_color'] != 'unknown' and
            cached_features.get('dominant_color') != 'unknown'):
            if detection_features['dominant_color'] == cached_features['dominant_color']:
                total_score += 1.0
            score_count += 1

        if (detection_features['jersey_number'] and
            detection_features['jersey_confidence'] > 0.3 and
            cached_features.get('jersey_number')):
            if detection_features['jersey_number'] == cached_features['jersey_number']:
                total_score += 1.0
            score_count += 1

        return total_score / max(score_count, 1)

    def calculate_size_score(self, detection_features, player_id):
        if player_id not in self.size_history or len(self.size_history[player_id]) == 0:
            return 0.5

        recent_sizes = list(self.size_history[player_id])
        avg_width = np.mean([s[0] for s in recent_sizes])
        avg_height = np.mean([s[1] for s in recent_sizes])

        current_width, current_height = detection_features['size']

        width_ratio = min(avg_width/current_width, current_width/avg_width) if current_width > 0 else 0
        height_ratio = min(avg_height/current_height, current_height/avg_height) if current_height > 0 else 0

        return (width_ratio + height_ratio) / 2

    def calculate_combined_similarity(self, detection_features, detection_center, player_id):
        distance_score = self.calculate_distance_score(detection_center, player_id)
        appearance_score = self.calculate_appearance_score(detection_features, player_id)
        size_score = self.calculate_size_score(detection_features, player_id)

        combined_score = (
            self.matching_weights['distance'] * distance_score +
            self.matching_weights['appearance'] * appearance_score +
            self.matching_weights['size'] * size_score
        )

        return combined_score

    def assign_player_ids_distance_based(self, detections):
        if not detections:
            return {}

        active_player_ids = list(self.active_players.keys())
        assignments = {}

        if not active_player_ids:
            for i, det in enumerate(detections):
                player_id = self.next_player_id
                assignments[i] = player_id
                self.next_player_id += 1
                self.active_players[player_id] = det['features']
        else:
            similarity_matrix = np.zeros((len(detections), len(active_player_ids)))

            for i, det in enumerate(detections):
                detection_center = det['features']['center']
                detection_features = det['features']

                for j, player_id in enumerate(active_player_ids):
                    similarity_matrix[i, j] = self.calculate_combined_similarity(
                        detection_features, detection_center, player_id
                    )

            if similarity_matrix.size > 0 and np.max(similarity_matrix) > 0:
                row_indices, col_indices = linear_sum_assignment(-similarity_matrix)

                matched_detections = set()
                matched_players = set()

                for i, j in zip(row_indices, col_indices):
                    similarity = similarity_matrix[i, j]

                    if similarity > 0.3:
                        player_id = active_player_ids[j]
                        assignments[i] = player_id
                        matched_detections.add(i)
                        matched_players.add(player_id)

                        if player_id in self.disappeared_players:
                            del self.disappeared_players[player_id]

                for i in range(len(detections)):
                    if i not in matched_detections and len(self.active_players) < self.max_players:
                        player_id = self.next_player_id
                        assignments[i] = player_id
                        self.next_player_id += 1
                        self.active_players[player_id] = detections[i]['features']

                for j, player_id in enumerate(active_player_ids):
                    if player_id not in matched_players:
                        self.disappeared_players[player_id] = self.disappeared_players.get(player_id, 0) + 1

                        if self.disappeared_players[player_id] > self.max_disappeared:
                            self._cleanup_player_data(player_id)

        return assignments

    def _cleanup_player_data(self, player_id):
        if player_id in self.active_players:
            del self.active_players[player_id]
        if player_id in self.disappeared_players:
            del self.disappeared_players[player_id]

        if player_id in self.last_positions:
            del self.last_positions[player_id]
        if player_id in self.position_history:
            del self.position_history[player_id]
        if player_id in self.velocity_estimates:
            del self.velocity_estimates[player_id]
        if player_id in self.appearance_cache:
            del self.appearance_cache[player_id]
        if player_id in self.size_history:
            del self.size_history[player_id]

        if player_id in self.player_jerseys:
            jersey_num = self.player_jerseys[player_id][0]
            if jersey_num in self.jersey_assignments:
                del self.jersey_assignments[jersey_num]
            del self.player_jerseys[player_id]

    def update_player_tracking_data(self, assignments, detections):
        for det_idx, player_id in assignments.items():
            features = detections[det_idx]['features']
            current_center = features['center']

            if player_id in self.last_positions:
                last_pos = self.last_positions[player_id]
                velocity = euclidean(current_center, last_pos)
                self.velocity_estimates[player_id] = velocity

            self.last_positions[player_id] = current_center
            self.position_history[player_id].append(current_center)
            self.size_history[player_id].append(features['size'])

            if player_id in self.appearance_cache:
                cached = self.appearance_cache[player_id]

                if features['dominant_color'] != 'unknown':
                    cached['dominant_color'] = features['dominant_color']

                if len(features['color_histogram']) > 0:
                    if len(cached.get('color_histogram', [])) > 0:
                        cached['color_histogram'] = (
                            0.7 * cached['color_histogram'] +
                            0.3 * features['color_histogram']
                        )
                    else:
                        cached['color_histogram'] = features['color_histogram']
            else:
                self.appearance_cache[player_id] = {
                    'color_histogram': features['color_histogram'],
                    'dominant_color': features['dominant_color'],
                    'jersey_number': features['jersey_number']
                }

            if features['jersey_number'] and features['jersey_confidence'] > 0.4:
                current_jersey_player = self.jersey_assignments.get(features['jersey_number'])

                if (current_jersey_player is None or
                    current_jersey_player == player_id or
                    features['jersey_confidence'] > self.player_jerseys.get(current_jersey_player, (None, 0))[1]):

                    if current_jersey_player and current_jersey_player != player_id:
                        if current_jersey_player in self.player_jerseys:
                            del self.player_jerseys[current_jersey_player]

                    self.player_jerseys[player_id] = (features['jersey_number'], features['jersey_confidence'])
                    self.jersey_assignments[features['jersey_number']] = player_id

                    if player_id in self.appearance_cache:
                        self.appearance_cache[player_id]['jersey_number'] = features['jersey_number']

    def classify_player_role_simple(self, player_id, position):
        x, y = position

        in_goal = False
        for gx1, gy1, gx2, gy2 in self.goal_areas:
            if gx1 <= x <= gx2 and gy1 <= y <= gy2:
                in_goal = True
                break

        jersey_num = None
        if player_id in self.player_jerseys:
            jersey_num, _ = self.player_jerseys[player_id]

        if in_goal and jersey_num == 1:
            return 'goalkeeper'
        elif jersey_num and jersey_num > 90:
            return 'referee'
        else:
            return 'player'

    def process_frame(self, frame):
        self.current_frame += 1
        results = self.model(frame, conf=0.2, iou=0.4)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())

                if confidence < 0.2 or class_id != self.player_class_id:
                    continue

                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area < self.min_bbox_area:
                    continue

                width, height = x2 - x1, y2 - y1
                aspect_ratio = height / width if width > 0 else 0
                if not (1.2 <= aspect_ratio <= 4.5):
                    continue

                if self.field_boundary:
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    fx1, fy1, fx2, fy2 = self.field_boundary
                    if not (fx1 <= center_x <= fx2 and fy1 <= center_y <= fy2):
                        continue

                player_crop = frame[y1:y2, x1:x2]
                features = self.extract_appearance_features(player_crop, (x1, y1, x2, y2))

                if features:
                    features['center'] = ((x1 + x2) // 2, (y1 + y2) // 2)

                    detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'confidence': confidence,
                        'features': features
                    })

        assignments = self.assign_player_ids_distance_based(detections)
        self.update_player_tracking_data(assignments, detections)

        tracked_players = []

        for det_idx, player_id in assignments.items():
            det = detections[det_idx]
            features = det['features']

            role = self.classify_player_role_simple(player_id, features['center'])

            jersey_num = None
            if player_id in self.player_jerseys:
                jersey_num, _ = self.player_jerseys[player_id]

            tracked_player = {
                'player_id': player_id,
                'role': role,
                'bbox': det['bbox'],
                'center': features['center'],
                'confidence': det['confidence'],
                'jersey_number': jersey_num,
                'dominant_color': features['dominant_color'],
                'distance_score': self.calculate_distance_score(features['center'], player_id) if player_id in self.last_positions else 0.0
            }
            tracked_players.append(tracked_player)

        stats = {
            'frame': self.current_frame,
            'active_players': len(self.active_players),
            'total_detections': len(detections),
            'jersey_locked': len(self.player_jerseys),
            'avg_distance_threshold': np.mean([self.calculate_adaptive_distance_threshold(pid) for pid in self.active_players.keys()]) if self.active_players else 0
        }
        self.tracking_stats.append(stats)

        return tracked_players


def process_video_distance_based(video_path, model_path, output_path=None, field_boundary=None):
    tracker = DistanceBasedPlayerReIdentification(
        model_path=model_path,
        max_disappeared=30,
        min_bbox_area=1500,
        field_boundary=field_boundary
    )

    cap = cv2.VideoCapture(video_path)

    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0

    colors = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
        (0, 255, 255), (128, 0, 128), (255, 165, 0), (0, 128, 128), (128, 128, 0),
        (255, 192, 203), (0, 191, 255), (50, 205, 50), (255, 20, 147), (0, 250, 154),
        (255, 69, 0), (138, 43, 226), (255, 215, 0), (220, 20, 60), (32, 178, 170),
        (75, 0, 130), (255, 99, 71), (255, 140, 0), (124, 252, 0), (0, 206, 209)
    ]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_players = tracker.process_frame(frame)

        for player in tracked_players:
            x1, y1, x2, y2 = player['bbox']
            player_id = player['player_id']
            role = player['role']
            confidence = player['confidence']
            jersey = player['jersey_number']
            color_name = player['dominant_color']
            distance_score = player['distance_score']

            color = colors[(player_id - 1) % len(colors)]
            thickness = 3 if role == 'goalkeeper' else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            if role == 'goalkeeper':
                label = f"GK{player_id}"
            elif role == 'referee':
                label = f"REF{player_id}"
            else:
                label = f"P{player_id}"

            if jersey:
                label += f" #{jersey}"

            label += f" ({color_name})"
            label += f" D:{distance_score:.2f}"
            label += f" C:{confidence:.2f}"

            label_y = max(25, y1 - 5)
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)

            cv2.rectangle(frame, (x1 - 2, label_y - text_height - 3),
                         (x1 + text_width + 2, label_y + 2), color, -1)
            cv2.putText(frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            center_x, center_y = player['center']
            cv2.circle(frame, (center_x, center_y), 3, color, -1)

            if player_id in tracker.last_positions:
                last_x, last_y = tracker.last_positions[player_id]
                if (last_x, last_y) != (center_x, center_y):
                    cv2.line(frame, (last_x, last_y), (center_x, center_y), color, 1)

        if tracker.tracking_stats:
            stats = tracker.tracking_stats[-1]

            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (380, 120), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            cv2.putText(frame, f"Frame: {frame_count}", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Active Players: {stats['active_players']}", (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame, f"Detections: {stats['total_detections']}", (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Jersey Locked: {stats['jersey_locked']}", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.putText(frame, f"Avg Distance Threshold: {stats['avg_distance_threshold']:.1f}", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            cv2.putText(frame, f"Tracking Mode: DISTANCE-BASED", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        if output_path:
            out.write(frame)

        if not IN_COLAB:
            cv2.imshow('Distance-Based Player Tracking', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

        if frame_count % 30 == 0:
            active_count = len(tracker.active_players)
            avg_velocity = np.mean([tracker.velocity_estimates.get(pid, 0) for pid in tracker.active_players.keys()]) if tracker.active_players else 0
            logging.info(f"Frame {frame_count}: Active={active_count}, Avg_Velocity={avg_velocity:.1f}")

    cap.release()
    if output_path:
        out.release()
    if not IN_COLAB:
        cv2.destroyAllWindows()

    tracking_data = {
        'active_players': tracker.active_players,
        'last_positions': tracker.last_positions,
        'jersey_assignments': tracker.jersey_assignments,
        'tracking_stats': tracker.tracking_stats,
        'appearance_cache': tracker.appearance_cache
    }

    with open('distance_based_tracking_data.pkl', 'wb') as f:
        pickle.dump(tracking_data, f)

    with open('distance_tracking_stats.csv', 'w') as f:
        f.write('Frame,Active_Players,Total_Detections,Jersey_Locked,Avg_Distance_Threshold\n')
        for stat in tracker.tracking_stats:
            f.write(f"{stat['frame']},{stat['active_players']},{stat['total_detections']},"
                   f"{stat['jersey_locked']},{stat['avg_distance_threshold']:.1f}\n")

    final_stats = tracker.tracking_stats[-1] if tracker.tracking_stats else {}
    print(f"\nProcessing Complete: {frame_count} frames")
    print(f"Active Players: {final_stats.get('active_players', 0)}")
    print(f"Players with Jersey Numbers: {final_stats.get('jersey_locked', 0)}")
    print(f"Average Distance Threshold: {final_stats.get('avg_distance_threshold', 0):.1f}px")

    for jersey_num, player_id in sorted(tracker.jersey_assignments.items()):
        confidence = tracker.player_jerseys.get(player_id, (None, 0))[1]
        print(f"#{jersey_num} -> Player{player_id} (confidence: {confidence:.2f})")

    total_frames = len(tracker.tracking_stats)
    if total_frames > 1:
        player_count_variance = np.var([s['active_players'] for s in tracker.tracking_stats])
        avg_detections = np.mean([s['total_detections'] for s in tracker.tracking_stats])

        print(f"Total Frames: {total_frames}")
        print(f"Player Count Variance: {player_count_variance:.2f}")
        print(f"Average Detections per Frame: {avg_detections:.1f}")

        if player_count_variance < 2.0:
            stability = "EXCELLENT"
        elif player_count_variance < 5.0:
            stability = "GOOD"
        elif player_count_variance < 10.0:
            stability = "FAIR"
        else:
            stability = "NEEDS IMPROVEMENT"

        print(f"Tracking Stability: {stability}")

    return tracker


if __name__ == "__main__":
    MODEL_PATH = "best.pt"
    VIDEO_PATH = "15sec_input_720p.mp4"
    OUTPUT_PATH = "distance_based_tracked_video.mp4"
    FIELD_BOUNDARY = (20, 20, 1900, 1060)

    tracker = process_video_distance_based(
        video_path=VIDEO_PATH,
        model_path=MODEL_PATH,
        output_path=OUTPUT_PATH,
        field_boundary=FIELD_BOUNDARY
    )