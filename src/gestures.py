
# MediaPipe Hand Landmark Indices
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

import numpy as np
from collections import deque, Counter

class GestureClassifier:
    def __init__(self, window_size=8, debounce_count=4):
        self.window_size = window_size
        self.debounce_count = debounce_count
        self.history = deque(maxlen=window_size)
        self.last_stable = None
        self.stable_counter = 0

        # default thresholds 
        self.angle_open_threshold = 45.0   
        self.tip_dist_open = 0.65        
        self.tip_dist_folded = 0.45       
        self.thumb_up_dot_threshold = 0.5 

        # calibration data
        self.calibrated_open_palm_feats = None

    def calibrate_open_palm(self, feat_mean):
        tip_dists = feat_mean[5:10]
        mean_tip = np.mean(tip_dists)
        self.tip_dist_open = float(mean_tip * 0.85)
        self.tip_dist_folded = float(self.tip_dist_open * 0.5)
        self.calibrated_open_palm_feats = feat_mean.copy()

    def finger_is_extended(self, angle_deg, tip_dist):
        angle_ok = angle_deg < self.angle_open_threshold  
        dist_ok = tip_dist > self.tip_dist_open
        return angle_ok and dist_ok

    def thumb_is_up(self, feat, lm_norm):
        thumb_tip = np.array(lm_norm[THUMB_TIP])
        thumb_ip = np.array(lm_norm[THUMB_IP])
        thumb_mcp = np.array(lm_norm[THUMB_MCP])
        wrist = np.array(lm_norm[WRIST])
        
        thumb_vec = thumb_tip - thumb_mcp
        thumb_vec = thumb_vec / (np.linalg.norm(thumb_vec) + 1e-6)
        
        index_mcp = np.array(lm_norm[INDEX_FINGER_MCP])
        pinky_mcp = np.array(lm_norm[PINKY_MCP])
        palm_vec = index_mcp - pinky_mcp
        palm_vec = palm_vec / (np.linalg.norm(palm_vec) + 1e-6)
        
        up_vec = np.array([0, -1, 0])  
        
        thumb_dot_up = np.dot(thumb_vec, up_vec)
        
        thumb_tip_dist = np.linalg.norm(thumb_tip - wrist)
        thumb_ip_dist = np.linalg.norm(thumb_ip - wrist)
        thumb_extended = thumb_tip_dist > thumb_ip_dist * 1.2
        
        other_fingers_folded = True
        for finger_tip in [INDEX_FINGER_TIP, MIDDLE_FINGER_TIP, RING_FINGER_TIP, PINKY_TIP]:
            tip_pos = np.array(lm_norm[finger_tip])
            mcp_pos = np.array(lm_norm[finger_tip - 3])  
            tip_to_mcp_dist = np.linalg.norm(tip_pos - mcp_pos)
            palm_width = np.linalg.norm(index_mcp - pinky_mcp)
            
            if tip_to_mcp_dist > palm_width * 0.5:  
                other_fingers_folded = False
                break
        
        # Thumbs up conditions
        thumb_up = (thumb_dot_up > 0.7 and  
                    thumb_extended and      
                    other_fingers_folded)   
        
        return thumb_up



    def map_states_to_gesture(self, feat, lm_norm):
        if self.thumb_is_up(feat, lm_norm):
            return "Thumbs Up"
        
        angles = feat[0:4]
        thumb_angle = feat[4]
        dists = feat[5:10]  

        extended = []
        for i in range(4):
            ext = self.finger_is_extended(angles[i], dists[i+1])  
            extended.append(ext)

        if all(extended) and np.mean(dists[1:]) > self.tip_dist_open * 0.9:
            return "Open Palm"

        if not any(extended) and np.mean(dists[1:]) < self.tip_dist_folded * 1.5:
            return "Fist"
        if (extended[0] and extended[1] and 
            not extended[2] and not extended[3] and
            dists[1] > self.tip_dist_open * 0.8 and  
            dists[2] > self.tip_dist_open * 0.8 and  
            dists[3] < self.tip_dist_folded * 1.3 and 
            dists[4] < self.tip_dist_folded * 1.3):   
            return "Peace"
        single_finger_extended = sum(extended) == 1
        if single_finger_extended:
            if extended[0]:  
                return "Pointing"
            elif extended[1]:  
                return "Middle Finger"

        return None

    

    def debug_thumb_features(self, lm_norm):
        thumb_tip = np.array(lm_norm[THUMB_TIP])
        thumb_ip = np.array(lm_norm[THUMB_IP])
        thumb_mcp = np.array(lm_norm[THUMB_MCP])
        wrist = np.array(lm_norm[WRIST])
        
        thumb_vec = thumb_tip - thumb_mcp
        thumb_vec = thumb_vec / (np.linalg.norm(thumb_vec) + 1e-6)
        
        up_vec = np.array([0, -1, 0])
        
        thumb_dot_up = np.dot(thumb_vec, up_vec)
        
        print(f"Thumb dot up: {thumb_dot_up:.2f}")
        print(f"Thumb vector: {thumb_vec}")
        
        return thumb_dot_up

    def draw_thumb_debug_info(self, image, lm_norm):
        height, width, _ = image.shape
        thumb_tip = (int(lm_norm[THUMB_TIP][0] * width), int(lm_norm[THUMB_TIP][1] * height))
        thumb_mcp = (int(lm_norm[THUMB_MCP][0] * width), int(lm_norm[THUMB_MCP][1] * height))
        
        cv2.arrowedLine(image, thumb_mcp, thumb_tip, (0, 255, 255), 2)
        
        up_vector_end = (thumb_mcp[0], thumb_mcp[1] - 50)  
        cv2.arrowedLine(image, thumb_mcp, up_vector_end, (255, 255, 0), 2)
        
        return image

    def get_hand_orientation(self, lm_norm):
        wrist = np.array(lm_norm[WRIST])
        middle_mcp = np.array(lm_norm[MIDDLE_FINGER_MCP])
        
        orientation_vec = middle_mcp - wrist
        orientation_vec = orientation_vec / (np.linalg.norm(orientation_vec) + 1e-6)
        
        return orientation_vec

    def predict(self, feat, lm_norm=None):
        label = self.map_states_to_gesture(feat, lm_norm)
        self.history.append(label)
        non_none = [l for l in self.history if l is not None]
        if not non_none:
            return (self.last_stable, 1.0 if self.last_stable else 0.0)

        counts = Counter(non_none)
        candidate, count = counts.most_common(1)[0]
        confidence = count / len(self.history)

        if candidate != self.last_stable:
            self.stable_counter = self.stable_counter + 1 if self.last_stable != candidate else 1
            if count >= self.debounce_count and self.stable_counter >= 1:
                self.last_stable = candidate
                self.stable_counter = 0
        else:
            self.stable_counter = 0  

        return (self.last_stable, confidence)

    def register_no_hand(self):
        self.history.append(None)
        return
