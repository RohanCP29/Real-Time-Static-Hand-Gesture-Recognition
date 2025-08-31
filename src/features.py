import numpy as np

def normalize_landmarks(lm):
    wrist = lm[0].copy()
    lm_t = lm - wrist  # translate
    palm_size = np.linalg.norm(lm[9] - lm[0])  # wrist to middle_MCP
    if palm_size < 1e-6:
        palm_size = 1.0
    lm_norm = lm_t / palm_size
    return lm_norm

def angle_between(a, b):
    a = np.array(a); b = np.array(b)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    cosv = np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0)
    return np.degrees(np.arccos(cosv))

def extract_features_from_landmarks(lm_norm):
    feats = []
    fingers = {
        'thumb': (2, 3, 4),
        'index': (5, 6, 8),
        'middle': (9, 10, 12),
        'ring': (13, 14, 16),
        'pinky': (17, 18, 20)
    }

    wrist = lm_norm[0]

    # angles
    for finger in ['index', 'middle', 'ring', 'pinky']:
        mcp, pip, tip = fingers[finger]
        pip_pt = lm_norm[pip]
        mcp_pt = lm_norm[mcp]
        # DIP assumed at (pip + tip)/2
        dip_pt = (lm_norm[pip] + lm_norm[tip]) / 2.0
        v1 = pip_pt - mcp_pt
        v2 = dip_pt - pip_pt
        a = angle_between(v1, v2)
        feats.append(a)  # angle in degrees

    # thumb angle (MCP->IP vs IP->tip)
    thumb_mcp = lm_norm[2]
    thumb_ip = lm_norm[3]
    thumb_tip = lm_norm[4]
    a_thumb = angle_between(thumb_ip - thumb_mcp, thumb_tip - thumb_ip)
    feats.append(a_thumb)

    # tip distances (norm to wrist)
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        if finger == 'thumb':
            tip_idx = 4
        elif finger == 'index':
            tip_idx = 8
        elif finger == 'middle':
            tip_idx = 12
        elif finger == 'ring':
            tip_idx = 16
        else:
            tip_idx = 20
        d = np.linalg.norm(lm_norm[tip_idx] - wrist)
        feats.append(d)

    # thumb direction (projected)
    thumb_vec = (thumb_tip - thumb_mcp)[:2]  # x,y
    feats.extend([thumb_vec[0], thumb_vec[1]])

    # hand area proxy: bounding box area
    xs = lm_norm[:, 0]
    ys = lm_norm[:, 1]
    area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    feats.append(area)

    return np.array(feats, dtype=float)
