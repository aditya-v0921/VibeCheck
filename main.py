#!/usr/bin/env python3
"""
Vibe-Check: Biometric-Driven Crowd Analytics
Main server with live video streaming and skeleton visualization
"""

import time
import asyncio
import base64
import numpy as np
import cv2
import socketio
from ultralytics import YOLO
from aiohttp import web

# Import the C++ motion engine
import vibe_core

# CONFIGURATION

CAM_SOURCE = 1  # "auto", integer index (0,1,2...), or RTSP URL
PREFERRED_RESOLUTION = (1280, 720)
PREFERRED_FPS = 30

GRID_H, GRID_W = 8, 8
POSE_EVERY_N_FRAMES = 2
CONF_THRESH = 0.35
EMIT_RATE_HZ = 15
JPEG_QUALITY = 70

# SOCKET.IO SERVER SETUP

sio = socketio.AsyncServer(
    async_mode="aiohttp",
    cors_allowed_origins="*",
    ping_timeout=60,
    ping_interval=25,
    max_http_buffer_size=10 * 1024 * 1024
)
app = web.Application()
sio.attach(app)

# KEYPOINT INDICES & SKELETON

KP_NOSE = 0
KP_L_EYE, KP_R_EYE = 1, 2
KP_L_EAR, KP_R_EAR = 3, 4
KP_L_SHOULDER, KP_R_SHOULDER = 5, 6
KP_L_ELBOW, KP_R_ELBOW = 7, 8
KP_L_WRIST, KP_R_WRIST = 9, 10
KP_L_HIP, KP_R_HIP = 11, 12
KP_L_KNEE, KP_R_KNEE = 13, 14
KP_L_ANKLE, KP_R_ANKLE = 15, 16

SKELETON_CONNECTIONS = [
    (KP_L_EYE, KP_R_EYE), (KP_L_EYE, KP_NOSE), (KP_R_EYE, KP_NOSE),
    (KP_L_EYE, KP_L_EAR), (KP_R_EYE, KP_R_EAR),
    (KP_L_SHOULDER, KP_R_SHOULDER),
    (KP_L_SHOULDER, KP_L_ELBOW), (KP_R_SHOULDER, KP_R_ELBOW),
    (KP_L_ELBOW, KP_L_WRIST), (KP_R_ELBOW, KP_R_WRIST),
    (KP_L_SHOULDER, KP_L_HIP), (KP_R_SHOULDER, KP_R_HIP),
    (KP_L_HIP, KP_R_HIP),
    (KP_L_HIP, KP_L_KNEE), (KP_R_HIP, KP_R_KNEE),
    (KP_L_KNEE, KP_L_ANKLE), (KP_R_KNEE, KP_R_ANKLE),
]

COLORS = {
    'face': (255, 200, 100),
    'arms': (100, 255, 100),
    'torso': (100, 200, 255),
    'legs': (255, 100, 255),
}


def get_limb_color(idx1, idx2):
    if idx1 in {0, 1, 2, 3, 4} or idx2 in {0, 1, 2, 3, 4}:
        return COLORS['face']
    elif idx1 in {7, 8, 9, 10} or idx2 in {7, 8, 9, 10}:
        return COLORS['arms']
    elif idx1 in {13, 14, 15, 16} or idx2 in {13, 14, 15, 16}:
        return COLORS['legs']
    else:
        return COLORS['torso']


def compute_hype_metrics(kpts_xyc, prev_kpts=None):
    result = {"hands_up": 0.0, "jumping": 0.0, "total": 0.0}
    
    nose = kpts_xyc[KP_NOSE]
    l_wrist, r_wrist = kpts_xyc[KP_L_WRIST], kpts_xyc[KP_R_WRIST]
    l_shoulder, r_shoulder = kpts_xyc[KP_L_SHOULDER], kpts_xyc[KP_R_SHOULDER]
    
    if nose[2] < CONF_THRESH and l_shoulder[2] < CONF_THRESH and r_shoulder[2] < CONF_THRESH:
        return result
    
    head_y = nose[1] if nose[2] >= CONF_THRESH else (l_shoulder[1] + r_shoulder[1]) / 2 - 50
    
    hands_up_count = 0
    if l_wrist[2] >= CONF_THRESH and l_wrist[1] < head_y:
        hands_up_count += 1
    if r_wrist[2] >= CONF_THRESH and r_wrist[1] < head_y:
        hands_up_count += 1
    result["hands_up"] = hands_up_count / 2.0
    
    if prev_kpts is not None:
        l_ankle, r_ankle = kpts_xyc[KP_L_ANKLE], kpts_xyc[KP_R_ANKLE]
        prev_l_ankle, prev_r_ankle = prev_kpts[KP_L_ANKLE], prev_kpts[KP_R_ANKLE]
        
        movements = []
        if l_ankle[2] >= CONF_THRESH and prev_l_ankle[2] >= CONF_THRESH:
            movements.append(prev_l_ankle[1] - l_ankle[1])
        if r_ankle[2] >= CONF_THRESH and prev_r_ankle[2] >= CONF_THRESH:
            movements.append(prev_r_ankle[1] - r_ankle[1])
        
        if movements and np.mean(movements) > 15:
            result["jumping"] = min(1.0, np.mean(movements) / 40.0)
    
    result["total"] = min(1.0, result["hands_up"] * 0.7 + result["jumping"] * 0.3)
    return result


def classify_zones(heatmap):
    flat = heatmap.flatten()
    if flat.max() == 0:
        return [["low"] * heatmap.shape[1] for _ in range(heatmap.shape[0])]
    
    nonzero = flat[flat > 0]
    p33 = np.percentile(nonzero, 33) if len(nonzero) > 0 else 0
    p66 = np.percentile(nonzero, 66) if len(nonzero) > 0 else 0
    
    zones = []
    for row in heatmap:
        zone_row = []
        for val in row:
            if val <= p33:
                zone_row.append("low")
            elif val <= p66:
                zone_row.append("medium")
            else:
                zone_row.append("high")
        zones.append(zone_row)
    return zones


def draw_skeleton(frame, keypoints, conf_thresh=0.3):
    overlay = frame.copy()
    
    for person_kpts in keypoints:
        # Draw bones
        for idx1, idx2 in SKELETON_CONNECTIONS:
            pt1, pt2 = person_kpts[idx1], person_kpts[idx2]
            if pt1[2] >= conf_thresh and pt2[2] >= conf_thresh:
                x1, y1 = int(pt1[0]), int(pt1[1])
                x2, y2 = int(pt2[0]), int(pt2[1])
                color = get_limb_color(idx1, idx2)
                cv2.line(overlay, (x1, y1), (x2, y2), color, 8)
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw joints
        for i, kpt in enumerate(person_kpts):
            if kpt[2] >= conf_thresh:
                x, y = int(kpt[0]), int(kpt[1])
                if i in {0, 1, 2, 3, 4}:
                    color = COLORS['face']
                elif i in {5, 6, 7, 8, 9, 10}:
                    color = COLORS['arms']
                elif i in {11, 12}:
                    color = COLORS['torso']
                else:
                    color = COLORS['legs']
                
                cv2.circle(overlay, (x, y), 10, color, -1)
                cv2.circle(frame, (x, y), 6, color, -1)
                cv2.circle(frame, (x, y), 6, (255, 255, 255), 2)
        
        # "HYPE!" indicator
        nose = person_kpts[KP_NOSE]
        l_wrist, r_wrist = person_kpts[KP_L_WRIST], person_kpts[KP_R_WRIST]
        
        if nose[2] >= conf_thresh:
            hands_up = (l_wrist[2] >= conf_thresh and l_wrist[1] < nose[1]) or \
                       (r_wrist[2] >= conf_thresh and r_wrist[1] < nose[1])
            if hands_up:
                cv2.putText(frame, "HYPE!", (int(nose[0]) - 30, int(nose[1]) - 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    return frame


# CAMERA SETUP

def find_camera():
    print("\nScanning for cameras...")
    for i in range(10):
        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"  ✓ Camera {i}: {w}x{h}")
                if w >= 1280:
                    return i
    return 0


def open_camera(source):
    if source == "auto":
        source = find_camera()
    
    print(f"\nOpening camera: {source}")
    
    cap = cv2.VideoCapture(source, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {source}")
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, PREFERRED_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Resolution: {w}x{h}")
    
    return cap


# MAIN VISION LOOP

async def vision_loop():
    cap = open_camera(CAM_SOURCE)
    engine = vibe_core.MotionEngine(GRID_H, GRID_W)
    
    print("\nLoading YOLOv8 pose model...")
    model = YOLO("yolov8n-pose.pt")
    
    ret, warm = cap.read()
    if ret:
        _ = model.predict(warm, verbose=False, conf=CONF_THRESH)
        print("  ✓ Model ready")
    
    frame_count = 0
    last_emit = time.time()
    prev_kpts_map = {}
    last_keypoints = None
    energy_hist, hype_hist = [], []
    
    print("\n" + "=" * 50)
    print("Vision loop started!")
    print("   Dashboard: http://localhost:3000")
    print("=" * 50 + "\n")
    
    try:
        while True:
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                await asyncio.sleep(0.01)
                continue
            
            display = frame.copy()
            heatmap, mean_energy = engine.process(frame)
            
            hype_score = 0.0
            people_count = 0
            hands_up_ratio = 0.0
            jumping_ratio = 0.0
            keypoints_list = []
            
            if frame_count % POSE_EVERY_N_FRAMES == 0:
                results = model.predict(frame, verbose=False, conf=CONF_THRESH)
                r0 = results[0]
                
                if r0.keypoints is not None and r0.keypoints.data is not None:
                    kpts = r0.keypoints.data.cpu().numpy()
                    people_count = kpts.shape[0]
                    last_keypoints = kpts
                    
                    if people_count > 0:
                        hype_results = []
                        for i in range(people_count):
                            metrics = compute_hype_metrics(kpts[i], prev_kpts_map.get(i))
                            hype_results.append(metrics)
                            prev_kpts_map[i] = kpts[i].copy()
                            
                            keypoints_list.append([
                                {"x": float(k[0]), "y": float(k[1]), "conf": float(k[2])}
                                for k in kpts[i]
                            ])
                        
                        hype_score = float(np.mean([h["total"] for h in hype_results]))
                        hands_up_ratio = float(np.mean([h["hands_up"] for h in hype_results]))
                        jumping_ratio = float(np.mean([h["jumping"] for h in hype_results]))
            
            # Draw skeleton
            if last_keypoints is not None and len(last_keypoints) > 0:
                display = draw_skeleton(display, last_keypoints, CONF_THRESH)
            
            # Stats overlay
            cv2.putText(display, f"People: {people_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Energy: {mean_energy:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, f"Hype: {hype_score*100:.0f}%", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            frame_count += 1
            t1 = time.time()
            
            # Smoothing
            energy_hist.append(float(mean_energy))
            hype_hist.append(hype_score)
            if len(energy_hist) > 10:
                energy_hist.pop(0)
                hype_hist.pop(0)
            
            # Encode frame
            _, buffer = cv2.imencode('.jpg', display, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_b64 = base64.b64encode(buffer).decode('utf-8')
            
            payload = {
                "ts": time.time(),
                "meanEnergy": float(np.mean(energy_hist)),
                "heatmap": heatmap.tolist(),
                "zones": classify_zones(heatmap),
                "hypeScore": float(np.mean(hype_hist)),
                "handsUpRatio": hands_up_ratio,
                "jumpingRatio": jumping_ratio,
                "peopleCount": people_count,
                "latencyMs": round((t1 - t0) * 1000, 1),
                "fps": round(1.0 / max(0.001, t1 - t0), 1),
                "frame": frame_b64,
                "keypoints": keypoints_list,
                "frameWidth": display.shape[1],
                "frameHeight": display.shape[0]
            }
            
            now = time.time()
            if now - last_emit >= (1.0 / EMIT_RATE_HZ):
                await sio.emit("vibe_update", payload)
                last_emit = now
            
            await asyncio.sleep(0)
    finally:
        cap.release()


# HTTP ROUTES

async def index_handler(request):
    return web.Response(text="Vibe-Check Server Running", content_type="text/plain")

app.router.add_get("/", index_handler)

# APP LIFECYCLE

async def start_background_tasks(app):
    app["vision_task"] = asyncio.create_task(vision_loop())

async def cleanup_background_tasks(app):
    app["vision_task"].cancel()
    try:
        await app["vision_task"]
    except asyncio.CancelledError:
        pass

app.on_startup.append(start_background_tasks)
app.on_cleanup.append(cleanup_background_tasks)


if __name__ == "__main__":
    print("""
VIBE-CHECK
Live Video + Skeleton Visualization""")
    web.run_app(app, host="0.0.0.0", port=8000)