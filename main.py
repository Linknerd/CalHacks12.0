import cv2
import numpy as np
from ultralytics import YOLO

# COCO ids you care about
PERSON, BACKPACK, HANDBAG, SUITCASE, LAPTOP, CELL_PHONE = 0, 24, 26, 28, 63, 67
WATCHED_CLASSES = [BACKPACK, SUITCASE, LAPTOP, CELL_PHONE]  # ignore PERSON for planting

MODEL_PATH = "yolo11m.pt"
TRACKER    = "botsort.yaml"
DEVICE     = "cuda:0"   # use "cpu" if you don't have CUDA
CONF       = 0.4
IMGSZ      = 640

model = YOLO(MODEL_PATH)

# Per-class table:
#   objects_by_class[c] = [ {"ref": (x,y), "miss": 0, "taken": False}, ... ]
objects_by_class = {c: [] for c in WATCHED_CLASSES}
planted = False

MISS_LIMIT = 8       # require N consecutive misses to call TAKEN
BASE_OFFSET = 20     # pixel tolerance; we’ll scale it by image size a bit

def cls_id(box): return int(box.cls[0].item())
def center_xy(box):
    x, y, w, h = box.xywh[0].cpu().numpy()
    return int(x), int(y)

for r in model.track(
    source=1,                     # <-- set to 0 unless you truly have a second cam
    device=DEVICE,
    half=DEVICE.startswith("cuda"),
    tracker=TRACKER,
    persist=True,
    stream=True,
    conf=CONF,
    imgsz=IMGSZ,
    classes=[PERSON, BACKPACK, SUITCASE, LAPTOP, CELL_PHONE],
    verbose=False
):
    frame = r.orig_img.copy()
    H, W = frame.shape[:2]
    # dynamic tolerance: 2% of min(H,W), clamped around BASE_OFFSET
    OFFSET = max(BASE_OFFSET, int(0.02 * min(H, W)))

    # ---------- DETECTIONS (per class) ----------
    current_centers = {c: [] for c in WATCHED_CLASSES}
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            c = cls_id(box)
            if c in WATCHED_CLASSES:
                current_centers[c].append(center_xy(box))

    # ---------- PLANT (one-time) ----------
    if not planted:
        for c in WATCHED_CLASSES:
            for (x, y) in current_centers[c]:
                objects_by_class[c].append({"ref": (x, y), "miss": 0, "taken": False})
                print(f"Planted {r.names[c]} at ({x},{y})")
        planted = True  # plant everything visible once; or make this a keypress to control when to snapshot

    # ---------- MATCH & UPDATE (per object) ----------
    # We do a greedy nearest-neighbor assignment with gating (OFFSET).
    # That’s enough for “static should-not-move” monitoring.
    status_lines = []
    for c, planted_list in objects_by_class.items():
        # Copy detections we can consume as we match items one-by-one
        dets = current_centers[c][:]  # list[(x,y)]
        used = [False] * len(dets)

        for idx, obj in enumerate(planted_list):
            if obj["taken"]:
                #INSERT FUNCTION HERE#
                status_lines.append(f"{r.names[c]} #{idx}: TAKEN")
                continue

            rx, ry = obj["ref"]
            # find nearest unused detection within OFFSET
            best_j, best_d2 = -1, None
            for j, (x, y) in enumerate(dets):
                if used[j]:
                    continue
                dx, dy = x - rx, y - ry
                d2 = dx * dx + dy * dy
                if d2 <= OFFSET * OFFSET and (best_d2 is None or d2 < best_d2):
                    best_d2 = d2
                    best_j = j

            if best_j >= 0:
                # matched this planted item → it's present this frame
                obj["miss"] = 0
                used[best_j] = True
                status_lines.append(f"{r.names[c]} #{idx}: OK")
            else:
                # no match near the planted ref → miss
                obj["miss"] += 1
                if obj["miss"] >= MISS_LIMIT:
                    obj["taken"] = True
                    status_lines.append(f"{r.names[c]} #{idx}: TAKEN")
                else:
                    status_lines.append(f"{r.names[c]} #{idx}: MISSING {obj['miss']}/{MISS_LIMIT}")

    # ---------- DRAW ----------
    # Draw current detections
    if r.boxes is not None and len(r.boxes) > 0:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            c = cls_id(box)
            color = (0, 255, 0) if c in WATCHED_CLASSES else (128, 128, 128)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, r.names[c], (x1, max(0, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Draw planted refs + label
    for c, planted_list in objects_by_class.items():
        for idx, obj in enumerate(planted_list):
            rx, ry = obj["ref"]
            col = (0, 255, 0) if not obj["taken"] else (0, 0, 255)
            cv2.circle(frame, (rx, ry), 5, col, -1)
            cv2.putText(frame, f"{r.names[c]}#{idx}", (rx + 6, ry - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

    # Status text (first few lines)
    y0 = 24
    for s in status_lines[:12]:
        cv2.putText(frame, s, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        y0 += 24

    cv2.imshow("Static items: per-object match with misses", frame)
    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cv2.destroyAllWindows()
