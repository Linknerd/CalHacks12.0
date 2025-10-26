import cv2
from ultralytics import YOLO

PERSON     = 0
BACKPACK   = 24
HANDBAG    = 26
SUITCASE   = 28
LAPTOP     = 63
CELL_PHONE = 67

CLASS_FILTER = [PERSON, BACKPACK, HANDBAG, SUITCASE, LAPTOP, CELL_PHONE]
TRACKER      = "botsort.yaml"
MODEL_PATH   = "yolo11m.pt"

model = YOLO(MODEL_PATH)
initPosX = -100
initPosY = -100
started = False
# Iterate the tracker stream once (this opens and reads the webcam internally)
for r in model.track(
    source=0,                 # webcam index
    device="cuda:0",
    half= True,
    tracker=TRACKER,
    persist=True,             # keep IDs across frames
    stream=True,              # generator: yields one Results per frame
    conf = 0.3,
    classes=CLASS_FILTER,     # filter classes
    verbose=False
):
    frame = r.plot()  # annotated image with boxes/ids/labels
    trackedObjectID = -10
    if(not started):
        for box in r.boxes:
            if int(box.cls) == SUITCASE:
                trackedObjectID = int(box.id)
                started = True
                initPosX, initPosY, w, h = box.xywh[0].cpu().numpy().astype(int)
    idFound = False
    for box in r.boxes:
        if box.id == trackedObjectID:
            x,y,w,h = box.xywh[0].cpu().numpy().astype(int)
            if initPosX-5 < x < initPosX+5 and initPosY-5 < y < initPosY+5:
                pass
            else:
                print("OBJECT MOVED")

                

    cv2.imshow("YOLO11 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()