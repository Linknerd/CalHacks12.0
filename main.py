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
trackedObjectID = -10
started = False
def getTrackID(box):
    if box.id is None:
        return -1
    else:
        return int(box.id)
    

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
    if(not started):
        for box in r.boxes:
            if int(box.cls) == SUITCASE:
                trackedObjectID = int(box.id)
                started = True
                initPosX, initPosY, w, h = box.xywh[0].cpu().numpy().astype(int)
                print(initPosX)
                print(initPosY)
                print("SUITCASE HAS BEEN PLANTED")
    idFound = False
    for box in r.boxes:
        if getTrackID(box) == trackedObjectID or int(box.cls) == SUITCASE:
            found = True
            x,y,w,h = box.xywh[0].cpu().numpy().astype(int)
            if(getTrackID(box) != trackedObjectID):
                print("Changed IDs")
                trackedObjectID = getTrackID(box)
            offset = 20
            if initPosX-offset < x < initPosX+offset and initPosY-offset < y < initPosY+offset:
                print("NOT MOVED")
            else:
                print("OBJECT MOVED")


    

                

    cv2.imshow("YOLO11 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()