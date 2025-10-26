import cv2
from ultralytics import YOLO

PERSON     = 0
BACKPACK   = 24
HANDBAG    = 26
SUITCASE   = 28
LAPTOP     = 63
CELL_PHONE = 67

CLASS_FILTER = [PERSON, BACKPACK, SUITCASE, LAPTOP, CELL_PHONE]
TRACKER      = "botsort.yaml"
MODEL_PATH   = "yolo11m.pt"

model = YOLO(MODEL_PATH)
initPosX = -100
initPosY = -100
suitcaseMap = {}
phoneMap = {}
backpackMap = {}
laptopMap = {}

def getMap(n):
    if n == SUITCASE:
        return suitcaseMap
    elif n == CELL_PHONE:
        return phoneMap
    elif n == LAPTOP:
        return laptopMap
    elif n == BACKPACK:
        return backpackMap



trackedObjectID = -10
started = False
def getTrackID(box):
    if box.id is None:
        return -1
    else:
        return int(box.id)
    

# Iterate the tracker stream once (this opens and reads the webcam internally)
for r in model.track(
    source=1,                 # webcam index
    device="cuda:0",
    half= True,
    tracker=TRACKER,
    persist=True,             # keep IDs across frames
    stream=True,              # generator: yields one Results per frame
    conf = 0.4,
    classes=CLASS_FILTER,     # filter classes
    verbose=False
):
    laptopIDX = 0
    backpackIDX = 0
    phoneIDX = 0
    suitcaseIDX = 0
    frame = r.plot()  # annotated image with boxes/ids/labels
    if(not started):
        for box in r.boxes:
            val = int(box.cls) 
            if val != PERSON:
                trackedObjectID = int(box.id)
                started = True
                initPosX, initPosY, w, h = box.xywh[0].cpu().numpy().astype(int)
                print(initPosX)
                print(initPosY)
                objStr = ""
                if val == SUITCASE:
                    suitcaseMap[suitcaseIDX] = [initPosX,initPosY,SUITCASE]
                    objStr = "SUITCASE"
                    suitcaseIDX+=1
                elif val == LAPTOP:
                    laptopMap[laptopIDX] = [initPosX, initPosY, LAPTOP]
                    objStr = "LAPTOP"
                    laptopIDX += 1
                elif val == CELL_PHONE:
                    phoneMap[phoneIDX] = [initPosX, initPosY, CELL_PHONE]
                    objStr = "CELL PHONE"
                    phoneIDX+=1
                elif val == BACKPACK:
                    backpackMap[backpackIDX] = [initPosX, initPosY, BACKPACK]
                    objStr = "CELL PHONE"
                    backpackIDX+=1
                print(objStr + " HAS BEEN PLANTED")
    idFound = False
    for box in r.boxes:
        classes = int(box.cls)
        
        idFound = True
        x,y,w,h = box.xywh[0].cpu().numpy().astype(int)
        offset = 20
        if initPosX-offset < x < initPosX+offset and initPosY-offset < y < initPosY+offset and 0 <= x <= 640 and 0<=y<=480:
            print("NOT MOVED")
        else:
            print("OBJECT MOVED")
    if not idFound:
        print("OBJECT MOVED")
    

                

    cv2.imshow("YOLO11 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()