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
suitcaseMap = {id0: {"ref": (x, y), "cls": SUITCASE}}
phoneMap = {id0: {"ref": (x, y), "cls": CELL_PHONE}}
backpackMap = {id0: {"ref": (x, y), "cls": BACKPACK}}
laptopMap ={id0: {"ref": (x, y), "cls": LAPTOP}}

def getMap(n):
    if n == SUITCASE:
        return suitcaseMap
    elif n == CELL_PHONE:
        return phoneMap
    elif n == LAPTOP:
        return laptopMap
    elif n == BACKPACK:
        return backpackMap
    else:
        return {}



trackedObjectID = -10
started = False
def getTrackID(box):
    if box.id is None:
        return -1
    else:
        return int(box.id)
laptopIDX = 0
backpackIDX = 0
phoneIDX = 0
suitcaseIDX = 0
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
    frame = r.plot()  # annotated image with boxes/ids/labels
    if(not started):
        for box in r.boxes:
            val = int(box.cls) 
            if val != PERSON:
                started = True
                initPosX, initPosY, w, h = box.xywh[0].cpu().numpy().astype(int)
                print(initPosX)
                print(initPosY)
                objStr = ""
                if val == SUITCASE:
                    suitcaseMap.append({"ref": (initPosX, initPosY), "cls": SUITCASE})
                    suitcaseIDX += 1
                    objStr = "SUITCASE"
                elif val == LAPTOP:
                    laptopMap.append({"ref": (initPosX, initPosY), "cls": LAPTOP})
                    objStr = "LAPTOP"
                    laptopIDX += 1
                elif val == CELL_PHONE:
                    phoneMap.append({"ref": (initPosX, initPosY), "cls": CELL_PHONE})
                    objStr = "CELL PHONE"
                    phoneIDX+=1
                elif val == BACKPACK:
                    backpackMap.append({"ref": (initPosX, initPosY), "cls": CELL_PHONE} )
                    objStr = "BACKPACK"
                    backpackIDX+=1
                print(objStr + " HAS BEEN PLANTED")
    found = False
    if(started):
        for box in r.boxes:
            classes = cls_id = int(box.cls[0].item())
            iMap = getMap(classes)        
            for idx in iMap:
                refx,refy = iMap[idx]["ref"] 
                offset = 20
                if refx-offset < x < refx+offset and refy-offset < y < refy+offset and 0 <= x <= 640 and 0<=y<=480:
                    found = True
        if not found:
            print("OBJECT MOVED")
    

                

    cv2.imshow("YOLO11 Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()