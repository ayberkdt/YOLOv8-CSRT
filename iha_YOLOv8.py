import torch
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator



# model = YOLO('yolov8n.pt')  # load an official model

model = YOLO('uav_yolov8.pt')  # load a custom model


# Predict with model (videos)
video_path = "videos_iha/ihavd_base.mp4"
cap = cv2.VideoCapture(video_path)

ret = True
last = None
while True:
    ret, frame =cap.read()

    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = model.predict(img)
    # Extract boxes, scores, and class labels
    boxes = results[0].boxes.xyxy  # xyxy format
    scores = results[0].boxes.conf  # confidence scores
    print(boxes,scores)
    annotator = Annotator(frame)
    empty_tensor = torch.empty(0, 4)
    tensor_values = (results[0].boxes.xyxy).clone().detach()

    #If YOLO can detect loop
    if torch.equal(results[0].boxes.xywh, empty_tensor) == False:
        has = max(scores)

        for a in range(len(results[0].boxes.conf)):
            if results[0].boxes.conf[a] == has:
                last_yolo = [int(i) for i in results[0].boxes.xyxy[a]]
                last = [int(i) for i in results[0].boxes.xywh[a]]
                print(last)

            annotator.box_label(last_yolo, "uav")

    #If YOLO cannot detect loop
    elif torch.equal(results[0].boxes.xywh, empty_tensor) and last != None:
        print("Tracking") #Tracking başlayacak (last değeri trackerın ilk bboxu olacak)


    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()
