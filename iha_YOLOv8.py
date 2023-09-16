import torch
from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator


tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'MOSSE', 'CSRT']
tracker_type = tracker_types[3]

if tracker_type == 'BOOSTING':
    tracker = cv2.legacy.TrackerBoosting.create()
if tracker_type == 'MIL':
    tracker = cv2.legacy.TrackerMIL.create()
if tracker_type == 'KCF' :
    tracker = cv2.legacy.TrackerKCF.create()
if tracker_type == 'TLD' :
    tracker = cv2.legacy.TrackerTLD.create()
if tracker_type == 'MEDIANFLOW' :
    tracker = cv2.legacy.TrackerMedianFlow.create()
if tracker_type == 'MOSSE' :
    tracker = cv2.legacy.TrackerMOSSE.create()
if tracker_type == 'CSRT' :
    tracker = cv2.legacy.TrackerCSRT.create()

model = YOLO('uav_yolov8.pt')  # load a custom model


# Predict with model (videos)
video_path = "videos_iha/ihavd_base.mp4"
video = cv2.VideoCapture(video_path)
ret, frame = video.read()

frame_height, frame_width = frame.shape[:2]
output = cv2.VideoWriter(f'{tracker_type}.avi',
                         cv2.VideoWriter_fourcc(*'XVID'), 60.0,
                         (frame_width, frame_height), True)
# bbox = 200,200,100,100 #Just the test if bbox properly created
bbox = 1,1,10,10
last = bbox
# ret = tracker.init(frame, bbox)
bbox = None
tracking = False
while True:
    ret, frame =video.read()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(img)
    for r in results:

        annotator = Annotator(frame)

        empty_tensor = torch.empty(0, 4)
        tensor_values = (r.boxes.xywh).clone().detach()

        if torch.equal(r.boxes.xywh, empty_tensor):
            cv2.putText(frame, "YOLO : OFF", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if tracking:
                tracker.init(frame,last)
                # Update the tracker to get the new bounding box position
                success, bbox = tracker.update(frame)

                if success:
                    # Draw the bounding box on the frame
                    x, y, w, h = [int(i) for i in last]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else:
                    tracking = False

            if bbox is not None:
                # Draw the last known bounding box (if available)
                x, y, w, h = [int(i) for i in last]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Display the frame
            cv2.imshow("YOLO V8 Detection", frame)


            # Manually adjust the bounding box by selecting a new one
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, last)
            tracking = True

        else:
            tracker = None
            bbox = None
            cv2.putText(frame, "YOLO : ON", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            bb_x = int(tensor_values[0, 0])
            bb_y = int(tensor_values[0, 1])
            bb_width = int(tensor_values[0, 2])
            bb_height = int(tensor_values[0, 3])

            bbox = (bb_x, bb_y, bb_width, bb_height)
            # print(f"Bbox in loop: {bbox}")
            last = (bb_x, bb_y, bb_width, bb_height)


            # print(f"X: {bb_x}")
            # print(f"Y: {bb_y}")
            # print(f"Width: {bb_width}")
            # print(f"Height: {bb_height}")


        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]  # get box coordinates in (top, left, bottom, right) format
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
            # print(b,c)

    frame = annotator.result()
    cv2.imshow('YOLO V8 Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

video.release()
cv2.destroyAllWindows()
