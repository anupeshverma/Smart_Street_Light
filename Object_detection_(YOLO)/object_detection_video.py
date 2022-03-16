import cv2
import numpy as np

yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []

with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]
print(classes)

cap = cv2.VideoCapture('sample_video.mp4')

while True:
    _, frame = cap.read()
    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

    yolo.setInput(blob)
    layer_names = yolo.getLayerNames()
    output_layer_name = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
    layer_output = yolo.forward(output_layer_name)

    class_ids = []
    confidences = []
    boxes = []
    for output in layer_output:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confi = (round(confidences[i], 2)) * 100

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(frame, f"{label} - {confi}%", (x, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
