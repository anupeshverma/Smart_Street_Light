import cv2
import numpy as np

# load Yolo files   (dnn == deep neural network)
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# import names and put it in a form of a list
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()
print(f"Length of class: {len(classes)}")

# Read image
img = cv2.imread('sample_image.jfif')
height, width, channels = img.shape

# Convert image from BGR to RGB
blob = cv2.dnn.blobFromImage(img, 1 / 255, size=(640, 640), mean=(0, 0, 0), swapRB=True,
                             crop=False)  # swapRB == swap red blue

print(f"Blob Shape: {blob.shape}")

yolo.setInput(blob)  # set blob image as input
layer_names = yolo.getLayerNames()
output_layer_name = [layer_names[i[0] - 1] for i in yolo.getUnconnectedOutLayers()]
layer_output = yolo.forward(output_layer_name)

# Capture bounding boxes
boxes = []
confidences = []  # By what confidence value is being predicted
class_ids = []  # Includes what we have imported

for output in layer_output:
    for detection in output:
        # Score return an array of size 18, first four contains position of our boxes rest of give confidence or probaility of that particular characteter in the box
        """
        By detection we are going to get a list
        Through score we get a place or placeholder in array where is tha maximum probability of that image
        class_ids give just whatever the values or percentage of that probability
        """
        score = detection[5:]  # Exclude first four boxes
        class_id = np.argmax(score)
        confidence = score[class_id]

        # If confidence is greater than threshold then we'll extract our features
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # For finding corners
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(f"Total Boxes = {len(boxes)}")

indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

font = cv2.FONT_HERSHEY_SIMPLEX

for i in range(len(boxes)):
    if i in indexes:
        x1, y1, w1, h1 = boxes[i]

        label = str(classes[class_ids[i]])
        confi = (round(confidences[i], 2)) * 100

        cv2.rectangle(img, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
        cv2.putText(img, f"{label} - {confi}%", (x1, y1 + 20), font, 0.5, (0, 25, 255), 2)

cv2.imshow("Image", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
