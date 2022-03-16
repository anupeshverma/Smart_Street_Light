import numpy as np
from matplotlib import pyplot as plt
import cv2

cap = cv2.VideoCapture('motion_detection_sample_video.mp4')

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

out = cv2.VideoWriter("motion_detection_output_video.avi", fourcc, 5.0, (1280, 720))

_, frame1 = cap.read()
_, frame2 = cap.read()

while True:

    difference = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, hierarchy = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)
    for contour in contours:
        (x, y, width, height) = cv2.boundingRect(contour)

        if cv2.contourArea(contour) < 900:
            continue

        cv2.rectangle(frame1, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(frame1, "Status:{}".format("Movement"), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 2), 2)

    """
    titles = ['Diff_Video', 'Gray_Video', 'Blur_Video', 'Thresh_Video', 'Dilated_Video', 'Final_Video']
    videos = [difference, gray, blur, thresh, dilated, frame1]
    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.title(titles[i])
        plt.imshow(videos[i], 'gray')
    plt.show()
    """

    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    
    cv2.imshow('Video', frame1)
    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:
        break

cap.release()
cv2.destroyAllWindows()
