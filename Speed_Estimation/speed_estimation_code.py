import cv2
import dlib
import time
import math

carCascade = cv2.CascadeClassifier('vehicle.xml')
video = cv2.VideoCapture('speed_estimate_sample_video.mp4')

WIDTH = 1280
HEIGHT = 720


def estimateSpeed(location1, location2):
    dist_of_pixels = math.sqrt(math.pow(location2[0] - location1[0], 2) + math.pow(location2[1] - location1[1], 2))
    # ppm = location2[2] / carWidht
    pixel_per_meter = 8.8
    dist_in_meters = dist_of_pixels / pixel_per_meter
    fps = 20
    speed = dist_in_meters * fps * 3.6  # because 3600 seconds in one hour
    return speed


def trackMultipleObjects():
    rectangleColor = (0, 255, 0)
    frameCounter = 0
    currentCarID = 0
    fps = 0

    carTracker = {}
    carNumbers = {}
    carLocation1 = {}
    carLocation2 = {}
    speed = [None] * 1000  # To take 1000 inputs at a time

    out = cv2.VideoWriter('speed_estimate_output_video.mp4', cv2.VideoWriter_fourcc(
        'M', 'J', 'P', 'G'), 10, (WIDTH, HEIGHT))

    while True:
        start_time = time.time()
        _, frame = video.read()

        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        resultImage = frame.copy()

        frameCounter = frameCounter + 1
        carIDtoDelete = []

        for carID in carTracker.keys():
            trackingQuality = carTracker[carID].update(frame)

            if trackingQuality < 7:
                carIDtoDelete.append(carID)

        for carID in carIDtoDelete:
            #   print("Removing carID " + str(carID) + ' from list of trackers. ')
            carTracker.pop(carID, None)
            #   print("Removing carID " + str(carID) + ' previous location. ')
            carLocation1.pop(carID, None)
            #   print("Removing carID " + str(carID) + ' current location. ')
            carLocation2.pop(carID, None)

        if not (frameCounter % 10):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cars = carCascade.detectMultiScale(gray, 1.2, 20, 28, (28, 28))

            for (_x, _y, _w, _h) in cars:
                x = int(_x)
                y = int(_y)
                w = int(_w)
                h = int(_h)

                x_bar = x + w / 2
                y_bar = y + h / 2

                matchCarID = None

                for carID in carTracker.keys():
                    trackedPosition = carTracker[carID].get_position()

                    t_x = int(trackedPosition.left())
                    t_y = int(trackedPosition.top())
                    t_w = int(trackedPosition.width())
                    t_h = int(trackedPosition.height())

                    t_x_bar = t_x + t_w / 2
                    t_y_bar = t_y + t_h / 2

                    if ((t_x <= x_bar <= (t_x + t_w)) and (t_y <= y_bar <= (t_y + t_h)) and (
                            x <= t_x_bar <= (x + w)) and (y <= t_y_bar <= (y + h))):
                        matchCarID = carID

                if matchCarID is None:
                    print(' Creating new tracker' + str(currentCarID))

                    tracker = dlib.correlation_tracker()
                    tracker.start_track(
                        frame, dlib.rectangle(x, y, x + w, y + h))

                    carTracker[currentCarID] = tracker
                    carLocation1[currentCarID] = [x, y, w, h]

                    currentCarID = currentCarID + 1

        for carID in carTracker.keys():
            trackedPosition = carTracker[carID].get_position()

            t_x = int(trackedPosition.left())
            t_y = int(trackedPosition.top())
            t_w = int(trackedPosition.width())
            t_h = int(trackedPosition.height())

            cv2.rectangle(resultImage, (t_x, t_y),
                          (t_x + t_w, t_y + t_h), rectangleColor, 4)

            carLocation2[carID] = [t_x, t_y, t_w, t_h]

        end_time = time.time()

        if not (end_time == start_time):
            fps = 1.0 / (end_time - start_time)

        for i in carLocation1.keys():
            if frameCounter % 1 == 0:
                [x1, y1, w1, h1] = carLocation1[i]
                [x2, y2, w2, h2] = carLocation2[i]

                carLocation1[i] = [x2, y2, w2, h2]

                if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                    if (speed[i] == None or speed[i] == 0) and y1 >= 275 and y1 <= 285:
                        speed[i] = estimateSpeed(
                            [x1, y1, w1, h1], [x1, y2, w2, h2])

                    if speed[i] != None and y1 >= 180:
                        cv2.putText(resultImage, str(int(speed[i])) + "km/h", (int(
                            x1 + w1 / 2), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 100), 2)

        cv2.imshow('result', resultImage)

        out.write(resultImage)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    out.release()


if __name__ == '__main__':
    trackMultipleObjects()
