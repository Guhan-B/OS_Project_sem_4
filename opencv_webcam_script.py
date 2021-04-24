import cv2 as cv

# creating neuralnet with opencv dnn and yolo model
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")

# here the number specifies the video source => 0 for built in webcam
webcamInput = cv.VideoCapture(0)
webcamInput.set(3,550)
webcamInput.set(4,550)
webcamInput.set(10,100)

while True:
    Success, frameImage = webcamInput.read()
    blob = cv.dnn.blobFromImage(frameImage, 0.00392, (416, 416), (0,0,0), True, crop=False)

    cv.imshow("Video", frameImage)

    for b in blob:
        for i, blobFrameImage in enumerate(b):
            cv.imshow(str(i), blobFrameImage)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break