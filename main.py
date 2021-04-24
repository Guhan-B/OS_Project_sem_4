import cv2 as cv

net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Loading classes from coco.names
classes = []
with open("coco.names") as data :
    classes = [line.strip() for line in data.readlines()]
print("Number of classes avalible:{}".format(len(classes)))
print(classes)

layerNames = net.getLayerNames()
outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

#loading image using opencv
image = cv.imread("room.jpg")
image = cv.resize(image, None, fx=0.3, fy=0.3)
cv.imshow("Image", image)

blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0,0,0), True, crop=False)

for b in blob:
    for i, blobImage in enumerate(b):
        cv.imshow(str(i), blobImage)

cv.waitKey(0)
cv.destroyAllWindows()


