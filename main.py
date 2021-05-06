import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# creating a deep learning model from yolo weights and config file
net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Loading classes from coco.names
classes = []
with open("coco.names") as data:
    classes = [line.strip() for line in data.readlines()]
print("Number of classes avalible:{}".format(len(classes)))
print(classes)
colors = np.random.uniform(0, 255, size=(len(classes), 3))

layerNames = net.getLayerNames()
outputLayers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# loading image using opencv
image = cv.imread("test.jpeg")
image = cv.resize(image, None, fx=1, fy=1)
height, width, channels = image.shape

# preprocessing image
blob = cv.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

# detecting objects
net.setInput(blob)
outs = net.forward(outputLayers)

# print(outs)

# for b in blob:
#     for i, blobImage in enumerate(b):
#         cv.imshow(str(i), blobImage)



# variables for holding results
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:
            # extracting coordinates for drawing bounding box
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # calculating coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print("total number of objects detected:{0}".format(len(boxes)))

# removing redundant detection
indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# preprocessing image to use pillow package
imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
imagePillow = Image.fromarray(imageRGB)
draw = ImageDraw.Draw(imagePillow)
font = ImageFont.truetype("Roboto-Medium.ttf", 14)

# drawing bounding boxes using pillow
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = classes[class_ids[i]]
        r, g, b = colors[class_ids[i]]
        color = (int(r), int(g), int(b), 255)
        coord = [(x, y), (x + w, y + h)]
        draw.rectangle(coord, fill=None, outline=color, width=2)
        draw.text((x, y - 15), label, font=font, fill=color)

pillowProcessedImage = cv.cvtColor(np.array(imagePillow), cv.COLOR_RGB2BGR)

cv.imshow("Image", pillowProcessedImage)
cv.waitKey(0)
cv.destroyAllWindows()
