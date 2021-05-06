import cv2 as cv
import numpy as np
from PIL import Image, ImageDraw, ImageFont

class Utility:
	def __init__(self):
		self.classes = []
		with open("coco.names") as data:
			for line in data.readlines():
				self.classes.append(line.strip())
		self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

class NeuralNetwork:
	def __init__(self):
		self.net = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
		self.layerNames = self.net.getLayerNames()
		self.outputLayers = [self.layerNames[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

	def predict(self, blob):
		self.net.setInput(blob)
		results = self.net.forward(self.outputLayers)
		return results

class ImageObjectDetector:
	def __init__(self, imagePath):
		self.imagePath = imagePath
		self.image = cv.imread(imagePath)
		self.resizedImage = cv.resize(self.image, None, fx=1, fy=1)
		self.height, self.width, _ = self.resizedImage.shape
		self.boxes = []
		self.confidences = []
		self.class_ids = []
		self.uniqueDetections = []

	def detect(self, model):
		blob = cv.dnn.blobFromImage(self.resizedImage, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
		outs = model.predict(blob)

		for out in outs:
			for detection in out:
				scores = detection[5:]
				class_id = np.argmax(scores)
				confidence = scores[class_id]

				if confidence > 0.5:
					center_x = int(detection[0] * self.width)
					center_y = int(detection[1] * self.height)

					w = int(detection[2] * self.width)
					h = int(detection[3] * self.height)

					x = int(center_x - w / 2)
					y = int(center_y - h / 2)

					self.boxes.append([x, y, w, h])
					self.confidences.append(float(confidence))
					self.class_ids.append(class_id)

		self.uniqueDetections = cv.dnn.NMSBoxes(self.boxes, self.confidences, 0.5, 0.4)


	def drawBoundingBox(self,classes,class_colors):
		imageRGB = cv.cvtColor(self.resizedImage, cv.COLOR_BGR2RGB)
		imagePillow = Image.fromarray(imageRGB)
		draw = ImageDraw.Draw(imagePillow)
		font = ImageFont.truetype("Roboto-Medium.ttf", 14)

		for i in range(len(self.boxes)):
			if i in self.uniqueDetections:
				x, y, w, h = self.boxes[i]
				label = classes[self.class_ids[i]]
				r, g, b = class_colors[self.class_ids[i]]
				color = (int(r), int(g), int(b), 255)
				coord = [(x, y), (x + w, y + h)]
				draw.rectangle(coord, fill=None, outline=color, width=2)
				draw.text((x, y - 15), label, font=font, fill=color)

		self.outputImage = cv.cvtColor(np.array(imagePillow), cv.COLOR_RGB2BGR)

	def showOutput(self):
		cv.imshow("Result", self.outputImage)
		cv.waitKey(0)
		cv.destroyAllWindows()

utils = Utility()
detector = ImageObjectDetector("test.jpeg")
model = NeuralNetwork()

detector.detect(model)
detector.drawBoundingBox(utils.classes,utils.colors)
detector.showOutput()