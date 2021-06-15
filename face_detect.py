import cv2
import sys
# Get user supplied values
imagePath = sys.argv[1]
cascPath = "haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)

src = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
# set a new width in pixels
if (max(image.shape)>3000):
    scale_percent = 25
elif (max(image.shape)>1000):
    scale_percent = 50
elif (max(image.shape)>500):
    scale_percent = 70
else:
    scale_percent = 100

#calculate the 50 percent of original dimensions
width = int(src.shape[1] * scale_percent / 100)
height = int(src.shape[0] * scale_percent / 100)

# dsize
dsize = (width, height)

# resize image
resizeImg = cv2.resize(src, dsize)

gray = cv2.cvtColor(resizeImg, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=2,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE #cv2.cv.CV_HAAR_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(resizeImg, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow("Faces found", resizeImg)
cv2.waitKey(0)

#use face_detect
#python face_detect.py photo haarcascade_frontface_default.xml