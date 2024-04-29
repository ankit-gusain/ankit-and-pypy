import os  # Importing operating system module for file operations
import cvzone  # Importing cvzone library for computer vision tasks
import cv2  # Importing OpenCV library
from cvzone.PoseModule import PoseDetector  # Importing PoseDetector class from cvzone

# Get screen resolution
screen_width = 1920
screen_height = 1080

# Open the capture device (webcam)
cap = cv2.VideoCapture(0)

# Set capture dimensions to match screen resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)

# Create a detector object for detecting poses
detector = PoseDetector()

# Path to folder containing shirt images
shirtFolderPath = "Resources/Shirts"

# List all shirt image files in the folder
listShirts = os.listdir(shirtFolderPath)

# Ratio used for resizing shirt images
fixedRatio = 262 / 190  # widthOfShirt / widthOfPoint11to12

# Aspect ratio of shirt images
shirtRatioHeightWidth = 581 / 440

# Initialize variables for controlling shirt selection
imageNumber = 0
imgButtonRight = cv2.imread("Resources/button.png", cv2.IMREAD_UNCHANGED)
imgButtonLeft = cv2.flip(imgButtonRight, 1)
counterRight = 0
counterLeft = 0
selectionSpeed = 10

# Create a window and set to full screen
cv2.namedWindow("Image", cv2.WINDOW_NORMAL)  # Create a window with the name "Image"
cv2.setWindowProperty("Image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)  # Set the window to full screen


# Main loop for capturing and processing video frames
while True:
    # Capture a frame from the webcam
    success, img = cap.read()

    # Detect poses in the captured frame
    img = detector.findPose(img)

    # Find landmarks and bounding box information in the detected poses
    lmList, bboxInfo = detector.findPosition(img, bboxWithHands=False, draw=False)

    # If landmarks are detected
    if lmList:
        # Get landmarks for left shoulder and right shoulder
        lm11 = lmList[11][1:3]
        lm12 = lmList[12][1:3]

        # Load the shirt image corresponding to the current selection
        imgShirt = cv2.imread(os.path.join(shirtFolderPath, listShirts[imageNumber]), cv2.IMREAD_UNCHANGED)

        # Calculate the width of the shirt image based on shoulder landmarks
        widthOfShirt = max(int((lm11[0] - lm12[0]) * fixedRatio), 1)  # Ensure widthOfShirt is at least 1

        # Resize the shirt image
        imgShirt = cv2.resize(imgShirt, (widthOfShirt, int(widthOfShirt * shirtRatioHeightWidth)))

        # Calculate offset for placing shirt image correctly on the pose
        currentScale = (lm11[0] - lm12[0]) / 190
        offset = int(44 * currentScale), int(48 * currentScale)

        # To Overlay the shirt image on the pose
        try:
            img = cvzone.overlayPNG(img, imgShirt, (lm12[0] - offset[0], lm12[1] - offset[1]))
        except:
            pass

        # Overlay navigation buttons on the frame
        img = cvzone.overlayPNG(img, imgButtonRight, (1074, 293))
        img = cvzone.overlayPNG(img, imgButtonLeft, (72, 293))

        # Check if right navigation gesture is detected
        if lmList[16][1] < 300:
            counterRight += 1
            cv2.ellipse(img, (139, 360), (66, 66), 0, 0,
                        counterRight * selectionSpeed, (0, 255, 0), 20)
            if counterRight * selectionSpeed > 360:
                counterRight = 0
                if imageNumber < len(listShirts) - 1:
                    imageNumber += 1
        # Check if left navigation gesture is detected
        elif lmList[15][1] > 900:
            counterLeft += 1
            cv2.ellipse(img, (1138, 360), (66, 66), 0, 0,
                        counterLeft * selectionSpeed, (0, 255, 0), 20)
            if counterLeft * selectionSpeed > 360:
                counterLeft = 0
                if imageNumber > 0:
                    imageNumber -= 1
        else:
            counterRight = 0
            counterLeft = 0

    # Display the processed frame
    cv2.imshow("Image", img)

    # Check for user input
    cv2.waitKey(1)

