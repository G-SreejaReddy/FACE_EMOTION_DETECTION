import cv2
from keras.models import model_from_json
import numpy as np

# Load the trained emotion detection model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Initialize the face detection classifier using Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

# Function to extract features from a face image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Initialize the webcam (0 is the default camera index)
webcam = cv2.VideoCapture(0)

# Check if the webcam is accessible
if not webcam.isOpened():
    print("Error: Could not access the webcam.")
    exit()  # Exit if webcam can't be accessed
else:
    print("Webcam is working correctly.")

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    # Capture a frame from the webcam
    i, im = webcam.read()

    # Check if the frame is valid
    if not i:
        print("Error: Failed to capture frame.")
        break  # Exit the loop if no frame is captured

    # Convert the frame to grayscale
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(im, 1.3, 5)

    try:
        # Process each detected face
        for (p, q, r, s) in faces:
            # Extract the region of interest (the face)
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)

            # Resize the face image to 48x48 (input size for the model)
            image = cv2.resize(image, (48, 48))

            # Extract features from the face image
            img = extract_features(image)

            # Make a prediction using the emotion detection model
            pred = model.predict(img)

            # Get the label for the predicted emotion
            prediction_label = labels[pred.argmax()]

            # Display the predicted emotion on the image
            cv2.putText(im, '% s' % (prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255))

        # Show the frame with the detected faces and predicted emotion
        cv2.imshow("Output", im)

        # Exit the loop if the 'Esc' key is pressed
        if cv2.waitKey(27) & 0xFF == ord('q'):  # 27 is for Esc key
            break

    except cv2.error as e:
        print(f"OpenCV error: {e}")
        pass

# Release the webcam and close all OpenCV windows
webcam.release()
cv2.destroyAllWindows()
