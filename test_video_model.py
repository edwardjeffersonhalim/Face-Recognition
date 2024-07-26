'''
This file can output facial recognition based on a real time video!
Needs work on the machine learning model.
'''

import cv2
import tensorflow as tf
import numpy as np
from PIL import Image, ExifTags

# Load model
graph_def = tf.compat.v1.GraphDef()
filename = "model.pb"
labels_filename = "labels.txt"

with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

labels = []
with open(labels_filename, 'rt') as lf:
    for l in lf:
        labels.append(l.strip())

def update_orientation(image):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = image._getexif()
        if exif is not None:
            orientation = exif[orientation]
            if orientation == 3:
                image = image.rotate(180, expand=True)
            elif orientation == 6:
                image = image.rotate(270, expand=True)
            elif orientation == 8:
                image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        pass
    return image

def preprocess_image(image):
    image = Image.fromarray(image)
    image = update_orientation(image)
    image = image.convert('RGB')
    image = np.array(image)
    image = cv2.resize(image, (224, 224)) 
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

# Initialize webcam
cap = cv2.VideoCapture(0)

with tf.compat.v1.Session() as sess:
    output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            face = frame[y:y+h, x:x+w]

            preprocessed_face = preprocess_image(face)

            predictions = sess.run(output_tensor, {'Placeholder:0': preprocessed_face})
            predictions = predictions[0]

            max_index = np.argmax(predictions)
            label = labels[max_index]
            probability = predictions[max_index] * 100

            text = f"{label}: {probability:.1f}%"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.imshow('Emotion Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

'''
Notes:
- Emotion recognition would not work if a person's head is tilted
- Emotion recognition sometimes would not work if a part of the face is covered
- Sad & surprised emotions dominated the emotion recognition
- Happy has a precision of 100% but just won't be recognized when we are actually smilling
'''