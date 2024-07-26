import tensorflow as tf
import os
from PIL import Image, ExifTags
import numpy as np
import cv2

graph_def = tf.compat.v1.GraphDef()
labels = []

filename = "model.pb"
labels_filename = "labels.txt"

with tf.io.gfile.GFile(filename, 'rb') as f:
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')

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

imageFile = "/Users/edwardjefferson/Downloads/images/validation/happy/187.jpg"
image = Image.open(imageFile)
image = update_orientation(image)

image = image.convert('RGB')

image = np.array(image)

input_size = (224, 224)
image = cv2.resize(image, input_size)
image = image.astype(np.float32)
image = np.expand_dims(image, axis=0)

with tf.compat.v1.Session() as sess:
    output_tensor = sess.graph.get_tensor_by_name('model_outputs:0')
    predictions = sess.run(output_tensor, {'Placeholder:0': image})

predictions = predictions[0]
output = "Tag\nProbability\n"
for label, prob in zip(labels, predictions):
    output += f"{label}\n{prob * 100:.1f}%\n"

print(output)
