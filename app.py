from flask import Flask,jsonify,request,render_template,request
import  json
import os
import numpy as np
import keras.backend.tensorflow_backend as tb
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
app = Flask(__name__)

# draw each face separately
def draw_faces(filename, result_list):
	# load the image
	data = pyplot.imread(filename)
	# plot each face as a subplot
	for i in range(len(result_list)):
		# get coordinates
		x1, y1, width, height = result_list[i]['box']
		x2, y2 = x1 + width, y1 + height
		# define subplot
		pyplot.subplot(1, len(result_list), i+1)
		pyplot.axis('off')
		# plot face
		pyplot.imshow(data[y1:y2, x1:x2])
	# show the plot
	pyplot.show()
@app.route("/show_image",methods=['POST'])
def Hoome():
    img = request.files.get('data')
    # stream = BytesIO(img)
    # image = Image.open(stream).convert("RGBA")
    # stream.close()
    
    filename = img
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    faces = detector.detect_faces(pixels)
    # display faces on the original image
    draw_faces(filename, faces)

    return "succesfully abstract faces"
   
    
@app.route("/")
def Upload_image():
    return render_template('upload_image.html')


if __name__ == "__main__":
    app.run(debug=False,threaded=False)