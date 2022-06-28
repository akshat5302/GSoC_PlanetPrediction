import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from werkzeug.utils import secure_filename
import numpy as np



ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
IMAGE_SIZE = (256, 256)
UPLOAD_FOLDER = 'uploads'
model = load_model('model/my_model.h5')
Map = {
        0:"Earth",
        1:"Jupiter",
        2:"MakeMake",
        3:"Mars",
        4:"Mercury",
        5:"Moon",
        6:"Neptune",
        7:"Pluto",
        8:"Saturn",
        9:"Uranus",
        10:"Venus",
}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def predict(file):
    img  = load_img(file, target_size=IMAGE_SIZE)
    img = img_to_array(img)/255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    stmt= Map[np.argmax(prediction)]
    return stmt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def template_test():
    return render_template('home.html', label="", imagesource='file://null')


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    output = ""
    file_path = ""
    if request.method == 'POST':
        file = request.files['file']
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            output = predict(file_path)
    return render_template("home.html", label=output, imagesource=file_path)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

if __name__ == "__main__":
    app.run(debug=False, threaded=False)
