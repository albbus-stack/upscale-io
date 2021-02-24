from flask import Flask, flash, render_template, url_for, request, redirect
import numpy as np
from PIL import Image
from ISR.models import RDN
from ISR.models import RRDN
import tensorflow as tf
import pathlib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

IMAGES_FOLDER = os.path.join('static', 'images')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

final_filename = ''
img = None
patch_size = 250

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    global final_filename
    global img
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            final_filename = filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect('/processed')
    return render_template('home.html')

# home page where tensorflow acts
@app.route('/processed', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        return redirect('/')
    else:
        lr_img = np.array(img) 
        model = 'noise-cancel'
        # RDN: psnr-large, psnr-small, noise-cancel
        # RRDN: gans
        if(model == 'noise-cancel'):
            rdn = RDN(weights='noise-cancel')
        elif(model == 'psnr-large'):
            rdn = RDN(weights='psnr-large')
        elif(model == 'psnr-small'):
            rdn = RDN(weights='psnr-small')
        elif(model == 'gans'):
            rdn = RRDN(weights='gans')  

        sr_img = rdn.predict(lr_img, by_patch_of_size=patch_size)
        final_im = Image.fromarray(sr_img)
        final_im.save(os.path.join(app.config['UPLOAD_FOLDER'], final_filename))
        full_filename = os.path.join(app.config['UPLOAD_FOLDER'], final_filename)
        return render_template('index.html', image = final_filename)

if __name__ == '__main__':
    app.run()
