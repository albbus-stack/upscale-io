from flask import Flask, flash, render_template, url_for, request, redirect, send_from_directory
import numpy as np
from PIL import Image
from ISR.models import RDN
from ISR.models import RRDN
import tensorflow as tf
import pathlib
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

IMAGES_FOLDER = 'tmp/' # '/tmp/' for Heroku
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
app.config['UPLOAD_FOLDER'] = IMAGES_FOLDER

final_filename = ''
img = None
patch_size = 100  # Increase this for better quality images

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Home page with upload
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
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img = Image.open(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            final_filename = filename
            return redirect('/processed')
    return render_template('home.html')

# Processing page where tensorflow acts
@app.route('/processed', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.form['submit'] == 'Go back':
            return redirect('/')
        elif request.form['submit'] == 'Download':
            return redirect('/display/' + final_filename)   
    else:
        lr_img = np.array(img) 
        model = 'noise-cancel'
        # You can try various models :
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
        return render_template('index.html', filename= final_filename)

# Processed image routing
@app.route('/display/<filename>')
def display_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run()
