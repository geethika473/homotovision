from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = '17a99bc413cecd16e981884714eec79a'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load model
model = load_model("Blood Cell.h5")
CLASS_LABELS = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def predict_image_class(image_path):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    img_preprocessed = preprocess_input(img_resized.reshape((1, 224, 224, 3)))
    predictions = model.predict(img_preprocessed)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    predicted_class_label = CLASS_LABELS[predicted_class_idx]
    confidence = float(np.max(predictions))
    return predicted_class_label, confidence, img_rgb


@app.route('/')
def home():
    return render_template("index.html")


@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/contact')
def contact():
    return render_template("contact.html")


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected', 'error')
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Predict
            pred_class, confidence, img_rgb = predict_image_class(filepath)

            # Save processed image
            processed_filename = f"processed_{filename}"
            processed_path = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            cv2.imwrite(processed_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            return render_template("predict.html",
                                   prediction=pred_class,
                                   confidence=confidence,
                                   image_path=processed_path)

    return render_template("predict.html")


if __name__ == '__main__':
    app.run(debug=True, port=2222)