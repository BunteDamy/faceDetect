import os
from flask import Flask, render_template, request, redirect, url_for, send_file
import easyocr as ocr
import cv2
import numpy as np


app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ocr_motoru = ocr.Reader(['en', 'tr'])


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    

    file_name, file_ext = os.path.splitext(file.filename)
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)


    img = np.fromfile(file_path, np.uint8)
    img_cv2 = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    words = ocr_motoru.readtext(img_cv2)


    txt_path = os.path.join(app.config['UPLOAD_FOLDER'], 'DAMY.txt')
    with open(txt_path, 'w') as f:
        for word in words:
            f.write(f"{word[1]}\n")


    for word in words:
        cv2.rectangle(img_cv2,
                      (int(word[0][0][0]), int(word[0][0][1])),
                      (int(word[0][2][0]), int(word[0][2][1])),
                      (0, 0, 255), 2)


    img_output_name = f"{file_name}_ocr{file_ext}"
    img_output_path = os.path.join(app.config['UPLOAD_FOLDER'], img_output_name)
    cv2.imwrite(img_output_path, img_cv2)


    return render_template('result.html', image_file=img_output_name, txt_file=txt_path)


@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)
