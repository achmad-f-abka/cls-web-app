"""Aplikasi yang meringkas teks."""

from flask import Flask, render_template, request 

import tensorflow as tf
import tensorflow_text as tf_text

def pre_process(text):
    # Pisahkan string berdasarkan baris menjadi list.
    lines = text.splitlines()
    
    cleaned_text = []
    for line in lines:
        # Menormalkan setiap string UTF-8.
        temp = tf_text.normalize_utf8(line, 'NFKD')
    
        # Mengubah huruf besar menjadi huruf kecil.
        temp = tf.strings.lower(temp)

        # Hapus teks di dalam tanda kurung, termasuk tanda kurungnya: (), [], dan {}.
        temp = tf.strings.regex_replace(temp, '\(.*?\)', '()')
        temp = tf.strings.regex_replace(temp, '\[.*?\]', '[]')
        temp = tf.strings.regex_replace(temp, '\{.*?\}', '{}')

        # Pertahankan spasi, huruf a hingga z, dan tanda baca tertentu.
        temp = tf.strings.regex_replace(temp, '[^ a-z0-9,.?!]', '')

        # Tambahkan spasi di sekitar tanda baca.
        temp = tf.strings.regex_replace(temp, '[,.?!]', r' \0 ')

        # Ganti banyak spasi dengan satu spasi.
        temp = tf.strings.regex_replace(temp, '\s+', ' ')

        # Hapus spasi pada bagian depan dan belakang.
        temp = tf.strings.strip(temp)

        # Menambahkan elemen ke akhir list.
        cleaned_text.append(temp)
    
    return cleaned_text[0]

cls_eng_ina = tf.saved_model.load('models/mwe-mcls')
cls_ina_eng = tf.saved_model.load('models/r-mwe-mcls')

# Membuat app object menggunakan Flask class.
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods = ['POST'])
def summarize():
    document = request.form['document']
    cls_type = request.form['cls-type']
    
    pre_processed_document = pre_process(document).numpy().decode('utf-8')
    
    output = 'None'
    if cls_type == '0':
        output = cls_eng_ina(pre_processed_document).numpy().decode('utf-8')
    
    if cls_type == '1':
        output = cls_ina_eng(pre_processed_document).numpy().decode('utf-8')
    
    return render_template('index.html', document=document, summary='Ringkasan: {}'.format(output))

if __name__ == '__main__':
    app.run(debug = True)
    # app.run()