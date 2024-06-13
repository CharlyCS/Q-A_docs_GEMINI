from flask import Flask, redirect, url_for,request,flash, render_template
import subprocess
import os

app = Flask(__name__, static_folder="public")
UPLOAD_FOLDER = 'Q_A_docs_GEMINI/docs'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/run-script', methods=['GET', 'POST'])
def run_script_route():
    if request.method == 'POST':
        variable = request.form.get('variable')  # Get the variable from the POST request
        result = subprocess.run(['python', 'Q_A_docs_GEMINI/app.py', variable], capture_output=True, text=True)
    else:
        result = subprocess.run(['python', 'Q_A_docs_GEMINI/app.py'], capture_output=True, text=True)
    return result.stdout

@app.route('/process-pdf', methods=['POST'])
def process_pdf():
    if 'pdf' not in request.files:
        return 'No file part', 400

    file = request.files['pdf']

    if file.filename == '':
        return 'No selected file', 400

    if file and file.filename.endswith('.pdf'):
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        return 'Valid file type', 200
    else:
        return 'Invalid file type', 400

if __name__ == "__main__":
    app.run(debug=True)