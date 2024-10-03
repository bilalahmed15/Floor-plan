from flask import Flask, render_template, request, jsonify, send_file, Blueprint
import base64
import io
import PyPDF2
import json
import os
from PIL import Image
import uuid
import zipfile
import tempfile
from ultralytics import YOLO
from openpyxl import Workbook

app = Flask(__name__)

# Load YOLO model
model = YOLO('best.pt')

# Store annotations and annotation images in memory
annotations = {}
annotation_images = {}
@app.route('/')
def home():
    return render_template('first_page.html')

@app.route('/second_page')
def index():
    return render_template('second_page.html')
@app.route('/third_page')
def third_page():
    # Get all annotation images
    all_images = []
    for page_annotations in annotation_images.values():
        all_images.extend(page_annotations)
    
    return render_template('third_page.html', images=all_images)

@app.route('/upload', methods=['POST'])
def upload():
    global annotations, annotation_images
    file = request.files['file']
    if file:
        # Clear previous annotations and images when a new file is uploaded
        annotations = {}
        annotation_images = {}
        
        pdf_data = file.read()
        base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
        total_pages = get_total_pages(pdf_data)
        return jsonify({
            'base64_pdf': base64_pdf,
            'total_pages': total_pages
        })
    return jsonify({'error': 'No file uploaded'}), 400

@app.route('/annotations', methods=['POST'])
def save_annotations():
    data = request.json
    page_num = data['page']
    annotations[str(page_num)] = data['annotations']
    return jsonify({'status': 'success'})

@app.route('/annotations', methods=['GET'])
def get_annotations():
    page_num = request.args.get('page')
    return jsonify(annotations.get(str(page_num), []))

@app.route('/confirm_annotation', methods=['POST'])
def confirm_annotation():
    data = request.json
    image_data = data['image_data']
    page_num = data['page_num']
    x = data['x']
    y = data['y']
    width = data['width']
    height = data['height']
    
    annotation_id = str(uuid.uuid4())
    
    img = Image.open(io.BytesIO(base64.b64decode(image_data.split(',')[1])))
    
    # Create the static/annotation_images directory if it doesn't exist
    os.makedirs('static/annotation_images', exist_ok=True)
    
    img_path = f'annotation_images/{annotation_id}.png'
    full_path = os.path.join(app.static_folder, img_path)
    img.save(full_path)
    
    if page_num not in annotation_images:
        annotation_images[page_num] = []
    annotation_images[page_num].append({
        'id': annotation_id,
        'path': img_path,
        'x': x,
        'y': y,
        'width': width,
        'height': height
    })
    
    return jsonify({'status': 'success', 'id': annotation_id})

@app.route('/download_annotation/<annotation_id>', methods=['GET'])
def download_annotation(annotation_id):
    for page_annotations in annotation_images.values():
        for annotation in page_annotations:
            if annotation['id'] == annotation_id:
                return send_file(annotation['path'], as_attachment=True)
    return jsonify({'error': 'Annotation not found'}), 404

@app.route('/download_all_annotations', methods=['GET'])
def download_all_annotations():
    with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as temp_zip:
        with zipfile.ZipFile(temp_zip, 'w') as zipf:
            for page_num, page_annotations in annotation_images.items():
                for annotation in page_annotations:
                    zipf.write(annotation['path'], f"page_{page_num}_{annotation['id']}.png")
    
    return send_file(temp_zip.name, as_attachment=True, download_name='all_annotations.zip')

@app.route('/detect_symbols', methods=['POST'])
def detect_symbols():
    data = request.json
    image_path = data['image_path']
    room_id = data['room_id']
    
    full_path = os.path.join(app.static_folder, image_path)
    
    if not os.path.exists(full_path):
        return jsonify({'error': f"Image not found: {image_path}"}), 404
    
    try:
        results = model(full_path)
        
        object_counts = {}
        for r in results:
            for c in r.boxes.cls:
                class_name = model.names[int(c)]
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Store the results for later use in Excel generation
        if 'symbol_counts' not in annotation_images:
            annotation_images['symbol_counts'] = {}
        annotation_images['symbol_counts'][room_id] = object_counts
        
        return jsonify({'counts': object_counts})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_excel', methods=['GET'])
def download_excel():
    wb = Workbook()
    ws = wb.active
    ws.title = "Room Symbols"

    for room_id, counts in annotation_images.get('symbol_counts', {}).items():
        ws.append([f"Room {room_id}"])  # Add room heading
        ws.append(["Symbol", "Count"])  # Add headers for symbols

        for symbol, count in counts.items():
            ws.append([symbol, count])  # Append data for each symbol

        ws.append([])  # Add a blank row between rooms

    with tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx') as tmp:
        wb.save(tmp.name)
        tmp_name = tmp.name
    
    return send_file(tmp_name, as_attachment=True, download_name='room_symbols.xlsx')

def get_total_pages(pdf_data):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_data))
    return len(pdf_reader.pages)

if __name__ == '__main__':
    app.run(debug=True)
