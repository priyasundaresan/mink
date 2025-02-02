from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from io import BytesIO
from PIL import Image
import base64

from interactive_scripts.dataset_recorder import ActMode

app = Flask(__name__)
annotations = []
frames = []
waypoints = []

def load_frames():
    global frames
    frames.clear()
    for fn in sorted(os.listdir('dev1')):
        if 'npz' in fn:
            demo = np.load(os.path.join('dev1', fn), allow_pickle=True)['arr_0']
            for step in demo:
                frames.append(step['obs']['viewer_image'])
            break

load_frames()

@app.route('/')
def index():
    return render_template('index.html', frame_count=len(frames))

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    global annotations, waypoints
    data = request.json
    annotations.append(data)
    if data['type'] == 'waypoint':
        waypoints.append(data['frame'])
    return jsonify({"status": "success", "annotations": annotations, "waypoints": waypoints})

@app.route('/frames/<int:frame_id>')
def get_frame(frame_id):
    if 0 <= frame_id < len(frames):
        img = frames[frame_id]
        img_pil = Image.fromarray(img.astype('uint8'))
        buffered = BytesIO()
        img_pil.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return jsonify({"image": f"data:image/png;base64,{img_str}"})
    return jsonify({"error": "Frame not found"}), 404

@app.route('/get_waypoints', methods=['GET'])
def get_waypoints():
    return jsonify({"waypoints": waypoints})

@app.route('/export_annotations', methods=['GET'])
def export_annotations():
    global frames, waypoints
    annotations_list = ['dense'] * len(frames)
    
    if waypoints:
        for i in range(len(waypoints)):
            if i == 0:
                start = 0
            else:
                start = waypoints[i - 1]
            end = waypoints[i]
            for j in range(start, end):
                annotations_list[j] = 'interpolate'
        for waypoint in waypoints:
            annotations_list[waypoint] = 'waypoint'
    
    return jsonify({"annotations": annotations_list})

if __name__ == '__main__':
    app.run(debug=True)

