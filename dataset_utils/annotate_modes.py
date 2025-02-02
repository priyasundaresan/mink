import os
import threading
import time
from flask import Flask, render_template, request, jsonify
from werkzeug.serving import make_server
import numpy as np
from io import BytesIO
from PIL import Image
import base64

from interactive_scripts.dataset_recorder import ActMode

app = Flask(__name__)
annotations = []
frames = []
waypoints = []
server = None
demo_name = ""

def load_demo(episode_fn):
    global demo_name
    demo = np.load(episode_fn, allow_pickle=True)['arr_0']
    demo_name = episode_fn
    return demo

def load_frames(demo):
    global frames
    frames.clear()
    for step in demo:
        frames.append(step['obs']['viewer_image'])

def relabel_demo(demo, annotations_list):
    waypoint_idx = -1
    waypoint_idxs = waypoints
    curr_waypoint_step = 0

    for t, step in enumerate(demo):
        if t == curr_waypoint_step and len(waypoint_idxs):
            waypoint_action = list(demo)[waypoint_idxs[0]]['action']
            step['action'] = waypoint_action
            curr_waypoint_step = waypoint_idxs.pop(0)
            waypoint_idx += 1
        step['mode'] = annotations_list[t]
        step['waypoint_idx'] = waypoint_idx
        print(step['waypoint_idx'], step['mode'], step['action'])
    return demo


@app.route('/')
def index():
    return render_template('index.html', frame_count=len(frames), demo_name=demo_name)

@app.route('/get_demo_name', methods=['GET'])
def get_demo_name():
    return jsonify({"demo_name": demo_name})

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

def get_annotations():
    global frames, waypoints
    annotations_list = [ActMode.Dense for i in range(len(frames))]

    if waypoints:
        waypoints.sort()
        prev_waypoint = 0
        for i in range(len(waypoints)):
            start = prev_waypoint
            end = waypoints[i]
            annotations_list[start] = ActMode.Waypoint
            if start < end:
                for j in range(start + 1, end):
                    annotations_list[j] = ActMode.Interpolate
            prev_waypoint = end

    return annotations_list

def run_flask():
    global server
    server = make_server('127.0.0.1', 5000, app)
    server.serve_forever()

def stop_flask():
    global server
    if server:
        server.shutdown()
        print("Flask server stopped.")

if __name__ == '__main__':
    demo_dir = 'dev1'
    if not os.path.exists('dev1_relabeled'):
        os.mkdir('dev1_relabeled')
    
    print('Go to http://127.0.0.1:5000')
    
    for fn in sorted(os.listdir(demo_dir)):
        if 'npz' in fn:
            episode_fn = os.path.join(demo_dir, fn)
            print('Annotating:', episode_fn)
            demo = load_demo(episode_fn)
            load_frames(demo)
        
            # Start Flask in a separate thread
            flask_thread = threading.Thread(target=run_flask)
            flask_thread.start()
    
            # Wait for user interactions
            input("Press Enter in terminal when done annotating...")
        
            # Get annotations after interactions
            annotations_result = get_annotations()
            demo_relabeled = relabel_demo(demo, annotations_result)
            np.savez(episode_fn.replace('dev1', 'dev1_relabeled'), demo_relabeled)
    
            # Stop Flask before moving to the next demo
            stop_flask()
            flask_thread.join()
