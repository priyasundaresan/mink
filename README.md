# Mobile-SPHINX

Extension of _What's the Move? Hybrid Imitation Learning via Salient Points_ (SPHINX) to mobile manipulation setting.

[![Paper](https://img.shields.io/badge/Paper-%20%F0%9F%93%84-blue)](https://sphinx-manip.github.io/assets/sphinx.pdf)
[![Website](https://img.shields.io/badge/Website-%F0%9F%8C%90-orange)](https://sphinx-manip.github.io/)

## Clone and create Python environment

### Clone the repo.
```shell
git clone https://github.com/priyasundaresan/mink.git
```

### Create conda env

First create a conda env:
```shell
conda env create -f mac_env.yml  
```

Then, source `set_env.sh` to activate the `tidybot` conda env and set the `PYTHONPATH` appropriately.

```shell
# NOTE: run this once per shell before running any script from this repo
source set_env.sh
```

## Teleop for Mobile-SPHINX
This is a good sanity check to test if the repo works locally for you, in the conda environment you created.
Remember to run `source set_env.sh`  once per shell before running any script from this repo.

To test whole-body teleoperation using simple mouse click-drag interactions, you can use the following script:
```shell
source set_env.sh
mjpython interactive_scripts/teleop_mouse.py
```
You will see a little red interaction cube appear at the end effector.
You can `Double Click` to select it, then use `Ctrl + Right Click and Drag` (two fingers down on trackpad) to move it positionally, and `Ctrl + Left Click and Drag` to control orientation (one finger down on trackpad).

To test out whole-body teleoperation using the iPhone as the teleoperation device, you can use:
```shell
source set_env.sh
mjpython interactive_scripts/teleop_phone.py
```
This will print out something like `Starting server at 10.30.163.179:5001`. Next, make sure your iPhone is connected to the same Wi-Fi network as your laptop, and that the app XRBrowser is installed. Open XRBrowser, and go to the IP address printed out by the script. You can then use your iPhone to teleoperate the robot. A few quick notes:
* Make sure your iPhone has Portrait Lock Orientaton ON, and start with top of phone facing up (towards your face) not forward (towards your toes).
* Click `Start episode` to start data collection, similarly `End episode` to finish an episode.
* Your actions will only be mirrored when you are pressing the screen.
* Swipe up/down to open/close the gripper. NOTE: the simulated gripper is currently a bit finicky.
* You'll definitely want to practice before collecting any useful datasets! :)

## Training/Evaluating Mobile-SPHINX

### Download data

Create a new `data` folder, download the data from [here](https://drive.google.com/drive/folders/1rzkMgkKm2slidJ2iLmBpVReOenwWI-uq?usp=sharing), and move it into `data` (NOTE, it is also available on `sc` cluster node: `/iliad/u/priyasun/mink/data`). See above instructions if you want to collect your own dataset.

### Inspect the datasets
You can run `python dataset_utils/waypoint_dataset.py` to load the cube dataset, and save some visualizations.

### Training (on the cluster)

To train Mobile-SPHINX, run the following (training logs and eval success rates will be logged to Weights & Biases).

#### Commands

To train the cube and/or cabinet task:
```shell
# cube
python scripts/train_waypoint.py --config_path cfgs/waypoint/cube.yaml

# drawer
python scripts/train_waypoint.py --config_path cfgs/waypoint/cabinet.yaml
```

Use `--save_dir PATH` to specify where to store the logs and models.
Use `--use_wb 0` to disable logging to W&B (this is useful when debugging, to avoid saving unnecessary logs).

### Evaluation (local, on a workstation)
Assuming the resulting checkpoints are saved to `exps/waypoint/cube`, to eval the waypoint policy, you can run the following.
If you have access to a workstation (with GPU and display), run:
```shell
python scripts/eval_waypoint.py --model exps/waypoint/cube/ema.pt --env_cfg envs/cfgs/cube.yaml 
```
Otherwise, if you are evaluating on the cluster or some machine without a display (or over SSH):
```shell
MUJOCO_GL=egl python scripts/eval_waypoint.py --model exps/waypoint/cube/ema.pt --env_cfg envs/cfgs/cube.yaml --headless 
```
This will by default run 20 rollouts and save videos to the folder `rollouts`. For easier viewing, you can then use `python common_utils/display_rollouts.py` to creat a grid of all the rollout videos in a `.html` file that can easily be viewed in any browser.

Note:
`--record 0` will run the rollouts without saving videos (faster if you don't care about visualizing)

## Collecting Data for Mobile-SPHINX
Remember to run `source set_env.sh`  once per shell before running any script from this repo.
This part walks through how to collect data for a task from scratch. You can use this general workflow to collect data for & train other custom tasks.

### Step 1: Collecting Teleoperated Data (Ex: Cube Task)
Run the following:
```shell
source set_env.sh
mjpython interactive_scripts/record_sim.py --env_cfg envs/cfgs/cube.yaml
```
* Open XRBrowser, and go to the IP address printed out by the script, and hit `Start episode.`
* Wait for the simulator window to load, then begin teleoperation.
* Once done, you can click `End episode.`
  * If `is_success` is implemented, you should also see some feedback in Terminal when you have successfully teleoperated the task (after which you should hit `End episode`).
* After you see `Done saving` in Terminal, you can click `Reset` to begin the next episode.
* In general, wait for the simulator to load before teleoperating, and if the robot is not responsive to your iPhone actions, just try refreshing the page. 

Each teleoperated episode will be saved as an `npz` to `dev1` as follows:
```
dev1/
└── demo00000.npz
└── demo00001.npz
...

```
NOTE: If you do mess up a demo after starting an episode and click `End episode`, you will need to manually delete the last recorded `npz` file. Every time you run the script `record_sim.py`, it will start saving from the last recorded demo index if there is one (i.e. if you just recorded `demo00004.npz` and quit, then re-run, it will save from `demo00005.npz`).

To sanity check how your recorded demos look, we provide a script that loads demos from `dev1` and replays them.
```shell
python interactive_scripts/replay_sim.py --env_cfg envs/cfgs/cube.yaml
```

### Step 2: Post-Processing: Labeling Modes
Once happy with the demos recorded in `dev1`, we need to post-process them into a SPHINX-compatible format (e.g., with mode labels and salient point annotations).
The first step is to break up each demo temporally into `waypoint` and `dense` modes.

Run the following script, which will load each demo in `dev1`, visualize it, and allow you to temporally annotate modes.
```shell
python dataset_utils/annotate_modes.py
```
* Go to `http://127.0.0.1:5000` in your browser.
* Use the blue circular cursor to scroll through the frames of the first demo, and `Shift+Click` to specify a waypoint at that frame.
  * `Delete` will remove the last waypoint (if you mess up)
  * Try to use a consistent strategy and number of waypoints across demos. For `cube`, I typically use 3 waypoints: one at the frame where the gripper 'approaches' the cube, one when it 'grasps', and one frame towards the very end of the demo (to 'lift').
* When you're happy labeling that demo, then go to Terminal and press `Enter.`
* Refresh the page to load the next demo.
* If the script for some reason crashes/hangs, interrupt it, re-run, and go back to the URL & refresh. It should load the most recent un-annotated demo.

After this step, you will have a new folder `dev1_relabeled` which contains all the demos, now annotated with modes.

### Step 3: Post-Processing: Labeling Salient Points
The purpose of the above was to temporally relabel demos into dense/waypoint modes.
The last step is to label salient points for the extracted waypoint observations above.
Run:
```shell
python dataset_utils/annotate_salient_points.py
```
* In the window that appears, you will see the point cloud of the first extracted waypoint timestep from the first demo in `dev1_relabeled`
  * You can drag the point cloud around with just `Click` interactions and zoom in using the trackpad to get a better view of where you want to put a salient point
* `Shift+Click` to label a salient point (a colored sphere will appear)
  * You can re-click if you mess up, just note that only the last click will be recorded.
* Press `q` or `Esc` to go to the next obs.  

After this step, `dev1_relabeled` contains all the demos, now annotated with both modes and salient points!
Rename this folder to whatever you want and put it in `data.` See above for how to train/eval the policy on the task for which you just collected data.

### Adding Custom Tasks
To add a custom task, you need to do the following:
* Create XML's:
  * In `interactive_scripts/stanford_tidybot`, create two files: `<task_name>.xml` and `tidybot_<task_name>.xml` (you can basically just copy over `cube.xml` and `tidybot_cube.xml`, replacing with your object assets in `<task_name>.xml`).
* Create a new `env_cfg`: `envs/cfgs/<task_name>.yaml` and give it a name in the `task` field
* Register the task in `envs/mj_env.py`
  * In the `__init__` method, update `xml_file` based on `task`
  * Update `reset_task` 
  * Update `is_success`
* Finally, you can try `mjpython interactive_scripts/record_sim.py --env_cfg envs/cfgs/<task_name>.xml`
  * NOTE: If you get the following error: `ValueError: Error: keyframe 0: invalid qpos size, expected length X`, it means the `home` keyframe of `tidybot_<task_name>.xml` is not the right dimension. This keyframe represents the home configuration of the whole robot, plus the `free joints` of whatever assets are in the scene (i.e., there are 7 free joints for the `cube` task, so the `home` keyframe is padded with `0.6 0 0 0 0 0 0` representing that the cube should be initially positioned at `[0.6, 0, 0]` and a "zero" quaternion -- you can set this to something else if you want different initial object poses).
  * NOTE: You can make `reset_task` do nothing at first, and `is_success` trivially be `False` during debugging. I typically do this just so I can first focus on loading the scene properly, and once I am able to successfully teleoperate the task, I work backwards to figure out what initial scene randomizations are reasonable (for `reset_task`) and what the task success condition should be (for `is_success`, used to evaluate if rollouts are successful).
* If all of this works as expected, you should be able to just pass the `env_cfg` you created whenever you are collecting demos with `record_sim.py` or evaluating policies with the eval scripts in `scripts/eval_<waypoint/hybrid/dense>.py`.
