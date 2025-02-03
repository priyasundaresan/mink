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

## Data Collection for Mobile-SPHINX
Remember to run `source set_env.sh`  once per shell before running any script from this repo.
We collect data for training Mobile-SPHINX policies using Jimmy Wu's iPhone teleop interface to control TidyBot (along with a Whole Body Controller written using Kevin Zakka's `mink` library). I usually do this locally on my Mac:
```shell
source set_env.sh
mjpython interactive_scripts/record_sim.py
```

This will print out something like `Starting server at 10.30.163.179:5001`. Next, make sure your iPhone is connected to the same Wi-Fi network as your laptop, and that the app XRBrowser is installed. Open XRBrowser, and go to the IP address printed out by the script. You can then use your iPhone to teleoperate the robot. A few quick notes:
* Make sure your iPhone has Portrait Lock Orientaton ON, and start with top of phone facing up not forward.
* Click `Start episode` to start data collection, similarly `End episode` to finish an episode.
* Your actions will only be mirrored when you are pressing the screen.
* Swipe up/down to open/close the gripper. NOTE: the gripper is currently a bit finicky.
* You'll definitely want to practice before collecting any useful datasets! :)

If you just want to test out teleoperation using the iPhone (without recording any data), you can use:
```shell
source set_env.sh
mjpython interactive_scripts/teleop_phone.py
```
Same instructions as above for connecting your iPhone and teleoperating.

To just test out the whole body controller, you can use the following script to control the robot using only mouse click & drag actions:
```shell
source set_env.sh
mjpython interactive_scripts/teleop_mouse.py
```
You will see a little red interaction cube at the end effector appear. You can `Double Click` to select it, then use `Ctrl + Right Click and Drag` to move it positionally, and `Ctrl + Left Click and Drag` to control orientation.


## Training Mobile-SPHINX

### Download data

Create a new `data` folder, and copy the data from `sc`: `/iliad/u/priyasun/mink/data` there.

### Training (on the cluster)

To train SPHINX, run the following (training logs and eval success rates will be logged to Weights & Biases).

#### Commands

```shell
# cube
python scripts/train_waypoint.py --config_path cfgs/waypoint/cube.yaml

# drawer
python scripts/train_waypoint.py --config_path cfgs/waypoint/drawer.yaml
```

Use `--save_dir PATH` to specify where to store the logs and models.
Use `--use_wb 0` to disable logging to W&B (useful when debugging).

### Evaluation (local, on a workstation)
Assuming the resulting checkpoints are saved to `exps/waypoint/cube`, to eval the waypoint policy:
```shell
python scripts/eval_waypoint.py --model exps/waypoint/run1/ema.pt --record 1 --save_dir rollouts
```

This will save rollouts to the folder `rollouts`. You can use `python common_utils/display_rollouts.py` to creat a grid of all the rollout videos in a `.html` file that can easily be viewed in any browser.



