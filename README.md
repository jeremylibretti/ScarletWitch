# ScarletWitch
A project for developing webcam based static and dynamic gesture recognition to be integrated into [MultiCraft](https://github.com/tiilt-lab/MultiCraft) and allow for gesture based game control.

*Currently only tested on Mac*

<!-- This repository contains the following contents.
* ScarletWitch program
* Static gesture recognition model (TFLite)
* Dynamic gesture recognition model (TFLite)
* Learning data for static gesture recognition and notebook for learning
* Learning data for dynamic gesture recognition and notebook for learning -->

## Setup
### Python Environment
1. Create virtual environment with python version 3.8
    * `conda create --name sw_env python=3.8`
    * `conda activate sw_env`
2. Install required modules
    * Navigate to project directory
    * Run `pip install -r requirements.txt`
3. <s>Finish pywin32 installation</s>
    * <s>Navigate to virutal environment directory</s>
    * <s>Run `python Scripts/pywin32_postinstall.py -install`</s>

## Usage
### Gesture Recognition Demo
After setting up your python environment, the gesture recognition demo can be launched by running `python app.py` and controlled using the following keyboard inputs:
* 'esc': close program

The following arguments can be used from the command line to specify the corresponding options:
* --device<br>Specifying the camera device number (Default：0)
* --width<br>Width at the time of camera capture (Default：854)
* --height<br>Height at the time of camera capture (Default：480)
* --collect_static<br>Specifies static gesture data collection mode
* --collect_static<br>Specifies dynamic gesture data collection mode
* --min_detection_confidence<br>
Detection confidence threshold (Default：0.7)
* --min_tracking_confidence<br>
Tracking confidence threshold (Default：0.5)

### Gameplay
Not yet implemented

## Training
### Static Gesture Recognition Training
#### Learning data collection
In python environment with required modules installed:
1. Modify "model/static_classifier/static_classifier_labels.csv" to specify desired gestures
2. Run `python app.py --collect_static`
* Use number keys 0-9 to select gesture corresponding to labels in csv file
* Use 'R' key to record current frame of hand landmarks for selected gesture
* Use 'N' key to deselect current gesture
* Use 'esc' key to terminate program and write to datasets folder
#### Model training
(Note: training notebook is out of date)
In python environment with required modules installed:
1. Start jupyterlab in project directory
2. Open static_training.ipynb
3. Run notebook

### Dynamic Gesture Recognition Training
#### Learning data collection
In python environment with required modules installed:
1. Modify "model/dynamic_classifier/dynamic_classifier_labels.csv" to specify desired gestures
2. Run `python app.py --collect_dynamic`
* Use number keys 0-9 to select gesture corresponding to labels in csv file
* Use 'R' key to record two seconds of hand landmarks for selected gesture
* Use 'N' key to deselect current gesture
* Use 'esc' key to terminate program and write to datasets folder
#### Model training
(Note: training notebook is out of date)
In python environment with required modules installed:
1. Start jupyterlab in project directory
2. Open dynamic_training.ipynb
3. Run notebook

## Directory
<pre>
│  app.py
│  scarletwitch.py
│  webcam.py
│  handtracker.py
│  gestureclassifier.py
│  collect_static.py
│  collect_dynamic.py
│  static_training.ipynb
│  dynamic_training.ipynb
│  requirements.txt
│  
├─model
│  ├─static_classifier
│      │  dynamic_classifier.hdf5
│      │  dynamic_classifier.py
│      │  dynamic_classifier.tflite
│      └─ dynamic_classifier_labels.csv
│  │          
│  └─dynamic_classifier
│      │  static_classifier.hdf5
│      │  static_classifier.py
│      │  static_classifier.tflite
│      └─ static_classifier_labels.csv
│    
├─datasets      
│    
└─utils
    └─cvfpscalc.py
</pre>
### app.py
This is a wrapper for the main scarletwitch program and data collection scripts that handles command line arguments.

### scarletwitch.py
This is the main program.

### webcam.py
This is an object that initializes a new thread on which it captures webcam input. It allows for the current webcam stream to be accessed by the main scarletwitch process and the handtracker process when needed.

### handtracker.py
This is an object that initializes a new thread on which it processes frames from the webcam thread using the MediaPipe Hands solution. It allows for the hand landmarks in the latest frame to be accessed by the main scarletwitch process when needed.

### gestureclassifier.py
This is an object that loads the static and dynamic gesture models upon initialization and is used from the main scarletwitch script to classify gestures while running.

### collect_static.py
This is a subclass of the main scarletwitch class that is instantiated when collecting dynamic gesture data.

### collect_dynamic.py
This is a subclass of the main scarletwitch class that is instantiated when collecting dynamic gesture data.

### static_training.ipynb
This is a model training script for static gesture recognition.

### dynamic_training.ipynb
This is a model training script for dynamic gesture recognition.

### requirements.txt
This file lists all necessary packages and dependencies.

### model/static_classifier
This directory stores files related to static gesture recognition.<br>
The following files are stored.
* Inference module(static_classifier.py)
* Trained model(static_classifier.tflite)
* Label data(static_classifier_label.csv)

### model/dynamic_classifier
This directory stores files related to dynamic gesture recognition.<br>
The following files are stored.
* Inference module(dynamic_classifier.py)
* Trained model(dynamic_classifier.tflite)
* Label data(dynamic_classifier_label.csv)

### datasets
This directory stores training data in the form of numpy arrays.

### utils/cvfpscalc.py
This is a module for FPS measurement.

## Known Issues
* Training scripts use wrong datasets
* Dynamic data collection records nothing when hand landmarks are not present; should record empty data (maybe fixed?)
* Numpy ndarray warning in collect_dynamic.py

## TODO
1. Update training scripts to use correct datasets
2. Improve model/training
4. Implement OS detection
3. Reimplement game control module for windows script

## Sources
* [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html) - Google
* Hand Gesture Recognition Using Mediapipe: [EN](https://github.com/kinivi/hand-gesture-recognition-mediapipe)/[JA](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe) - Kazuhito00
* [Gestop](https://arxiv.org/abs/2010.13197) - Krishna, Sinha
* [Gesture Recognition](https://github.com/kairess/gesture-recognition) - kairess
