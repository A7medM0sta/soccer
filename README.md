

---

#  Soccer AI 🧠
## Demo
<p align="center">
  <img src="assets/outputs/1.png" alt="Image 1" width="400"/> 
  <img src="assets/outputs/2.png" alt="Image 2" width="400"/>
  <img src="assets/outputs/3.png" alt="Image 3" width="400"/>
  <img src="assets/outputs/4.png" alt="Image 4" width="400"/>
</p>

## 📝 Overview
Research and development of AI models for soccer analytics and insights. This project aims to provide tools for detecting players, goalkeepers, referees, and the ball in soccer videos. It also includes features for tracking player movements, classifying players into teams, and visualizing player positions on the soccer field.
### Train ball detectors
<p align="center">
  <div style="display: inline-block; margin: 10px;">
    <p><strong>Train Results</strong></p>
    <img src="assets/img_1.png" alt="Train Results" width="400"/>
  </div>
  <div style="display: inline-block; margin: 10px;">
    <p><strong>Confusion Matrix</strong></p>
    <img src="assets/img.png" alt="Confusion Matrix" width="400"/>
  </div>
  <div style="display: inline-block; margin: 10px;">
    <p><strong>Result for Validation</strong></p>
    <img src="assets/img_2.png" alt="Result for Validation" width="400"/>
  </div>
</p>

### Train player detectors
* train results
![train results](assets/img_3.png)
* Confession matrix
![Confusion matrix](assets/img_5.png)
* Result for Validation
![Result for Validation](assets/img_4.png)

### train pitch keypoint detectors
* train results
![train results](assets/img_6.png)
* Result for Validation
* ![Result for Validation](assets/img_7.png)

## 💻 Installation Guide

We don't have a Python package yet, but you can install from the source in a [**Python>=3.8**](https://www.python.org/) environment. Follow the steps below:

```bash
pip install https://github.com/A7medM0sta/soccer.git
pip install -r requirements.txt
./setup.sh
```

## 📊 Datasets

Original data comes from the [DFL - Bundesliga Data Shootout](https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout) Kaggle competition. This data has been processed to create new datasets, which can be downloaded from [Roboflow Universe](https://universe.roboflow.com/).

| ⚽ Use Case | 📂 Dataset | 🧑‍🏫 Train Model |
|:-------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| 👨‍🏫 Soccer Player Detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-players-detection-3zvbc) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_player_detector.ipynb) |
| ⚽ Soccer Ball Detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-ball-detection-rejhg) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_ball_detector.ipynb) |
| 🏟️ Soccer Pitch Keypoint Detection | [![Download Dataset](https://app.roboflow.com/images/download-dataset-badge.svg)](https://universe.roboflow.com/roboflow-jvuqo/football-field-detection-f07vi) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/sports/blob/main/examples/soccer/notebooks/train_pitch_keypoint_detector.ipynb) |

## 🧠 Models

- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Player Detection) - Detects players, goalkeepers, referees, and the ball in the video.
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/) (Pitch Detection) - Identifies the soccer field boundaries and key points.
- [SigLIP](https://huggingface.co/docs/transformers/en/model_doc/siglip) - Extracts features from image crops of players.
- [UMAP](https://umap-learn.readthedocs.io/en/latest/) - Reduces the dimensionality of the extracted features for easier clustering.
- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) - Clusters the reduced-dimension features to classify players into two teams.

## ⚙️ Modes

### 🏟️ Pitch Detection
Detects the soccer field boundaries and key points in the video. Useful for identifying and visualizing the layout of the soccer pitch.

```bash
python main.py --source_video_path data/2e57b9_0.mp4 --target_video_path data/2e57b9_0-pitch-detection.mp4 --device mps --mode PITCH_DETECTION
```

![Pitch Detection](https://github.com/user-attachments/assets/cf4df75a-89fe-4c6f-b3dc-e4d63a0ed211)

### 🧑‍🤝‍🧑 Player Detection
Detects players, goalkeepers, referees, and the ball in the video. Essential for identifying and tracking the presence of players and other entities on the field.

```bash
python main.py --source_video_path data/2e57b9_0.mp4 --target_video_path data/2e57b9_0-player-detection.mp4 --device mps --mode PLAYER_DETECTION
```

![Player Detection](https://github.com/user-attachments/assets/c36ea2c1-b03e-4ffe-81bd-27391260b187)

### ⚽ Ball Detection
Detects the ball in the video frames and tracks its position. Useful for following ball movements throughout the match.

```bash
python main.py --source_video_path data/2e57b9_0.mp4 --target_video_path data/2e57b9_0-ball-detection.mp4 --device mps --mode BALL_DETECTION
```

![Ball Detection](https://github.com/user-attachments/assets/2fd83678-7790-4f4d-a8c0-065ef38ca031)

### 🏃‍♂️ Player Tracking
Tracks players across video frames, maintaining consistent identification. Useful for following player movements and positions throughout the match.

```bash
python main.py --source_video_path data/2e57b9_0.mp4 --target_video_path data/2e57b9_0-player-tracking.mp4 --device mps --mode PLAYER_TRACKING
```

![Player Tracking](https://github.com/user-attachments/assets/69be83ac-52ff-4879-b93d-33f016feb839)

### 🏳️‍ Team Classification
Classifies detected players into their respective teams based on their visual features. Helps differentiate between players of different teams for analysis and visualization.

```bash
python main.py --source_video_path data/2e57b9_0.mp4 --target_video_path data/2e57b9_0-team-classification.mp4 --device mps --mode TEAM_CLASSIFICATION
```

![Team Classification](https://github.com/user-attachments/assets/239c2960-5032-415c-b330-3ddd094d32c7)

### 🎯 Radar Mode
Combines pitch detection, player detection, tracking, and team classification to generate a radar-like visualization of player positions on the soccer field. Provides a comprehensive overview of player movements and team formations.

```bash
python main.py --source_video_path data/2e57b9_0.mp4 --target_video_path data/2e57b9_0-radar.mp4 --device mps --mode RADAR
```

![Radar](https://github.com/user-attachments/assets/263b4cd0-2185-4ed3-9be2-cf4d8f5bfa67)

---





## References
* https://github.com/roboflow/notebooks/blob/main/notebooks/train-yolov8-object-detection-on-custom-dataset.ipynb) 
* https://github.com/roboflow/notebooks
* 
