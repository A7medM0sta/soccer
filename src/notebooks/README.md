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
