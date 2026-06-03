# PPE Detection and Tracking System

## Project Structure

```text
ppe-detection-and-tracking/
├── ppe_app.py
├── best.pt
├── my_tracker.yaml
├── PPE_Detection_Tracking_Analysis.ipynb
├── requirements.txt
├── README.md
├── evaluation/
└── test_images/
```

## Dataset

The full dataset is not included in this repository due to size limitations.

📥 Dataset Download:

[[DATASET_KAGGLE_LINK](https://www.kaggle.com/datasets/muhammetzahitaydn/hardhat-vest-dataset-v3)]

## Demo Videos

Example videos used for testing and demonstration are available here:

📹 Demo Videos:

[[VIDEOS_GOOGLE_DRIVE_LINK](https://drive.google.com/drive/folders/1UDa3lHwAO9yxAKa9sI2o0RKMrYfGIu5L?usp=sharing)]

## Model Weights

Project Drive Link:

[PROJECT GOOGLE DRIVE LINK](https://drive.google.com/drive/folders/13XIPPedNYfqhsa5X9Ney1cI0qn5Ks61j?usp=sharing)

## Model Evaluation

The evaluation folder contains:

* Training results
* Confusion matrices
* Precision, Recall, and F1 score curves
* Validation prediction examples

### Performance Visualizations

* results.png
* confusion_matrix.png
* confusion_matrix_normalized.png
* BoxP_curve.png
* BoxR_curve.png
* BoxF1_curve.png

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/ppe-detection-and-tracking.git
cd ppe-detection-and-tracking
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run ppe_app.py
```

## Usage

### Image Detection

1. Select **Detection** mode.
2. Upload an image.
3. Click **Run Analysis**.
4. View and download the processed result.

### Video Tracking

1. Select **Tracking** mode.
2. Upload a video.
3. Click **Run Analysis**.
4. View and download the processed video.

## Technologies Used

* Python
* Streamlit
* Ultralytics YOLO
* OpenCV
* PyTorch
* FFmpeg
* ImageIO

## Future Improvements

* Additional PPE classes
* Real-time webcam monitoring
* Alert generation for PPE violations
* Multi-camera support

## Author

Habiba Khalid
