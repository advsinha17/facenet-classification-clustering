## Face Classification and Clustering using Facenet

Data for this can be found here: https://drive.google.com/drive/folders/1VU9AVJSZakOFCR49tTRdCauC-qkS13T-

I have built on this project to implement a pipeline which allows users to simply store images in a folder(even ones with multiple faces per image) and run a script to create separate folders for each person, which will have images in which that persons face is present.
This project can be found [here](https://github.com/advsinha17/crux-r3/tree/main/task2).

## Introduction

This project uses the FaceNet architecture to perform face classification and clustering. FaceNet learns a neural network that encodes facial images into a compact space where distances directly correspond to a measure of face similarity.

The Facenet model is fine-tuned on the above dataset and is then used for clustering using DBSCAN.

Faces are extracted from images using the MTCNN detector. `face_extraction.py` extracts faces from images in the data/Face Data directory and stores them in data/Extracted Faces.

`encoding.py` extracts faces from images, finds encodings and stores them in a pickle file. Run the file as follows:

```
python encoding.py input_image1.jpg input_image2.jpg output_embeddings.pkl
```

## Prerequisites

- Python: 3.10
- Libraries: TensorFlow 2.x, Matplotlib, MTCNN, NumPy, OpenCV, tqdm

## Model architecture

FaceNet model is defined in the `architecture.py` file. Pre-trained FaceNet weights are then loaded. Model is fine-tuned on the above dataset.

FaceNet model was introduced in [this](https://arxiv.org/abs/1503.03832) paper.

## Project Structure

- `architecture.py`: Defines the FaceNet architecture.
- `dataset.py`: Implements DatasetGenerator class which generates batches of images from the dataset.
- `encoding.py`: Extracts faces from images, finds encodings and stores them in a pickle file.
- `face_extraction.py`: Preprocesses images and uses MTCNN detector detect and extract faces from images.
- `model.py`: Implements FaceNet class which defines the train step, triplet loss function, fit and predict methods.
- `utils.py`: Defines utility functions.
- `main.ipynb`: Trains the model and performs clustering.
- `cluster_results`: Stores the clusters created using DBSCAN clustering.
- `model_data`: Pre-trained FaceNet weights.
- `weights`: Fine-tuned weights.
