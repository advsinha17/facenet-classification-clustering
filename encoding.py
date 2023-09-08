from model import Facenet
from face_extraction import preprocess_image
import argparse
import os
import numpy as np
import pickle
from mtcnn import MTCNN

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMAGE_SIZE = (160, 160)

# Initialize the argument parser
parser = argparse.ArgumentParser(description="Extract and store facial embeddings from images")

# Add an argument for specifying the input image files
parser.add_argument("image_files", nargs="+", help="Paths to the input image files")

# Add an argument for specifying the output pickle file
parser.add_argument("--output", help="Path to the output pickle file")

# Parse the command-line arguments
args = parser.parse_args()

# Set the output file path (default to "embeddings.pkl" if not provided)
output_file = args.output if args.output else "embeddings.pkl"

model = Facenet(default_weights = False)
all_embeddings = []
detector = MTCNN()

for input_image_path in args.image_files:
    input_image = preprocess_image(detector, input_image_path)

    if input_image is not None:
        input_image = np.expand_dims(input_image, axis=0)
        embeddings = model.predict(input_image)
        all_embeddings.append(embeddings)

        print(f"Facial embeddings extracted from {input_image_path}")
    else:
        print(f"No face detected in the input image: {input_image_path}")

with open(output_file, "wb") as f:
    pickle.dump(all_embeddings, f)

print(f"All facial embeddings saved to {output_file}")
