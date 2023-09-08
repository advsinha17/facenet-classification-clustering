"""

Uses MTCNN detector to extract faces from images given in data/Face Data.
Extracted faces stored in data/Extracted Faces.

"""

import cv2
import os
from mtcnn import MTCNN
import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

IMAGE_SIZE = (160, 160)

def preprocess_image(detector, image_path):
    """
    Preprocesses an image to extract and resize the face detected in the image.

    Args:
        detector: a pre-initialized face detector object
        image_path: str, the path to the image file to be preprocessed

    Returns:
        A preprocessed face image if a face is detected, otherwise None.
    """
    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)

    if faces:
        x, y, width, height = faces[0]['box']
        face_image = img[y:y+height, x:x+width]
        face_image = cv2.resize(face_image, IMAGE_SIZE)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        return face_image

    return None

if __name__ == "__main__":
    detector = MTCNN()

    CWD = os.path.dirname(__file__)
    input_dir = os.path.join(CWD, "data/Face Data")
    output_dir = os.path.join(CWD, "data/Extracted Faces")

    num_dirs = len(os.listdir(input_dir))

    for i in range(num_dirs):
        input_subdir = os.path.join(input_dir, str(i))
        output_subdir = os.path.join(output_dir, str(i))

        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)

        image_paths = [os.path.join(input_subdir, f"{j}.jpg") for j in range(len(os.listdir(input_subdir)))]
        for input_path in image_paths:
            preprocessed_image = preprocess_image(detector, input_path)

            if preprocessed_image is not None:
                output_path = os.path.join(output_subdir, os.path.basename(input_path))
                plt.imshow(preprocessed_image)
                plt.axis("off")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)

            else:
                print("No face detected in:", input_path)
