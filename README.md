## Face Classification and Clustering using Facenet 

Data for this can be found here: https://drive.google.com/drive/folders/1VU9AVJSZakOFCR49tTRdCauC-qkS13T-

Facenet model is fine-tuned on the above dataset and is then used for clustering using DBSCAN.

Faces are extracted from images using the MTCNN detector. `face_extraction.py` extracts faces from images in the data/Face Data directory and stores them in data/Extracted Faces. 

`encoding.py` extracts faces from images, finds encodings and stores them in a pickle file. Run the file as follows:
```
python encoding.py input_image1.jpg input_image2.jpg output_embeddings.pkl
```



