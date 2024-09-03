import streamlit as st
import numpy as np
import cv2
from PIL import Image
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
import pickle

# Load the pre-trained models
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load known face embeddings
def load_embeddings(file='face_db.pkl'):
    with open(file, 'rb') as f:
        data = pickle.load(f)
    return data['embeddings'], data['names']

known_embeddings, known_names = load_embeddings()

def extract_embeddings(image):
    img = Image.fromarray(image) if isinstance(image, np.ndarray) else image.convert('RGB')
    img_cropped_list, _ = mtcnn(img, return_prob=True)
    embeddings = []
    bounding_boxes = []

    if img_cropped_list is not None:
        for img_cropped in img_cropped_list:
            img_cropped = img_cropped.unsqueeze(0)
            embedding = model(img_cropped).detach().numpy()
            embeddings.append(embedding)
        
        boxes, _ = mtcnn.detect(img)
        bounding_boxes = boxes

    return embeddings, bounding_boxes

def recognize_face(embedding, known_embeddings, known_names):
    min_distance = float('inf')
    name = "Unknown"
    for i, known_embedding in enumerate(known_embeddings):
        dist = cosine(embedding, known_embedding)
        if dist < min_distance:
            min_distance = dist
            name = known_names[i]
    return name

def recognize_from_image(image):
    embeddings, bounding_boxes = extract_embeddings(image)
    recognized_faces = []
    results = []

    for idx, embedding in enumerate(embeddings):
        if embedding is not None:
            name = recognize_face(embedding[0], known_embeddings, known_names)
            recognized_faces.append(name)
            if bounding_boxes is not None:
                results.append({'name': name, 'box': bounding_boxes[idx]})

    return results

def main():
    st.title('Face Recognition App')

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        results = recognize_from_image(image)
        
        if results:
            img_np = np.array(image)
            for result in results:
                name = result['name']
                box = result['box']
                if box is not None:
                    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                    cv2.putText(img_np, name, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            result_image = Image.fromarray(img_np)
            st.image(result_image, caption='Processed Image', use_column_width=True)
            
            recognized_names = ', '.join([result['name'] for result in results])
            st.write(f"Recognized Faces: {recognized_names}")
        else:
            st.write("No faces detected.")

if __name__ == "__main__":
    main()
