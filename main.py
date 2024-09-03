import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
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

    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        results = recognize_from_image(image)
        try:
                font = ImageFont.truetype("arialbd.ttf", 26)  # You can use a path to a bold font file if available
        except IOError:
                font = ImageFont.load_default()
                
        bright_green = (0, 255, 127)
        if results:
            # Draw on the image
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)
            for result in results:
                name = result['name']
                box = result['box']
                if box is not None:
                    # Draw the rectangle and text
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline=bright_green, width=3)
                    draw.text((box[0], box[1] - 26), name, fill=bright_green, font=font)
            
            st.image(img_with_boxes, use_column_width=True)
            
            # recognized_names = ', '.join([result['name'] for result in results])
            # st.write(f"Recognized Faces: {recognized_names}")
        else:
            st.write("No faces detected.")

if __name__ == "__main__":
    main()
