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
        # st.image(image, caption='Uploaded Image', use_column_width=True)

        results = recognize_from_image(image)
        
        if results:
            # Draw on the image
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)

            # Define colors
            text_color = (255, 255,255)  # Semi-transparent black

            # Load a larger font
            try:
                font = ImageFont.truetype("ARIALBD.TTF", 26)  # Update path
            except IOError:
                font = ImageFont.load_default()  # Fallback to default if custom font is not available

            for result in results:
                name = result['name']
                box = result['box']
                if box is not None:
                    # Draw the rectangle
                    draw.rectangle([box[0], box[1], box[2], box[3]], outline="green", width=3)
                    
                    # Calculate text size and position using textbbox
                    text_bbox = draw.textbbox((box[0], box[1]), name, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]
                    
                    # Background rectangle
                    background_box = [box[0], box[1] - text_height - 10, box[0] + text_width + 10, box[1]]
                    draw.rectangle(background_box, fill="green")
                    
                    # Draw the text on top
                    draw.text((box[0] + 5, box[1] - text_height - 5), name, fill=text_color, font=font)
            
            st.image(img_with_boxes, use_column_width=True)
            
        else:
            st.markdown("<h2 style='color:red;'>No faces detected.</h2>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()