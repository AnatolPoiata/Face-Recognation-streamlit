import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from facenet_pytorch import InceptionResnetV1, MTCNN
from scipy.spatial.distance import cosine
import pickle
import os

# Load the pre-trained models
mtcnn = MTCNN(keep_all=True)
model = InceptionResnetV1(pretrained='vggface2').eval()

# Define a threshold for recognizing a face
RECOGNITION_THRESHOLD = 0.469  # Adjust this value based on your testing

def load_embeddings(file='face_db.pkl'):
    if os.path.exists(file):
        with open(file, 'rb') as f:
            data = pickle.load(f)
        # Convert embeddings to a list if they are loaded as numpy arrays
        return list(data['embeddings']), list(data['names'])
    else:
        return [], []

known_embeddings, known_names = load_embeddings()

# Save known face embeddings
def save_embeddings(embeddings, names, file='face_db.pkl'):
    with open(file, 'wb') as f:
        pickle.dump({'embeddings': embeddings, 'names': names}, f)

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
    
    # If the minimum distance is above the threshold, return "Unknown"
    print(name, min_distance)
    if min_distance > RECOGNITION_THRESHOLD:
        name = "Unknown"
    
    return name

def recognize_from_image(image):
    embeddings, bounding_boxes = extract_embeddings(image)
    results = []

    for idx, embedding in enumerate(embeddings):
        if embedding is not None:
            name = recognize_face(embedding[0], known_embeddings, known_names)
            if bounding_boxes is not None:
                results.append({'name': name, 'box': bounding_boxes[idx]})

    return results

def add_new_face(name, image):
    global known_embeddings, known_names

    # Extract embeddings from the image
    embeddings, _ = extract_embeddings(image)

    if embeddings:
        # Add new face data
        known_embeddings.append(embeddings[0][0])
        known_names.append(name)

        # Save updated data to the file
        save_embeddings(known_embeddings, known_names)
        st.success(f"{name} has been added to the dataset.")
    else:
        st.error("No face detected in the image. Please try again with a different photo.")

def main():
    st.title('Face Recognition App')

    # Upload an image for face recognition
    uploaded_file = st.file_uploader("Choose an image file for recognition", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = recognize_from_image(image)
        
        if results:
            # Draw on the image
            img_with_boxes = image.copy()
            draw = ImageDraw.Draw(img_with_boxes)

            # Define colors
            text_color = (255, 255, 255)  # White text color

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

    # Add a new face to the dataset with expander
    with st.expander("Add a New Face to the Dataset"):
        new_name = st.text_input("Enter your name")
        new_image_file = st.file_uploader("Upload an image for adding to the dataset", type=["jpg", "jpeg", "png"])

        if st.button("Add Face") and new_name and new_image_file:
            new_image = Image.open(new_image_file)
            add_new_face(new_name, new_image)

if __name__ == "__main__":
    main()
