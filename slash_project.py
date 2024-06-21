from ultralytics import YOLO   # the detector model
import streamlit as st    # for creating a streamlit applocation
from PIL import Image, ImageDraw, ImageFont  # for uploading the image and show the lables on the image


def image_detect(image):
    # Load YOLO model
    model = YOLO("yolov8m.pt")
    
    # Predict the image components
    predictions=model.predict(image)
    
    # Since we only have 1 image, we will have a 2d array with 1 entry.
    return predictions[0]

def application():
    # Streamlit interface
    st.set_page_config(page_title="Image Component Detector", page_icon="🔍", layout="wide")
    st.title("🔍 Image Component Detector")
    st.write("Upload an image to detect and display components present in the image.")

    # Sidebar
    st.sidebar.title("Options")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # File uploader
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert('RGB')  # Ensure the image is in RGB mode
            st.image(image, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("To Classify...")
            
            if st.button("Analyse Image"):
                result = image_detect(image)
                
                # Draw the bounding boxes and labels on the image
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()
                
                labels = []
                
                for box in result.boxes:
                    class_id = result.names[box.cls[0].item()]
                    cords = box.xyxy[0].tolist()
                    cords = [round(x) for x in cords]
                    conf = round(box.conf[0].item(), 2)
                    
                    if conf >= confidence_threshold:
                        labels.append((class_id, conf))
                        
                        # Draw rectangle
                        draw.rectangle(cords, outline="red", width=3)
                        
                        # Draw label
                        label = f"{class_id}: {conf}"
                        text_size = draw.textsize(label, font=font)
                        draw.rectangle([cords[0], cords[1] - text_size[1], cords[0] + text_size[0], cords[1]], fill="red")
                        draw.text((cords[0], cords[1] - text_size[1]), label, fill="white", font=font)
                
                # Sort labels alphabetically
                labels.sort()
                
                # Display the image with bounding boxes and labels
                st.image(image, caption='Processed Image.', use_column_width=True)
                st.write("Detected components:")
                for label in labels:
                    st.write(f"{label[0]}: {label[1]}")
        
        except Exception as e:
            st.write("Error processing image:", e)
            
            
            
          
application()            




