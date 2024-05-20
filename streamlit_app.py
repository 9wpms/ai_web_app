import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time

# Define the class names (make sure these match the classes your model is trained to detect)
class_names = ['Sign Plate']  # Adjust this list according to your model

def model_process(pil_image):
    # Load YOLO model
    model = YOLO('weight/Boinier.pt')
    
    # Convert PIL image to a format suitable for YOLO and OpenCV
    image = np.array(pil_image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    results = model(image)

    if isinstance(results, list):
        results = results[0]

    def get_complementary_color(color):
        return tuple(255 - int(c) for c in color)

    for bbox in results.boxes:
        x1, y1, x2, y2 = map(int, bbox.xyxy[0])
        conf = bbox.conf[0]
        label = int(bbox.cls[0])

        region = image[y1:y2, x1:x2]
        avg_color_per_row = np.average(region, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        avg_color = avg_color.astype(int)

        comp_color = get_complementary_color(avg_color)

        cv2.rectangle(image, (x1, y1), (x2, y2), comp_color, 2)

        class_name = class_names[label] if label < len(class_names) else f'Class {label}'
        conf_percentage = f'{conf * 100:.2f}%'
        display_text = f'{class_name}: {conf_percentage}'

        font_scale = 0.6
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        text_x, text_y = x1, y1 - 10 if y1 - 10 > 10 else y1 + text_height + 10

        cv2.rectangle(image, (text_x, text_y - text_height - baseline), (text_x + text_width, text_y + baseline), (0, 0, 0), cv2.FILLED)
        cv2.putText(image, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    return image_rgb

def webpage():
    st.balloons()
    st.title('💖AI Model: License Plate and Sign Detector')

    st.markdown('Identifying license plates and signs through image uploading is made easier by this web application.')
    st.caption('(เว็บไซต์สำหรับตรวจสอบป้ายต่าง ๆ แบบอย่างง่าย ผ่านการอัปโหลดรูปภาพ)')
    
    with st.sidebar.container():
        st.info("Click the 'Process' button if you make a selection.")
        selected_option = st.sidebar.radio(
            "Options for showing examples: 📝",
            ['Yes', 'No'], 
            index=1
        )
        
        if selected_option == 'Yes':
            example_selected = st.sidebar.radio(
                "Select an example:",
                ['1: Lots of cars are parked.', '2: Car in grey tones.', '3: A yellow car license.', '4: A sign of navigation.', '5: A metal sign plate.', '6: An exit sign plate.']
            )
            
            if example_selected == '1: Lots of cars are parked.':
                example_image = 'example/example1.jpg'
            elif example_selected == '2: Car in grey tones.':
                example_image = 'example/example2.jpg'
            elif example_selected == '3: A yellow car license.':
                example_image = 'example/example3.jpg'
            elif example_selected == '4: A sign of navigation.':
                example_image = 'example/example4.jpg'
            elif example_selected == '5: A metal sign plate.':
                example_image = 'example/example5.jpg'
            elif example_selected == '6: An exit sign plate.':
                example_image = 'example/example6.jpg'
            uploaded_file = example_image
    
    if selected_option == 'No':
        uploaded_file = st.file_uploader("Upload an image file then click the button below: 👇", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Image has uploaded', use_column_width=True)

        if st.button('Process Image: ✅'):
            image_rgb = model_process(image)
            
            progress_text = "Operation is in progress. Please wait."
            my_bar = st.progress(0, text=progress_text)
            for percent_complete in range(100):
                time.sleep(0.01)
                my_bar.progress(percent_complete + 1, text=progress_text)
            time.sleep(1)
            my_bar.empty()

            st.image(image_rgb, caption='Outcome Image', use_column_width=True)
           
    with st.expander("About members: 📦"):
        st.write('Joshua Boineir: Responsible for AI training and model development.')
        st.image('src/IMG_0338.jpg', use_column_width=True)
        st.write('Wunpeemai Boonta: Responsible for developing the web application interface and integration.')
        st.image('src/IMG_1767.jpg', use_column_width=True)
    st.caption('Thank you for using our webapp.')

webpage()
