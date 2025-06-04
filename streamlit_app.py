import streamlit as st
import cv2 
from PIL import Image, ImageEnhance
import numpy as np

FACE_CASCADE = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
EYE_CASCADE = cv2.CascadeClassifier('haarcascade_eye.xml')
SMILE_CASCADE = cv2.CascadeClassifier('haarcascade_smile.xml')

def detect_faces(image):
    new_image = np.array(image.convert('RGB'))
    img = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = EYE_CASCADE.detectMultiScale(roi_gray, 1.1, 22)
        smiles = SMILE_CASCADE.detectMultiScale(roi_gray, 1.8, 20)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 2)
    return img, faces

def main():
    """Face Detection App"""
    st.title("Face Detection App")
    st.text ("build with streamlit and open cv")

    activities = ["home", "about"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'home':
        st.subheader("Face Detection")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    
        if image_file is not None:
            image = Image.open(image_file)
            st.text("Original Image")
            st.image(image)

        task = ["face detection"]
        feature_choice = st.sidebar.selectbox("Task", task)
        if st.button("process"):
            if feature_choice == 'face detection':
                result_img, result_faces = detect_faces(image)
                st.success("found {} faces".format(len(result_faces)))
                st.image(result_img)


    elif choice == 'about':
        st.subheader("about face detection app")
        st.markdown("built with streamlit and open cv for ITelvore")
        st.text("On Progress")

if __name__ == '__main__':
    main()