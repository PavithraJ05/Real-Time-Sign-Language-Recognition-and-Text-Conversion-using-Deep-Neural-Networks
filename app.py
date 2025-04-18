import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import queue
import time
from threading import Thread, Lock
from streamlit_option_menu import option_menu
import pyttsx3  # Text-to-speech
import tempfile
import os
from fpdf import FPDF

st.set_page_config(page_title="Sign Language Translator", layout="wide")


def download_text():
    text = st.session_state.text
    if text:
        col1, col2 = st.columns([1, 1])  # Create two columns

        with col1:
            st.download_button(
                label="Download as TXT",
                data=text,
                file_name="recognized_text.txt",
                mime="text/plain"
            )

        with col2:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(190, 10, text)
            pdf_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            pdf.output(pdf_file.name)
            st.download_button(
                label="Download as PDF",
                data=open(pdf_file.name, "rb").read(),
                file_name="recognized_text.pdf",
                mime="application/pdf"
            )


def main():
    with st.sidebar:
        selected = option_menu(
            menu_title=None,
            options=["Home", "Translation"],
            icons=["house", "translate"],
            menu_icon=None,
            default_index=0,
            styles={
                "nav-link-selected": {"background-color": "green", "color": "white"},
                "icon": {"color": "black"}
            }
        )

    if selected == "Home":
        st.markdown("<h1 style='text-align: center;'>Welcome to Real-Time Sign Language Translator</h1>", unsafe_allow_html=True)
        st.write("""
        ### About This App  
        This application enables real-time translation of **sign language into text** using a deep learning model and computer vision.  
        It uses **MediaPipe** for hand tracking and a trained neural network for gesture recognition.
        """)
        st.write("""
        ### Features  
        - Real-time sign detection via webcam  
        - Real-time hand tracking with MediaPipe  
        - Displays recognized text in real-time  
        - User-friendly interface    
        """)
        st.write("""
        ### How It Works  
        - The camera captures **hand gestures**.  
        - MediaPipe extracts **hand landmarks**.  
        - A **deep learning model** predicts the sign.  
        - The recognized text is displayed in real-time.  
        """)


    elif selected == "Translation":
        st.markdown("<h1 style='text-align: center;'>Sign Language to Text Conversion</h1>", unsafe_allow_html=True)

        # Custom CSS for button styling
        st.markdown(
            """
            <style>
                div.stButton > button {
                    white-space: nowrap; 
                    background-color: green !important;
                    color: white !important;
                    border-radius: 5px;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        class VideoThread(Thread):
            def __init__(self, recognizer, video_source=0):
                Thread.__init__(self)
                self.recognizer = recognizer
                self.frame_queue = queue.Queue(maxsize=2)
                self.prediction_queue = queue.Queue(maxsize=2)
                self.running = False
                self.video_source = video_source
                self.video = None

            def run(self):
                self.video = cv2.VideoCapture(self.video_source)
                while self.running:
                    ret, frame = self.video.read()
                    if not ret:
                        break  

                    frame = cv2.flip(frame, 1)
                    result = self.recognizer.process_frame(frame)

                    if result:
                        prediction, confidence = result
                        cv2.putText(
                            frame,
                            f"{prediction} ({confidence:.2f})",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2
                        )
                        self._update_queue(self.prediction_queue, (prediction, confidence))

                    self._update_queue(self.frame_queue, frame)
                    time.sleep(0.01)

                self.video.release()

            def _update_queue(self, q, item):
                try:
                    q.put_nowait(item)
                except queue.Full:
                    q.get_nowait()
                    q.put_nowait(item)

            def start_camera(self):
                if not self.running:
                    self.running = True
                    self.start()

            def stop_camera(self):
                self.running = False
                self.join()
                if self.video:
                    self.video.release()
                    self.video = None

        class HandGestureRecognizer:
            def __init__(self, model_path="model_output/hand_gesture_model.h5", metadata_path="model_output/model_metadata.json"):
                self.model = load_model(model_path)
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)

                self.mp_hands = mp.solutions.hands
                self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
                self.labels = {int(k): v for k, v in self.metadata['labels'].items()}

            def process_frame(self, frame):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)

                if not results.multi_hand_landmarks:
                    return None  

                landmarks = results.multi_hand_landmarks[0]
                mp.solutions.drawing_utils.draw_landmarks(frame, landmarks, self.mp_hands.HAND_CONNECTIONS)

                wrist = landmarks.landmark[0]
                points = [coord for landmark in landmarks.landmark for coord in (landmark.x - wrist.x, landmark.y - wrist.y, landmark.z - wrist.z)]

                if len(points) != self.model.input_shape[1]:  
                    return None  

                prediction = self.model.predict(np.array([points]), verbose=0)
                predicted_class = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class]

                return self.labels[predicted_class], confidence

        def initialize_session_state():
            if 'text' not in st.session_state:
                st.session_state.text = ""
            if 'recognizer' not in st.session_state:
                st.session_state.recognizer = HandGestureRecognizer()
            if 'video_thread' not in st.session_state:
                st.session_state.video_thread = None
            if 'lock' not in st.session_state:
                st.session_state.lock = Lock()
            if 'video_file' not in st.session_state:
                st.session_state.video_file = None

        def capture_text():
            try:
                prediction, _ = st.session_state.video_thread.prediction_queue.get_nowait()
                with st.session_state.lock:
                    st.session_state.text += prediction
            except queue.Empty:
                pass

        def clear_text():
           with st.session_state.lock:
            st.session_state.text = ""  # Clear session state text
            st.session_state.text_display = ""  # Reset text area content

        def delete_last_char():
          with st.session_state.lock:
           if st.session_state.text:  # Check if there's text to delete
             st.session_state.text = st.session_state.text[:-1]  # Remove last character


        def add_space():
            with st.session_state.lock:
                st.session_state.text += " "

        def speak_text():
            engine = pyttsx3.init()
            engine.setProperty('rate', 125)
            engine.say(st.session_state.text)
            engine.runAndWait()

        initialize_session_state()

        col1, col2 = st.columns(2)

        with col1:
            frame_placeholder = st.empty()
            
            col1_btns = st.columns([1, 1])
            with col1_btns[0]:
                if st.button("Start Camera", use_container_width=True):
                    if st.session_state.video_thread is None or not st.session_state.video_thread.running:
                        st.session_state.video_thread = VideoThread(st.session_state.recognizer)
                        st.session_state.video_thread.start_camera()
            with col1_btns[1]:
                if st.button("Stop Camera", use_container_width=True):
                    if st.session_state.video_thread and st.session_state.video_thread.running:
                        st.session_state.video_thread.stop_camera()

        with col2:
            download_text()  # Added download buttons above the text box
            new_text = st.text_area("Composed Text", st.session_state.text, height=150)

                 # Update session state when user edits the text manually
            st.session_state.text = new_text
            if new_text != st.session_state.text:
              st.session_state.text = new_text


            col2_btns = st.columns([1, 1, 1, 1, 1])
            with col2_btns[0]:
                st.button("Capture", on_click=capture_text, use_container_width=True)
            with col2_btns[1]:
                st.button("Clear", on_click=clear_text, use_container_width=True)
            with col2_btns[2]:
                st.button("Space", on_click=add_space, use_container_width=True)
            with col2_btns[3]:  # New Backspace button
                st.button("Backspace", on_click=delete_last_char, use_container_width=True)
            with col2_btns[4]:
                st.button("🔊 Speak", on_click=speak_text, use_container_width=True)
            


        while st.session_state.video_thread and st.session_state.video_thread.running:
            try:
                frame = st.session_state.video_thread.frame_queue.get_nowait()
                frame_placeholder.image(frame, channels="BGR", use_container_width=True)
            except queue.Empty:
                pass
            time.sleep(0.01)

if __name__ == "__main__":
    main()
