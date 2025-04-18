# Hand Gesture Text Input

A real-time hand gesture recognition system that converts hand gestures into text input. Make hand gestures in front of your webcam to type letters!

## Quick Start

1. Install Python 3.8 or higher
2. Create and activate virtual environment:
   ```
   python -m venv env
   # Windows
   .\env\Scripts\activate
   # Linux/MacOS
   source env/bin/activate
   ```
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the app:
   ```
   streamlit run app.py
   ```

## Training Your Own Model

1. Check dataset structure

2. Run training:
   ```
   python train.py
   ```

## How It Works

### Hand Detection & Feature Extraction
- Uses MediaPipe to detect 21 hand landmarks in real-time
- Each landmark has 3 coordinates (x, y, z)
- Coordinates are normalized relative to wrist position for position-invariance
- Results in 63 features (21 landmarks × 3 coordinates)

### Model Architecture
- Input: 63 normalized coordinates
- Hidden Layer 1: 128 neurons (ReLU) with 30% dropout
- Hidden Layer 2: 64 neurons (ReLU) with 20% dropout
- Output: 26 classes (A-Z) with Softmax

### Streamlit App Flow
1. Video Processing:
   - Continuous webcam feed capture in background thread
   - Frame queue maintains latest video frames
   - Prediction queue stores recent gesture predictions
   - Thread-safe operations using locks

2. User Interface:
   - Live video display showing hand landmarks
   - Text composition area showing captured letters
   - Control buttons:
     - "Capture": Adds current gesture prediction to text
     - "Clear": Resets the text field

3. Real-time Processing:
   - Each frame goes through:
     1. Hand landmark detection (MediaPipe)
     2. Coordinate normalization
     3. Model prediction
     4. Visual feedback (drawing landmarks)
   - Predictions and confidence scores shown on video
   - Multi-threaded design prevents UI freezing

4. Text Composition:
   - Click "Capture" when desired gesture is recognized
   - Predicted letter appears in text area
   - Continuous predictions allow gesture adjustment
   - Thread-safe text updates prevent data races

### Technologies
- MediaPipe: Hand landmark detection
- TensorFlow: Neural network model
- Streamlit: Web interface
- OpenCV: Video capture
