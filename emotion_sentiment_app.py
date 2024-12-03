import streamlit as st
import librosa
import numpy as np
from pydub import AudioSegment
import tensorflow as tf
import tempfile
import os
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('model/model_03122024_400.h5')

@st.cache_data
def preprocess_audio(path):
    _, sr = librosa.load(path, sr=None)  # Ensure loading with the original sample rate
    raw_audio = AudioSegment.from_file(path)
    
    samples = np.array(raw_audio.get_array_of_samples(), dtype='float32')
    trimmed, _ = librosa.effects.trim(samples, top_db=25)
    
    # Ensure the length is at least as long as needed
    target_length = 180000
    if len(trimmed) < target_length:
        padded = np.pad(trimmed, (0, target_length - len(trimmed)), 'constant')
    else:
        padded = trimmed[:target_length]  # Truncate if longer
    
    return padded, sr


def extract_features(y, sr):
    FRAME_LENGTH = 2048
    HOP_LENGTH = 512
    
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=HOP_LENGTH)
    
    # Check if features have the same frame count
    min_frames = min(zcr.shape[1], rms.shape[1], mfccs.shape[1])
    zcr = zcr[:, :min_frames]
    rms = rms[:, :min_frames]
    mfccs = mfccs[:, :min_frames]
    
    # Concatenate features
    features = np.concatenate((zcr, rms, mfccs), axis=0)
    features = np.expand_dims(features.T, axis=0)  # Add batch dimension
    
    return features.astype('float32')

def get_emotion_label(prediction):
    emotion_map = {
        0: 'neutral',
        1: 'happy',
        2: 'sad',
        3: 'angry',
        4: 'fear',
        5: 'disgust'
    }
    return emotion_map[prediction]

# Streamlit UI
st.title("Audio Emotion Analysis")
st.write("Upload an audio file to analyze its emotional content")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3', 'ogg'])

if uploaded_file is not None:
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        st.audio(uploaded_file, format='audio/ogg', start_time=0)
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name
        
        status_text.text("Processing audio...")
        progress_bar.progress(30)
        
        # Preprocess audio
        y, sr = preprocess_audio(temp_path)
        progress_bar.progress(60)
        
        # Extract features
        features = extract_features(y, sr)
        progress_bar.progress(80)
        
        # Load model and make prediction
        st.write(f"Feature shape: {features.shape}")
        model = load_model()
        prediction = model.predict(features, batch_size=1)
        st.write(f"Prediction shape: {prediction.shape}")
        predicted_class = np.argmax(prediction[0])
        emotion = get_emotion_label(predicted_class)
        progress_bar.progress(100)
        
        # Display results
        st.success("Analysis Complete!")
        st.write("## Results")
        st.write(f"Detected Emotion: **{emotion.upper()}**")
        
        # Display confidence scores
        st.write("### Confidence Scores")
        emotion_names = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust']
        confidence_scores = prediction[0]
        
        # Create DataFrame for visualization
        data = pd.DataFrame({
            'Emotion': emotion_names,
            'Confidence': confidence_scores
        })
        st.bar_chart(data.set_index('Emotion'))  # Plot the bar chart
        
        # Cleanup
        os.unlink(temp_path)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if 'temp_path' in locals():
            os.unlink(temp_path)

st.markdown("""
---
### Instructions:
1. Upload an audio file (WAV, MP3, or OGG format)
2. Wait for the analysis to complete
3. View the detected emotion and confidence scores

Note: For best results, use clear audio recordings with minimal background noise.
""")