import os
import streamlit as st
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib


# ... (existing code)

def extract_mfcc_features(audio_data, n_mfcc=25, n_fft=2048, hop_length=512):
    sr = 22050  # Adjust this based on your audio data
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_delta_mfccs = librosa.feature.delta(delta_mfccs)
    combined_features = np.vstack([mfccs, delta_mfccs, delta_delta_mfccs])

    return np.mean(combined_features.T, axis=0)

# ... (existing code)

def analyze_audio(audio_data):
    model_filename = "svm_model.pkl"
    scaler_filename = "scaler.pkl"

    # In the Streamlit application code
    mfcc_features = extract_mfcc_features(audio_data)

    if mfcc_features is not None:
        scaler = joblib.load(scaler_filename)

        # Verify the number of features before scaling
        print(f"Number of features before scaling: {mfcc_features.shape}")

        mfcc_features_scaled = scaler.transform(mfcc_features.reshape(1, -1))

        svm_classifier = joblib.load(model_filename)
        prediction = svm_classifier.predict(mfcc_features_scaled)

        if prediction[0] == 0:
            result = "Audio Submitted is likely to Real"
        else:
            result = "Audio Submitted is likely to Deepfake"
    else:
        result = "Error: Unable to process the input audio."

    return result

# ... (existing code)

def main():
    st.title("Deepfake Voice Detection")

    uploaded_file = st.file_uploader("Upload a .mp3 audio file", type=["mp3"])

    if uploaded_file is not None:
        audio_data = librosa.load(uploaded_file, sr=None)[0]
        result = analyze_audio(audio_data)
        st.success(result)

        # Display the audio player
        st.audio(uploaded_file)

       

    

if __name__ == "__main__":
    main()
