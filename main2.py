import librosa
import numpy as np
from keras.models import load_model

def extract_features(file_path, max_length=500):
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=16000)
        # Extract features (example: using Mel-Frequency Cepstral Coefficients)
        mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=40)
        # Pad or trim the feature array to a fixed length
        if mfccs.shape[1] < max_length:
            mfccs = np.pad(mfccs, ((0, 0), (0, max_length - mfccs.shape[1])), mode='constant')
        else:
            mfccs = mfccs[:, :max_length]
        return mfccs
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}")
        return None

def predict_audio(model_path, file_path, max_length=500):
    # Extract features from the audio file
    features = extract_features(file_path, max_length)
    
    if features is not None:
        # Load the saved model
        model = load_model(model_path)

        # Make a prediction
        prediction = model.predict(np.reshape(features, (1, 40, max_length, 1)))

        # Convert the prediction to a binary result (0 or 1)
        result = 1 if prediction > 0.5 else 0

        return result
    else:
        # Return None if there was an error during feature extraction
        return None
    






if __name__ == "__main__":
    model_path = 'cnn.h5'

    user_input_file = input("Enter the path of the .wav file to analyze: ")
    

    prediction_result = predict_audio(model_path, user_input_file)

    if prediction_result is not None:
        # print(f"The prediction result for the audio file is: {prediction_result}")
        if prediction_result==1:
            print(f"The prediction result for the audio file is: fake")
        else:
            print(f"The prediction result for the audio file is: real")


        
    else:
        print("Error during feature extraction or prediction.")