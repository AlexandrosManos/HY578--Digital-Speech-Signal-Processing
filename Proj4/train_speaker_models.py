import os
import glob
import numpy as np
import h5py
import joblib
from sklearn.mixture import GaussianMixture
from feature_extraction import extract_and_save_mfcc

# Configuration
TIMIT_TRAIN_DIR = "/Users/pswmi64/Desktop/my-projects/hy578/HY578--Digital-Speech-Signal-Processing/Proj4/TIMIT/train"
FEATURES_DIR = "speaker_features"
MODELS_DIR = "speaker_models"
N_COMPONENTS = 16  # Number of Gaussians
COVARIANCE_TYPE = 'diag' # Diagonal covariance is standard for speaker ID

def load_features_for_speaker(speaker_id):
    """
    Loads all MFCC features for a given speaker from the features directory.
    """
    speaker_features_dir = os.path.join(FEATURES_DIR, speaker_id)
    if not os.path.exists(speaker_features_dir):
        print(f"No features found for speaker {speaker_id}")
        return None
        
    all_mfccs = []
    feature_files = glob.glob(os.path.join(speaker_features_dir, "*.h5"))
    
    for fpath in feature_files:
        with h5py.File(fpath, 'r') as hf:
            mfccs = hf['mfccs'][:]
            # mfccs shape is (n_mfcc, time). We need (time, n_mfcc) for sklearn
            all_mfccs.append(mfccs.T)
            
    if not all_mfccs:
        return None
        
    # Concatenate all features along the time axis
    return np.vstack(all_mfccs)

def train_gmm_for_speaker(speaker_id, X):
    """
    Trains a GMM for a specific speaker.
    """
    print(f"Training GMM for speaker {speaker_id} with data shape {X.shape}...")
    gmm = GaussianMixture(n_components=N_COMPONENTS, 
                          covariance_type=COVARIANCE_TYPE, 
                          random_state=42,
                          max_iter=100,
                          n_init=1,
                          verbose=0)
    gmm.fit(X)
    return gmm

def process_all_speakers():
    # Create output directories
    os.makedirs(FEATURES_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Get list of speakers
    speakers = [d for d in os.listdir(TIMIT_TRAIN_DIR) if os.path.isdir(os.path.join(TIMIT_TRAIN_DIR, d))]
    print(f"Found {len(speakers)} speakers.")
    
    for speaker_id in speakers:
        speaker_dir = os.path.join(TIMIT_TRAIN_DIR, speaker_id)
        wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        
        # 1. Feature Extraction
        speaker_feat_dir = os.path.join(FEATURES_DIR, speaker_id)
        os.makedirs(speaker_feat_dir, exist_ok=True)
        
        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            h5_filename = filename.replace('.wav', '.h5')
            h5_path = os.path.join(speaker_feat_dir, h5_filename)
            
            if not os.path.exists(h5_path):
                # Extract if not already exists
                extract_and_save_mfcc(wav_path, h5_path)
        
        # 2. Load Features
        X = load_features_for_speaker(speaker_id)
        
        if X is None:
            print(f"Skipping speaker {speaker_id} due to missing features.")
            continue
            
        # 3. Train GMM
        gmm = train_gmm_for_speaker(speaker_id, X)
        
        # 4. Save Model
        model_path = os.path.join(MODELS_DIR, f"{speaker_id}.joblib")
        joblib.dump(gmm, model_path)
        print(f"Saved model to {model_path}")

if __name__ == "__main__":
    process_all_speakers()
