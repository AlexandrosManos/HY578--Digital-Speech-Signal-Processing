import os
import glob
import numpy as np
import h5py
import joblib
from sklearn.mixture import GaussianMixture
from feature_extraction import extract_and_save_mfcc

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TIMIT_TRAIN_DIR = os.path.join(BASE_DIR, "TIMIT", "train")
TIMIT_TEST_DIR = os.path.join(BASE_DIR, "TIMIT", "test")

# Experiment Settings
CONFIGS = [
    {"name": "GMM_16_DIAG", "n_components": 16, "covariance_type": "diag"},
    {"name": "GMM_32_DIAG", "n_components": 32, "covariance_type": "diag"}
]

NOISE_LEVELS = [None, 15] # None for clean, 15dB for moderate noise

def load_features_for_speaker(speaker_id, features_dir):
    """
    Loads all MFCC features for a given speaker from the features directory.
    """
    speaker_features_dir = os.path.join(features_dir, speaker_id)
    if not os.path.exists(speaker_features_dir):
        return None
        
    all_mfccs = []
    feature_files = glob.glob(os.path.join(speaker_features_dir, "*.h5"))
    
    for fpath in feature_files:
        with h5py.File(fpath, 'r') as hf:
            mfccs = hf['mfccs'][:]
            all_mfccs.append(mfccs.T)
            
    if not all_mfccs:
        return None
        
    return np.vstack(all_mfccs)

def train_models(config):
    """
    Trains models for all speakers using the specified configuration.
    """
    print(f"\n--- Training Models for Config: {config['name']} ---")
    
    features_base_dir = f"features_train_clean"
    models_dir = f"models_{config['name']}"
    os.makedirs(models_dir, exist_ok=True)
    
    # Ensure features exist (we assume clean training features are common)
    # But for clarity, let's just check/extract them once if not present
    if not os.path.exists(features_base_dir):
         print("Extracting training features...")
         extract_features(TIMIT_TRAIN_DIR, features_base_dir, snr_db=None)

    speakers = [d for d in os.listdir(TIMIT_TRAIN_DIR) if os.path.isdir(os.path.join(TIMIT_TRAIN_DIR, d))]
    
    for speaker_id in speakers:
        model_path = os.path.join(models_dir, f"{speaker_id}.joblib")
        if os.path.exists(model_path):
            continue # Skip if already trained
            
        X = load_features_for_speaker(speaker_id, features_base_dir)
        if X is None:
            continue
            
        gmm = GaussianMixture(n_components=config['n_components'], 
                              covariance_type=config['covariance_type'], 
                              random_state=42,
                              max_iter=100,
                              n_init=1,
                              verbose=0)
        gmm.fit(X)
        joblib.dump(gmm, model_path)
        # print(f"Trained {speaker_id}")

def extract_features(source_dir, dest_dir, snr_db=None):
    """
    Extracts features from source_dir to dest_dir, optionally adding noise.
    """
    os.makedirs(dest_dir, exist_ok=True)
    speakers = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    
    for speaker_id in speakers:
        speaker_source_dir = os.path.join(source_dir, speaker_id)
        speaker_dest_dir = os.path.join(dest_dir, speaker_id)
        os.makedirs(speaker_dest_dir, exist_ok=True)
        
        wav_files = glob.glob(os.path.join(speaker_source_dir, "*.wav"))
        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            h5_filename = filename.replace('.wav', '.h5')
            h5_path = os.path.join(speaker_dest_dir, h5_filename)
            
            if not os.path.exists(h5_path):
                extract_and_save_mfcc(wav_path, h5_path, snr_db=snr_db)

def evaluate_system(config, snr_db):
    """
    Evaluates the system on the test set with specified noise level.
    """
    noise_label = "Clean" if snr_db is None else f"SNR_{snr_db}dB"
    print(f"\n--- Evaluating {config['name']} on {noise_label} Test Data ---")
    
    models_dir = f"models_{config['name']}"
    test_features_dir = f"features_test_{noise_label}"
    
    # Extract test features if needed
    if not os.path.exists(test_features_dir):
        print(f"Extracting test features for {noise_label}...")
        extract_features(TIMIT_TEST_DIR, test_features_dir, snr_db=snr_db)
        
    # Load all models
    models = {}
    model_files = glob.glob(os.path.join(models_dir, "*.joblib"))
    for mf in model_files:
        speaker_id = os.path.basename(mf).replace('.joblib', '')
        models[speaker_id] = joblib.load(mf)
        
    speakers = [d for d in os.listdir(TIMIT_TEST_DIR) if os.path.isdir(os.path.join(TIMIT_TEST_DIR, d))]
    
    total_tests = 0
    correct_identifications = 0
    
    for true_speaker_id in speakers:
        # Load features for this speaker's test files
        # We need to process file by file for evaluation
        speaker_feat_dir = os.path.join(test_features_dir, true_speaker_id)
        if not os.path.exists(speaker_feat_dir):
            continue
            
        feature_files = glob.glob(os.path.join(speaker_feat_dir, "*.h5"))
        
        for fpath in feature_files:
            with h5py.File(fpath, 'r') as hf:
                X = hf['mfccs'][:].T
            
            # Score against all models
            best_score = -float('inf')
            predicted_speaker = None
            
            for model_speaker_id, model in models.items():
                try:
                    score = model.score(X) # Average log-likelihood per sample
                    # Total log-likelihood = score * n_samples
                    # But for comparison, average or total is fine as long as consistent.
                    # Usually sum is better for variable length? 
                    # score() returns average log-likelihood.
                    # We want to maximize P(X|Model).
                    # log P(X|Model) = sum(log p(x_t|Model)) = avg * n_samples
                    # Since n_samples is constant for this specific comparison (same file), avg is fine.
                    if score > best_score:
                        best_score = score
                        predicted_speaker = model_speaker_id
                except:
                    pass
            
            if predicted_speaker == true_speaker_id:
                correct_identifications += 1
            total_tests += 1
            
    accuracy = (correct_identifications / total_tests) * 100 if total_tests > 0 else 0
    print(f"Accuracy: {accuracy:.2f}% ({correct_identifications}/{total_tests})")
    return accuracy

def run_experiment():
    results = {}
    
    # 1. Train both configurations
    for config in CONFIGS:
        train_models(config)
        
    # 2. Evaluate both on Clean and Noisy data
    for config in CONFIGS:
        results[config['name']] = {}
        for snr in NOISE_LEVELS:
            acc = evaluate_system(config, snr)
            label = "Clean" if snr is None else f"SNR_{snr}dB"
            results[config['name']][label] = acc
            
    # 3. Print Summary
    print("\n\n=== Final Results Comparison ===")
    print(f"{'Model':<20} | {'Clean Accuracy':<15} | {'Noisy (15dB) Accuracy':<20}")
    print("-" * 60)
    for config_name, res in results.items():
        clean_acc = res.get("Clean", 0)
        noisy_acc = res.get("SNR_15dB", 0)
        print(f"{config_name:<20} | {clean_acc:.2f}%          | {noisy_acc:.2f}%")

if __name__ == "__main__":
    run_experiment()
