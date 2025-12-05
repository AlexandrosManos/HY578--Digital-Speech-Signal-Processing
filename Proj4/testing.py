import os
import glob
import numpy as np
import h5py
import joblib
from feature_extraction import extract_and_save_mfcc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TIMIT_TEST_DIR = os.path.join(BASE_DIR, "TIMIT", "test")

MODELS_DIR = "speaker_models"
FEATURES_DIR = "features_test_clean"


def test_identification_system():
    models = {}
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.joblib"))

    for mf in model_files:
        speaker_id = os.path.basename(mf).replace('.joblib', '')
        models[speaker_id] = joblib.load(mf)

    os.makedirs(FEATURES_DIR, exist_ok=True)

    if not os.path.exists(TIMIT_TEST_DIR):
        print(f"Error: TIMIT test directory not found [{TIMIT_TEST_DIR}]")
        return

    speakers = [d for d in os.listdir(TIMIT_TEST_DIR) if os.path.isdir(os.path.join(TIMIT_TEST_DIR, d))]

    total_tests = 0
    correct_identifications = 0


    for true_speaker_id in speakers:
        speaker_source_dir = os.path.join(TIMIT_TEST_DIR, true_speaker_id)
        speaker_dest_dir = os.path.join(FEATURES_DIR, true_speaker_id)
        os.makedirs(speaker_dest_dir, exist_ok=True)

        wav_files = glob.glob(os.path.join(speaker_source_dir, "*.wav"))

        for wav_path in wav_files:
            filename = os.path.basename(wav_path)
            h5_path = os.path.join(speaker_dest_dir, filename.replace('.wav', '.h5'))

            if not os.path.exists(h5_path):
                extract_and_save_mfcc(wav_path, h5_path)

            with h5py.File(h5_path, 'r') as hf:
                mfccs = hf['mfccs'][:].T

            best_score = -float('inf')
            predicted_speaker = None

            for model_id, model in models.items():
                try:
                    score = model.score(mfccs)
                    if score > best_score:
                        best_score = score
                        predicted_speaker = model_id
                except:
                    pass

            if predicted_speaker == true_speaker_id:
                correct_identifications += 1

            total_tests += 1

    accuracy = (correct_identifications / total_tests) * 100 if total_tests > 0 else 0

    print(f"Accuracy: {accuracy:.2f}% ({correct_identifications}/{total_tests})")


if __name__ == "__main__":
    test_identification_system()