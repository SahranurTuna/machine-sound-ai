import numpy as np
import librosa

SR = 22050
MEL_SIZE = 128
# dB araligini sabit olcekle (dosya bazli z-score sinif farkini siler)
MEL_MIN_DB = -80.0
MEL_MAX_DB = 0.0


def extract_mel(file_path):
    audio, sr = librosa.load(file_path, sr=SR)

    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=MEL_SIZE)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_db = mel_db[:MEL_SIZE, :MEL_SIZE]

    if mel_db.shape[1] < MEL_SIZE:
        mel_db = np.pad(mel_db, ((0, 0), (0, MEL_SIZE - mel_db.shape[1])))

    mel_db = np.clip(mel_db, MEL_MIN_DB, MEL_MAX_DB)
    mel_db = (mel_db - MEL_MIN_DB) / (MEL_MAX_DB - MEL_MIN_DB)

    return mel_db[..., np.newaxis]
