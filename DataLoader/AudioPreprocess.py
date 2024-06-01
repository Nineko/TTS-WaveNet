import torch
import torch.nn.functional as F
import numpy as np
import librosa

def mu_law_encoding(audio, mu=255):
    audio = np.clip(audio, -1, 1)
    mu_law_encoded = np.sign(audio) * np.log1p(mu * np.abs(audio)) / np.log1p(mu)
    return mu_law_encoded

def quantize(mu_law_encoded, num_classes=256):
    mu_law_encoded = (mu_law_encoded + 1) / 2 * (num_classes - 1)
    return mu_law_encoded.astype(np.int64)

def load_and_process_audio(file_path, target_sr=16000, target_length=64000):
    audio, sr = librosa.load(file_path, sr=target_sr)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), 'constant')
    mu_law_encoded = mu_law_encoding(audio)
    quantized = quantize(mu_law_encoded)
    return quantized

def compute_mel_spectrogram(audio, sr=16000, n_mels=80, n_fft=1024, hop_length=256,target_length=64000):
    n_fft = int(50 / 1000 * sr)  # 幀長 50ms
    hop_length = int(12.5 / 1000 * sr)  # 幀移 12.5ms
    win_length = n_fft  # 窗長等於 n_fft
    window = 'hann'  # 漢寧窗
    audio = audio.astype(np.float32)
    # 計算 STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=window)
    # 計算 Mel 頻譜
    n_mels = 80  # 使用 80 個 Mel 頻帶
    mel_spectrogram = librosa.feature.melspectrogram(S=np.abs(stft)**2, sr=sr, n_mels=n_mels)
    
    return interpolate_mel_spectrogram(mel_spectrogram,target_length)

def interpolate_mel_spectrogram(mel_spectrogram, target_length):
    mel_spectrogram_tensor = torch.tensor(mel_spectrogram).unsqueeze(0)
    mel_spectrogram_tensor = F.interpolate(mel_spectrogram_tensor, size=target_length, mode='linear', align_corners=True)
    return mel_spectrogram_tensor

def one_hot_encode(quantized_audio, num_classes=256):
    audio_tensor = torch.tensor(quantized_audio).unsqueeze(0)  # 增加 batch 維度
    one_hot = F.one_hot(audio_tensor, num_classes=num_classes).float()
    one_hot = one_hot.transpose(1, 2)  # 轉換形狀為 (batch, num_classes, seq_length)
    return one_hot

def Load(audio_path):
    audio = load_and_process_audio(audio_path)
    mel_spectrogram = compute_mel_spectrogram(audio,target_length = audio.shape[-1])
    audio_final = one_hot_encode(audio)
    return audio_final,mel_spectrogram

