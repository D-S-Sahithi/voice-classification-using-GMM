import numpy as np
import librosa
import librosa.display
import scipy.signal
import soundfile as sf
import os

def pre_process(file_path):
    # Load the noisy speech signal
    print("File exists:", os.path.exists(file_path))
    print("Absolute path:", os.path.abspath(file_path)) # Replace with your actual file path
    noisy_signal, sr = librosa.load(file_path, sr=None)

    # Compute STFT
    n_fft = 1024  # FFT window size
    hop_length = 512  # Hop length (overlap)
    stft_noisy = librosa.stft(noisy_signal, n_fft=n_fft, hop_length=hop_length)

    # Convert to magnitude and phase
    magnitude_noisy, phase_noisy = np.abs(stft_noisy), np.angle(stft_noisy)

    # Estimate noise from initial silent frames (first 0.5 sec)
    num_frames_noise = int(sr * 0.5 / hop_length)  
    noise_spectrum = np.mean(magnitude_noisy[:, :num_frames_noise], axis=1, keepdims=True)


    # Apply spectral subtraction (ensure non-negative values)
    magnitude_denoised = np.maximum(magnitude_noisy - noise_spectrum, 0)

    # Combine magnitude with original phase
    stft_denoised = magnitude_denoised * np.exp(1j * phase_noisy)

    # Perform inverse STFT to obtain the time-domain signal
    denoised_signal = librosa.istft(stft_denoised, hop_length=hop_length)
    output_file = file_path
    sf.write(output_file, denoised_signal, sr)
    print(f"Denoised speech saved as {output_file}")



