import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Path to your audio file
audio_folder = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/graphemes_out_all/phrases30_26_3/wavs/'
# audio_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/TC_all/wavs/cooke_1999-11-19_052.wav'

for file in os.listdir(audio_folder):
    if '.wav' in file:
        audio_path = audio_folder + file
        # Load the audio file
        y, sr = librosa.load(audio_path, sr=22050)  # sr=None keeps the original sampling rate

        # Extract the Mel spectrogram
        # n_mels = number mel channels
        S = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmax=8000)
        # Convert the Mel spectrogram to dB scale
        S_dB = librosa.power_to_db(S, ref=np.max)       # ref=np.max to normalize 


        # Save as .pt:
        # Convert the Mel spectrogram to a PyTorch tensor
        # S_dB_tensor = torch.tensor(S_dB)
        S_dB_tensor = torch.tensor(S_dB * 0.1)      # ** Need to multiply by 0.1 for tensor values to be correct, not sure why?

        # print(S_dB_tensor)

        label = file[:-5]


        # Plot the Mel spectrogram
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(S_dB, sr=22050, hop_length=256, x_axis='time', y_axis='mel', fmax=8000, win_length=1024, n_fft=1024)
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig(f'figures/cooke_{label}_other.png')
        plt.show()
