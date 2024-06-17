import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import torch

# Path to your audio file
audio_path = 'audio/wavs/1994_first_sentence_edit.wav'
# audio_path = 'output3/audio_devset_1994/1994_first_sentence_edit.wav'
# Load the audio file
y, sr = librosa.load(audio_path, sr=22050)  # sr=None keeps the original sampling rate


# Normalize the audio signal


# Extract the Mel spectrogram
# n_mels = number mel channels
S = librosa.feature.melspectrogram(y=y, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_mels=80, fmax=8000)
# Convert the Mel spectrogram to dB scale
S_dB = librosa.power_to_db(S, ref=np.max)


# Save as .pt:
# Convert the Mel spectrogram to a PyTorch tensor
# S_dB_tensor = torch.tensor(S_dB)
S_dB_tensor = torch.tensor(S_dB)

print(S_dB_tensor)


# # Save the tensor as a .pt file
# TODO: not db now, plot 8 looks similar to LJS plot (but lower) so assuming right format 
output_tensor_path = 'audio/mels/1994_first_sentence_edit.pt'
torch.save(S_dB_tensor, output_tensor_path)

print(f"Mel spectrogram saved to {output_tensor_path}")

# # Plot the Mel spectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(S, sr=22050, hop_length=256, x_axis='time', y_axis='mel', fmax=8000)
# plt.colorbar(format='%+2.0f not dB')
# plt.title('Mel spectrogram')
# plt.tight_layout()
# plt.savefig('mels_plots/mel_spectrogram_1994_12.png')
# plt.show()
