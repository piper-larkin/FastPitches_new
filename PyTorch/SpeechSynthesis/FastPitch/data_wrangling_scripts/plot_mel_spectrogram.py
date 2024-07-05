import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

# Path to your .pt file containing the mel spectrogram
# mels_path = 'audio/mels/1994_first_sentence_edit.pt'
# mels_path = 'LJSpeech-1.1/mels/LJ001-0002.pt'
# mels_path = 'audio/mels/1948_last_sentence_edit.pt'
# mels_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/test_LJ_reagan/mels/LJ001-0001.pt'
mels_path = '/work/tc062/tc062/plarkin/FastPitches/PyTorch/SpeechSynthesis/FastPitch/marie/mels/saved15.pt'

# Load the spectrogram from the .pt file
mel_spectrogram = torch.load(mels_path)
S = mel_spectrogram.numpy()
print(S)
# S_dB = librosa.power_to_db(S, ref=np.max) 
# print(S_dB)

# Convert the torch.Tensor to a NumPy array for visualization
# S = mel_spectrogram.np()
# print(S)
# S_dB = librosa.power_to_db(S, ref=np.max)  
# print(S_dB_tensor)

# # Display mel spectrogram
# plt.figure(figsize=(10, 4))
# librosa.display.specshow(librosa.power_to_db(S, ref=numpy.max), x_axis='time', y_axis='mel')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Mel Spectrogram')
# plt.tight_layout()
# plt.savefig('mel_spectrogram_1994_4.png')
# plt.show()


# Plot the Mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(S, sr=22050, hop_length=256, x_axis='time', y_axis='mel', fmax=8000, win_length=1024, n_fft=1024)
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram test15')
plt.tight_layout()
plt.savefig('marie/test15.png')
plt.show()