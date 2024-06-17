import torch
import matplotlib.pyplot as plt
import librosa.display
import numpy

# Path to your .pt file containing the mel spectrogram
mels_path = 'audio/mels/1994_first_sentence_edit.pt'
# mels_path = 'LJSpeech-1.1/mels/LJ001-0001.pt'


# Load the spectrogram from the .pt file
mel_spectrogram = torch.load(mels_path)

# Convert the torch.Tensor to a NumPy array for visualization
S = mel_spectrogram.numpy()

# Display mel spectrogram
plt.figure(figsize=(10, 4))
librosa.display.specshow(librosa.power_to_db(S, ref=numpy.max), x_axis='time', y_axis='mel')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel Spectrogram')
plt.tight_layout()
plt.savefig('mel_spectrogram_1994_4.png')
plt.show()
