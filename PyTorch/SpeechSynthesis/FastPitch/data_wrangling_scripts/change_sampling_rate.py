import torchaudio

def change_sample_rate(input_wav_path, output_wav_path, new_sample_rate):
    waveform, original_sample_rate = torchaudio.load(input_wav_path)
    
    if original_sample_rate != new_sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        waveform = resampler(waveform)
    
    torchaudio.save(output_wav_path, waveform, new_sample_rate)

# Run
input_wav_path = 'audio/wavs/1994_first_sentence.wav'
output_wav_path = 'audio/wavs/1994_first_sentence_edit.wav'
new_sample_rate = 22050  # Desired sample rate

change_sample_rate(input_wav_path, output_wav_path, new_sample_rate)