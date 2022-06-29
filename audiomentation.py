from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
import numpy as np
import librosa
import soundfile as sf

augment = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-3, max_semitones=3, p=0.5),
    Shift(min_fraction=-0.1, max_fraction=0.1, p=0.5),
])


if __name__ == '__main__':
    signal, sr = librosa.load("test.wav")
    augmented_signal = augment(signal, sr)
    sf.write("Augmented_audiomentation.wav", augmented_signal, sr)
    print("DONE")
    # print(augmented_signal.shape)
    # print(augmented_signal)
    # print(augmented_signal.dtype)
    # print(augmented_signal.min())
    # print(augmented_signal.max())
    # print(augmented_signal.mean())
    # print(augmented_signal.std())
    # print(augmented_signal.var())
    # print(augmented_signal.sum())
