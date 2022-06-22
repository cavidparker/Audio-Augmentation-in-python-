import numpy as np
import librosa
import soundfile as sf
import pathlib
import tensorflow as tf


DATAPATH = "clap"
output_path = "output"
data_dir = pathlib.Path(DATAPATH)
commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
# print(filenames)
num_sample = len(filenames)
print(num_sample)


def add_white_noise(signal, noise_factor):
    noise = np.random.normal(0, signal.std(), signal.size)
    augmented_signal = signal + noise*noise_factor
    return augmented_signal


# for single image
def single_image():
    signal, sr = librosa.load("scale.wav")
    augmented_signal = add_white_noise(signal, 0.1)
    augmented_signal = signal + noise * noise_percentage_factor
    return augmented_signal


# for multiple images
def save_file():
    for y in filenames:
        signal, sr = librosa.load(y)
        print(y)
        augmented_signal = add_white_noise(signal, 0.1)
        sf.write(output_path+"/"+y.split('.')
                 [0]+"_augmented.wav", augmented_signal, sr)


if __name__ == '__main__':
    save_file()
    # single_image()
