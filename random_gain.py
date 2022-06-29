import numpy as np
import librosa
import soundfile as sf
import pathlib
import tensorflow as tf
import random


DATAPATH = "cow3"
output_path = "output3"
data_dir = pathlib.Path(DATAPATH)
# commands = np.array(tf.io.gfile.listdir(str(data_dir)))
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
# print(filenames)
print("NUMBER OF FILE :", len(filenames))


def random_gain(signal, min_gain_factor, max_gain_factor):
    gain_factor = random.uniform(min_gain_factor, max_gain_factor)
    augmented_signal = signal * gain_factor
    return augmented_signal


def single_image():
    signal, sr = librosa.load("test.wav")
    augmented_signal = random_gain(signal, 2, 4)
    sf.write("Augmented_random_gain.wav", augmented_signal, sr)
    print("DONE")
    return augmented_signal


def save_file():
    for y in filenames:
        signal, sr = librosa.load(y)
        print(y)
        augmented_signal = random_gain(signal, 2, 6)
        sf.write(output_path+"/"+y.split('.')
                 [0]+"_random_gain_augmented.wav", augmented_signal, sr)


if __name__ == '__main__':
    # single_image()
    save_file()
