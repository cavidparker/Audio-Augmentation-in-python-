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


def inverse_polarity(signal):
    augmented_signal = signal * -0.2
    return augmented_signal


def single_image():
    signal, sr = librosa.load("cow.wav")
    augmented_signal = inverse_polarity(signal)
    sf.write("Augmented_inverse_polarity.wav", augmented_signal, sr)
    print("DONE")
    return augmented_signal


def save_file():
    for y in filenames:
        signal, sr = librosa.load(y)
        print(y)
        augmented_signal = inverse_polarity(signal)
        sf.write(output_path+"/"+y.split('.')
                 [0]+"_inverse_polarity_augmented.wav", augmented_signal, sr)


if __name__ == '__main__':
    # single_image()
    save_file()
