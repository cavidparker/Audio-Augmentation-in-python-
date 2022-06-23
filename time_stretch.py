import numpy as np
import librosa
import soundfile as sf
import pathlib
import tensorflow as tf


DATAPATH = "clap"
output_path = "output"
data_dir = pathlib.Path(DATAPATH)
# commands = np.array(tf.io.gfile.listdir(str(data_dir)))
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
# print(filenames)
print("NUMBER OF FILE :", len(filenames))


def time_stretch(signal, time_stretch_rate):
    return librosa.effects.time_stretch(signal, time_stretch_rate)


def single_image():
    signal, sr = librosa.load("cow.wav")
    # 0.5 means its slow the sound
    augmented_signal = time_stretch(signal, 0.5)
    sf.write("Augmented_time_stretch.wav", augmented_signal, sr)
    print("DONE")
    return augmented_signal


def save_file():
    for y in filenames:
        signal, sr = librosa.load(y)
        print(y)
        augmented_signal = time_stretch(signal, 0.5)
        sf.write(output_path+"/"+y.split('.')
                 [0]+"_time_augmented.wav", augmented_signal, sr)


if __name__ == '__main__':
    # single_image()
    save_file()
