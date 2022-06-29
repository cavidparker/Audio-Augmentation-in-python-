import numpy as np
import librosa
import soundfile as sf
import pathlib
import tensorflow as tf


DATAPATH = "cow3"
output_path = "output3"
data_dir = pathlib.Path(DATAPATH)
# commands = np.array(tf.io.gfile.listdir(str(data_dir)))
filenames = tf.io.gfile.glob(str(data_dir) + '/*')
# print(filenames)
print("NUMBER OF FILE :", len(filenames))


def pitch_scale(signal, sr, num_semitones):
    return librosa.effects.pitch_shift(signal, sr, num_semitones)


def single_image():
    signal, sr = librosa.load("test.wav")
    augmented_signal = pitch_scale(signal, sr, 7)
    sf.write("pitch_scale_augmented.wav", augmented_signal, sr)
    print("DONE")
    return augmented_signal


def save_file():
    for y in filenames:
        signal, sr = librosa.load(y)
        print(y)
        augmented_signal = pitch_scale(signal, sr, 7)
        sf.write(output_path+"/"+y.split('.')
                 [0]+"_pitch_scale_augmented.wav", augmented_signal, sr)


if __name__ == '__main__':
    # single_image()
    save_file()
