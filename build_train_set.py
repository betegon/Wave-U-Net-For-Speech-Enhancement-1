'''
File to carry out combining the Voice Bank Corpus data for training the Wave-U-Net model.
It merges noise and clean data into folders, preparared to feed the Neural net.
'''
from progress.bar import IncrementalBar
import os
import argparse
import librosa


def parser():
    '''
    Argument parser.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_source",
        required=True,
        type=str,
        help="Source directory containing clean audio files from the Voice Bank Corpus (VCTK)] data set")
    parser.add_argument(
        "--noisy_source",
        required=True,
        type=str,
        help="Source directory containing the contaminated audio files from the Voice Bank Corpus (VCTK)] data set")
    parser.add_argument(
        "--sampling_rate",
        required=False,
        type=int,
        default=16000,
        help="Sampling rate for audio files (Hz). Default to 16000Hz.")
    parser.add_argument(
        "--out_directory",
        required=False,
        type=str,
        default="train_set_built"
        help="Destination directory. If it does not exits, it will be created by this script. Default to train_set_built")
    return parser.parse_args()


if __name__ == "__main__":
    args = parser()
    if not os.path.exists(args.out_directory):
        os.mkdir(args.out_directory)

    files = os.listdir(args.clean_source)
    for file_name in IncrementalBar('Processing').iter(files):
        if not os.path.exists("{0}/{1}".format(args.out_directory, file_name[:-4])):
            os.mkdir("{0}/{1}".format(args.out_directory, file_name[:-4]))

        clean_source_file = "{0}/{1}".format(args.clean_source, file_name)
        clean, _ = librosa.load(clean_source_file, args.sampling_rate)
        librosa.output.write_wav(
            "{0}/{1}/clean.wav".format(args.out_directory, file_name[:-4]), clean, 16000)

        mix_source_file = "{0}/{1}".format(args.noisy_source, file_name)
        mix, _ = librosa.load(mix_source_file, args.sampling_rate)
        librosa.output.write_wav(
            "{0}/{1}/mixed.wav".format(args.out_directory, file_name[:-4]), mix,args.sampling_rate)
        librosa.output.write_wav(
            "{0}/{1}/noise.wav".format(args.out_directory, file_name[:-4]), mix-clean, args.sampling_rate)
