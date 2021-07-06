from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
from os import path
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import librosa
from collections import Counter
import pandas as pd

np.random.seed(1)
inputs_df = pd.read_csv("Data/transcription.csv")

noise = np.arange(0,0.1,0.005)
f_dir = "Data/"

# In seconds
max_time = 20


def moving_average(audio, N):

    """
    Computes the moving average of speech data to find speech pauses

    INPUTS
    audio: Array of input speech data
    N: window size for moving average

    RETURNS
    returns moving_aves: an array of moving average documentation
    """

    cumsum, moving_aves = [0], []
    audio_abs = abs(audio)
    for i, x in enumerate(audio_abs, 1):
        cumsum.append(cumsum[i-1] + x)
        if i>=N:
            moving_ave = (cumsum[i] - cumsum[i-N])/N
            moving_aves.append(moving_ave)
    return(moving_aves)


def split_file(audio, sample_rate):

    """
    Split audio file greater than max_time (seconds), to prevent exceeding PC RAM limits

    INPUTS
    audio: Array of input speech data
    sample_rate: audio file sample rate

    RETURNS
    chunks: an array of audio speech split into RAM manageable chunks if audio length > max time
    or
    audio: input audio file if audio length < max time in an array
    """

    file_length = len(audio)/sample_rate
    chunks = []
    window_size = 800
    if file_length > max_time:
        moving_aves = moving_average(audio, window_size)

        silence_threshold = 0.05 * np.max(moving_aves)
        silence_mask = [moving_aves < silence_threshold]

        cut_number = int(np.ceil(file_length/max_time) - 1)
        ideal_points = np.linspace(1,cut_number, cut_number) * max_time * sample_rate

        silent_region = np.ceil(np.linspace(0, len(moving_aves), len(moving_aves))[silence_mask]).astype(np.int)

        cut_points = [0] + [silent_region[np.abs(silent_region - val).argmin()] + window_size for val in ideal_points] + [len(audio)]
        for i in range(0, len(cut_points)-1):
            chunks.append(audio[cut_points[i]:cut_points[i+1]])


        return(chunks)
    else:
        return([audio])
