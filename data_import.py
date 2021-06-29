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

def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    batch["speech"] = speech
    return batch

ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)
for i in range(0,5):
    wav_to_save = ds["speech"][i]
    f_name = "Data/test" +str(i)+ ".wav"
    sf.write(f_name,wav_to_save, 16000)
