
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

# Data import
# def map_to_array(batch):
#     speech, _ = sf.read(batch["file"])
#     batch["speech"] = speech
#     return batch
#
# ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
# ds = ds.map(map_to_array)
# wav_to_save = ds["speech"][10]
# sf.write("Data/test.wav",wav_to_save, 16000)

transcription_truth = ("NEAR THE FIRE AND THE ORNAMENTS FRED BROUGHT HOME FROM INDIA ON THE MANTLE BOARD").split()

y, s = librosa.load("Data/test.wav", sr=16000)
# for i in np.arange(0.01,0.1,0.01):
#     noise = np.random.normal(0,i * np.max(y),len(y))
#     noisy_wav = np.add(y, noise)
#     f_name = "Data/noise_"+ str(round(i,2)) +"_test.wav"
#     sf.write(f_name,noisy_wav, 16000)

#load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# # tokenize
input_values = tokenizer(y, return_tensors="pt", padding="longest").input_values  # Batch size 1

# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)
transcription = transcription[0].split()

print("Truth:", Counter(transcription_truth))
print("Predicted: ", Counter(transcription))

x = pd.DataFrame(np.array([Counter(transcription_truth).keys(), Counter(transcription_truth).items()]), columns=['word','count'])

print(x)
