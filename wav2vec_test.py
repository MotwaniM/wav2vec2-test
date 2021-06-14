
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
df = pd.read_csv("Results/test.csv")
#load model and tokenizer


f_name = "test.wav"
f_dir = "Data/"
y, s = librosa.load((f_dir + f_name), sr=16000)
acc_arr = []
noise = np.arange(0,0.1,0.005)
np.random.seed(1)
if f_name not in np.asarray(df['f_name']):
    tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

    for i in noise:
        noisy_wav = np.add(y, np.random.normal(0,i * np.max(y),len(y)))
        input_values = tokenizer(noisy_wav, return_tensors="pt", padding="longest").input_values  # Batch size 1
        logits = model(input_values).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = tokenizer.batch_decode(predicted_ids)
        transcription = transcription[0].split()

        truth_df = pd.DataFrame.from_dict(Counter(transcription_truth), orient='index').reset_index()
        truth_df = truth_df.rename(columns={"index": "index", 0: "truth_count"})

        pred_df = pd.DataFrame.from_dict(Counter(transcription), orient='index').reset_index()
        pred_df = pred_df.rename(columns={"index": "index", 0: "pred_count"})

        final_df = truth_df.merge(pred_df, how = "left", on = "index")
        final_df['pred_count'] = final_df['pred_count'].fillna(0)

        final_df['noise'] = i
        final_df['f_name'] = f_name

        # df = pd.concat([df,final_df])
        acc = (final_df['pred_count'].sum())/(final_df['truth_count'].sum())
        print(acc)
        acc_arr.append(acc)
    plt.plot(noise, acc_arr, "r.")
    plt.xlabel("Gaussian noise spread")
    plt.ylabel("Accuracy (ratio of predicted to actual counts)")
    plt.grid()
    plt.savefig("Mantle_board_acc.png", dpi=300, bbox_inches = "tight")

    final_name = "Results/test.csv"
    df.to_csv(final_name, index=False)

else:
    print("File has already been processed")
