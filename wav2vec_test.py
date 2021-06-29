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
word_df = pd.read_csv("Results/word_results.csv")
accuracy_df = pd.read_csv("Results/accuracy_results.csv")
inputs_df = pd.read_csv("Data/transcription.csv")

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
noise = np.arange(0,0.1,0.005)
f_dir = "Data/"
for index,f_name in enumerate(inputs_df["file name"]):

    print("processing file: ", f_name)
    y, s = librosa.load((f_dir + f_name), sr=16000)
    acc_arr = []
    if f_name not in np.asarray(word_df['f_name']) and f_name not in np.asarray(accuracy_df['f_name']):
        for i in noise:
            print("processing noise", str(i))
            noisy_wav = np.add(y, np.random.normal(0, i * np.max(y), len(y)))
            input_values = tokenizer(noisy_wav, return_tensors="pt", padding="longest").input_values  # Batch size 1
            logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.batch_decode(predicted_ids)
            transcription = transcription[0].split()

            t = np.asarray(inputs_df.iloc[[index]]["Transcription"])[0].split()

            truth_df = pd.DataFrame.from_dict(Counter(t), orient='index').reset_index()
            truth_df = truth_df.rename(columns={"index": "index", 0: "truth_count"})

            pred_df = pd.DataFrame.from_dict(Counter(transcription), orient='index').reset_index()
            pred_df = pred_df.rename(columns={"index": "index", 0: "pred_count"})

            entry_df = truth_df.merge(pred_df, how = "left", on = "index")
            entry_df['pred_count'] = entry_df['pred_count'].fillna(0)

            entry_df['noise'] = i
            entry_df['f_name'] = f_name

            word_df = pd.concat([word_df,entry_df])
            acc = (entry_df['pred_count'].sum())/(entry_df['truth_count'].sum())
            acc_arr.append(acc)

        # Add in code to append accuracies by file to accuracy_df
        fname_list = np.repeat(f_name, len(acc_arr))

        # Create the pandas DataFrame
        accuracy_entry = pd.DataFrame({'f_name' : fname_list,
                                       'noise' : noise,
                                       'accuracy' : acc_arr})

        accuracy_df = pd.concat([accuracy_df, accuracy_entry])

        word_df.to_csv("Results/word_results.csv", index=False)
        accuracy_df.to_csv("Results/accuracy_results.csv", index=False)

    else:
        print("File", f_name,"has already been processed")
        continue
