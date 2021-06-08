from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from datasets import load_dataset
import soundfile as sf
import torch
import numpy as np
# load model and tokenizer
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

# define function to read in sound file
def map_to_array(batch):
    speech, _ = sf.read(batch["file"])
    print(_)
    batch["speech"] = speech
    return batch

# load dummy dataset and read soundfiles
ds = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")
ds = ds.map(map_to_array)

print(np.array(ds["speech"][1]).shape)

# tokenize
input_values = tokenizer(np.array(ds["speech"][1]), return_tensors="pt", padding="longest").input_values  # Batch size 1
# retrieve logits
logits = model(input_values).logits

# take argmax and decode
predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)
print(transcription)
