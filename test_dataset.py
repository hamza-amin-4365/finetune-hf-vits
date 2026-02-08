from datasets import load_dataset
ds = load_dataset("muhammadsaadgondal/urdu-tts", split="train")
print(ds.column_names)