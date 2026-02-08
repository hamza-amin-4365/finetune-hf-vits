from transformers import VitsModel, VitsTokenizer
import torch
import scipy.io.wavfile

# Loading from your NEW repository!
model = VitsModel.from_pretrained("hamza-amin/mms-tts-urd-train")
tokenizer = VitsTokenizer.from_pretrained("hamza-amin/mms-tts-urd-train")

text = "آپ کا بہت شکریہ، ماڈل اب تیار ہے۔" # "Thank you very much, the model is now ready."
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

scipy.io.wavfile.write("final_urdu_output.wav", rate=model.config.sampling_rate, data=output[0].cpu().numpy())
print("Audio saved as final_urdu_output.wav")



# from transformers import VitsModel, AutoTokenizer
# import torch
# import scipy.io.wavfile

# # Path to your new test output
# model_path = "./tmp/vits_urdu_test"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = VitsModel.from_pretrained(model_path).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_path)

# text = "یہ ایک چھوٹا ٹیسٹ ہے تاکہ آواز کو چیک کیا جا سکے۔" 
# inputs = tokenizer(text, return_tensors="pt").to(device)

# with torch.no_grad():
#     output = model(**inputs).waveform

# scipy.io.wavfile.write("test_small_ds.wav", rate=model.config.sampling_rate, data=output[0].cpu().numpy())
# print("Saved: test_small_ds.wav")