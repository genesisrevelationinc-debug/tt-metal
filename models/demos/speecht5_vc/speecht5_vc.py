import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torchaudio
import torch.nn.functional as F
from tt_metal import Device, Program, get_program, Tensor, tt_speecht5_vc


def load_model():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return processor, model, vocoder


def load_speaker_embeddings():
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return speaker_embeddings


def preprocess_input(text, processor):
    inputs = processor(text=text, return_tensors="pt")
    return inputs


def generate_speech(inputs, speaker_embeddings, model, vocoder):
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    return speech


def convert_voice(input_text, output_path):
    processor, model, vocoder = load_model()
    speaker_embeddings = load_speaker_embeddings()
    inputs = preprocess_input(input_text, processor)
    speech = generate_speech(inputs, speaker_embeddings, model, vocoder)
    torchaudio.save(output_path, speech, sample_rate=16000)


def main():
    input_text = "Hello, how are you?"
    output_path = "output.wav"
    convert_voice(input_text, output_path)
    print(f"Converted speech saved to {output_path}")


if __name__ == "__main__":
    main()


# TTNN Integration
def ttnn_speecht5_vc(input_ids, speaker_embeddings, device):
    program = Program()
    with program:
        input_tensor = Tensor(shape=input_ids.shape, dtype=torch.int32, device=device)
        speaker_tensor = Tensor(shape=speaker_embeddings.shape, dtype=torch.float32, device=device)
        output_tensor = tt_speecht5_vc(input_tensor, speaker_tensor)
    program.compile()
    program.run()
    return output_tensor


def ttnn_convert_voice(input_text, output_path, device):
    processor, model, vocoder = load_model()
    speaker_embeddings = load_speaker_embeddings()
    inputs = preprocess_input(input_text, processor)
    input_ids = inputs["input_ids"].to(device)
    speaker_embeddings = speaker_embeddings.to(device)
    speech = ttnn_speecht5_vc(input_ids, speaker_embeddings, device)
    torchaudio.save(output_path, speech.cpu(), sample_rate=16000)


def ttnn_main():
    device = Device()
    input_text = "Hello, how are you?"
    output_path = "output_ttnn.wav"
    ttnn_convert_voice(input_text, output_path, device)
    print(f"Converted speech saved to {output_path}")


if __name__ == "__main__":
    ttnn_main()