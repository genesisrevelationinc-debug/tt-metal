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
    input_text = "Hello, how are you doing today?"
    output_path = "output.wav"
    convert_voice(input_text, output_path)
    print(f"Converted speech saved to {output_path}")


if __name__ == "__main__":
    main()


# TTNN Integration
def ttnn_speecht5_vc(input_ids, speaker_embeddings, device):
    # Initialize TTNN program
    program = Program()
    with program:
        # Define input tensors
        input_tensor = Tensor(shape=input_ids.shape, dtype=torch.int32, device=device)
        speaker_tensor = Tensor(shape=speaker_embeddings.shape, dtype=torch.float32, device=device)

        # Load model weights
        processor, model, vocoder = load_model()

        # Preprocess input
        inputs = processor(input_ids=input_ids, return_tensors="pt")
        input_tensor.from_torch(inputs["input_ids"])
        speaker_tensor.from_torch(speaker_embeddings)

        # Define model layers
        encoder = tt_speecht5_vc.Encoder(model.encoder)
        decoder = tt_speecht5_vc.Decoder(model.decoder)
        vocoder_layer = tt_speecht5_vc.Vocoder(vocoder)

        # Forward pass
        encoder_output = encoder(input_tensor)
        decoder_output = decoder(encoder_output, speaker_tensor)
        speech = vocoder_layer(decoder_output)

        # Define output tensor
        output_tensor = Tensor(shape=speech.shape, dtype=torch.float32, device=device)
        output_tensor.from_torch(speech)

    # Compile and run program
    program.compile()
    program.run()

    return output_tensor.to_torch()


def ttnn_main():
    device = Device()
    input_text = "Hello, how are you doing today?"
    processor, model, _ = load_model()
    inputs = processor(text=input_text, return_tensors="pt")
    speaker_embeddings = load_speaker_embeddings()
    speech = ttnn_speecht5_vc(inputs["input_ids"], speaker_embeddings, device)
    torchaudio.save("ttnn_output.wav", speech, sample_rate=16000)
    print("Converted speech saved to ttnn_output.wav")


if __name__ == "__main__":
    ttnn_main()