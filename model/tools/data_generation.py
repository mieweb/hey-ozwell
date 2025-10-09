import os
import random
import argparse
from elevenlabs import ElevenLabs

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate Ozwell wake word audio samples using ElevenLabs API.")
    parser.add_argument("--samples", type=int, default=30, help="Number of samples per phrase (default: 30)")
    args = parser.parse_args()
    samples = args.samples

    # Initialize ElevenLabs client
    client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))
    voices = client.voices.get_all()

    # Define target phrases
    phrases = ["Hey Ozwell", "Ozwell I’m done", "Go Ozwell", "Ozwell go"]

    # Generate samples
    for phrase in phrases:
        phrase_dir = f"data/{phrase.lower().replace(' ', '-')}"
        os.makedirs(phrase_dir, exist_ok=True)

        for i in range(samples):
            voice = random.choice(voices.voices)
            print(f"Generating sample {i+1}/{samples} for phrase '{phrase}' using voice '{voice.name}'...")
            audio = client.text_to_speech.convert(
                voice_id=voice.voice_id,
                model_id="eleven_multilingual_v2",
                text=phrase
            )
            file_path = f"{phrase_dir}/{i:03d}_{voice.name}.wav"
            with open(file_path, "wb") as f:
                f.write(audio)

if __name__ == "__main__":
    main()