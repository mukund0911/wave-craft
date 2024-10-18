import assemblyai as aai

aai.settings.api_key = "6c31a408f33646c0b271580e6db8857d"

def speech_to_text(PATH):
    config = aai.TranscriptionConfig(speaker_labels=True)

    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
    PATH,
    config=config
    )

    converted_text = ""
    for utterance in transcript.utterances:
        converted_text += f"Speaker {utterance.speaker}: {utterance.text}\n"

    return converted_text
