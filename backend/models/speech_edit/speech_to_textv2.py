import torch
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import assemblyai as aai
from flask import jsonify
from pydub import AudioSegment

from backend.models.config import ASSEMBLY_AI_KEY

class SpeechModel:
    """_summary_
    """
    def __init__(self, audio_path: str):
        self.audio_path = audio_path

    def speech_to_text(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # full_audio = AudioSegment.from_file(self.audio_path, format="wav")

        aai.settings.api_key = ASSEMBLY_AI_KEY
        config = aai.TranscriptionConfig(speaker_labels=True)

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(
        self.audio_path,
        config=config
        )

        conversations_sep = []
        for index, utterance in enumerate(transcript.utterances):
            words = {'original' : {}, 'modified' : {}}                                              # Create a dictionary for each conversation
            start, end = utterance.start, utterance.end             # Start and end times for each speaker speech
            # speaker_audio = full_audio[start:end]                 # Trim this specific speech from the original audio

            words['speaker'] = utterance.speaker                    # Save info
            words['original']['text'] = utterance.text
            words['original']['start'] = start
            words['original']['end'] = end
            # words['original']['speaker_audio'] = speaker_audio
            
            conversations_sep.append({f"conv_{index}" : words})

        return conversations_sep

    def text_to_speech(self, modified_convs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
        tts.tts_to_file(text="It took me quite a long time to develop a voice and now that I have it I am not going to be silent.", speaker_wav="D:/Personal Projects/wave-craft/backend/models/test_multi_speaker.wav", language="en", file_path="output.wav")

        # wav = tts.tts(text="Hello world!", speaker_wav="./test_multi_speaker.wav", language="en")
            


if __name__ == "__main__":
    _obj = SpeechModel("D:/Personal Projects/wave-craft/backend/models/test_multi_speaker.wav")
    # _obj.text_to_speech("None")
    json_output = _obj.speech_to_text()
    print(json_output)
