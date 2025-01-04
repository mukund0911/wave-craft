# import torch
import base64
import numpy as np
from io import BytesIO
from collections import defaultdict

# from TTS.api import TTS
import assemblyai as aai
from pydub import AudioSegment

from backend.models.config import ASSEMBLY_AI_KEY

# from sample_dict import *

SAMPLE_RATE = 22050  

class SpeechModel:
    """_summary_
    """
    def __init__(self, audio_path: str):
        self.audio_path = audio_path

    def wav_to_byte(self, audio_segment):
        buffer = BytesIO()
        audio_segment.export(buffer, format="wav")                          # Convert AudioSegment to bytes
        audio_bytes = buffer.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')  

        return audio_base64

    def byte_to_wav(self, byte_encode):
        audio_decode = base64.b64decode(byte_encode)                           # Convert base64 encoding to wav.
        buffer = BytesIO(audio_decode)
        audio_segment = AudioSegment.from_file(buffer, format="wav")
        return audio_segment
    
    def wav_to_audio_segment(self, wav):
        # Convert the list of integers to a NumPy array
        wav_array = np.array(wav, dtype=np.int16)  # 16-bit PCM format
        
        # Convert the NumPy array to bytes
        wav_bytes = wav_array.tobytes()
        
        # Create an AudioSegment from raw bytes
        audio_segment = AudioSegment(
            data=wav_bytes,
            frame_rate=SAMPLE_RATE,
            sample_width=2,  # 2 bytes per sample (16-bit PCM)
            channels=1  # Assuming mono audio
        )
        
        return audio_segment

    def speech_to_text(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        
        full_audio = AudioSegment.from_file(self.audio_path, format="wav")
        full_audio_base64 = self.wav_to_byte(full_audio)

        # Init Assembly AI API
        aai.settings.api_key = ASSEMBLY_AI_KEY
        config = aai.TranscriptionConfig(speaker_labels=True)

        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(
            self.audio_path,
            config=config
        )
        
        conversations_sep = []
        for index, utterance in enumerate(transcript.utterances):
            words = {'original' : {}, 'modified' : {}}                          # Create a dictionary for each conversation
            start, end = utterance.start, utterance.end                         # Start and end times for each speaker speech

            words['speaker'] = utterance.speaker                                # Save info
            words['original']['text'] = utterance.text
            words['original']['start'] = start
            words['original']['end'] = end

            speaker_audio = full_audio[start:end]                               # Trim this specific speech from the original audio
            audio_base64 = self.wav_to_byte(speaker_audio)
            words['original']['speaker_audio'] = audio_base64
            
            conversations_sep.append({f"conv_{index}" : words})

        return conversations_sep, full_audio_base64

    def text_to_speech(self, conversations_mod):
        pass
        # save_path = "./backend/models/speech_edit/temp"
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # # print(TTS().list_models())
        
        # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

        # # Store the audio files of all the utterances of each speaker
        # speaker_audio_dict = defaultdict(list)
        # for _, conversation in conversations_mod.items():
        #     speaker_label = conversation['speaker']
        #     audio_base64 = conversation["original"]["speaker_audio"]
        #     audio_segment = self.byte_to_wav(audio_base64)

        #     speaker_audio_dict[speaker_label].append(audio_segment)
        
        # # Combine audio segments for each speaker
        # for speaker, segments in dict(speaker_audio_dict).items():
        #     combined_audio = sum(segments)
        #     output_path = f"{save_path}/combined_audio_{speaker}.wav"
        #     combined_audio.export(output_path, format="wav")                   # Convert combined segments to wav for target speech

        # # Traverse through the modified text and convert it to speech matching with all speaker utterances.
        # modified_audio = AudioSegment.silent(duration=0)
        # for index, conversation in conversations_mod.items():
        #     speaker_label = conversation["speaker"]
        #     original_audio = self.byte_to_wav(conversation["original"]["speaker_audio"])
        #     original_audio.export(save_path + f"/original_audio/original_{index}_{speaker_label}.wav", format="wav")

        #     modified_text = conversation['modified']['text']                                # Get modified text

        #     tts.tts_to_file(text=modified_text,
        #                     reference_wav=save_path + f"/original_audio/original_{index}_{speaker_label}.wav",
        #                     speaker_wav=f"{save_path}/combined_audio_{speaker_label}.wav", 
        #                     language="en", 
        #                     file_path=f"{save_path}/modified_audio/modified_{index}_{speaker_label}.wav")
        #     modified_audio += AudioSegment.from_file(f"{save_path}/modified_audio/modified_{index}_{speaker_label}.wav", format="wav") 
        
        # # Export the combined audio to a WAV file
        # output_path = "modified_audio.wav"
        # modified_audio.export(output_path, format="wav")
        # # modified_audio_segment_base64 = self.wav_to_byte(modified_audio_segment)
        # # return modified_audio_segment_base64


# if __name__ == "__main__":
#     _obj = SpeechModel("D:/Personal Projects/wave-craft/backend/models/test_multi_speaker.wav")
#     _obj.text_to_speech(dictt)
    
