from vosk import Model, KaldiRecognizer
import os
import json
from colored_print import log
import wave
from pydub import AudioSegment

class AudioManager:
    def __init__(self):
        model = None
        if not os.path.exists("vosk-model/vosk-model-small-fr-0.22"):
            print("Veuillez télécharger le modèle Vosk pour le français et le placer dans le dossier 'vosk-model' avec le nom 'vosk-model-small-fr-0.22'.")
        else:
            model = Model("vosk-model/vosk-model-small-fr-0.22")
        if model is None:
            raise Exception("Le modèle Vosk n'a pas pu être chargé. Assurez-vous qu'il est correctement installé.")
        
        self.recognizer = KaldiRecognizer(model, 16000)
        self.recognizer.SetWords(True)
        self.file_path = "tmp_audio.wav"
        self.audio_file = None
    
    async def transcribe_audio(self, audio_file_path):
        try:
            self.audio_file = AudioSegment.from_file(audio_file_path)
            self.audio_file = self.audio_file.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            self.audio_file.export(self.file_path, format="wav")
            with wave.open(self.file_path, "rb") as audio:                
                results = []
                
                while True:
                    data = audio.readframes(4000)
                    if len(data) == 0:
                        break
                    if self.recognizer.AcceptWaveform(data):
                        chunk_result = json.loads(self.recognizer.Result())
                        results.append(chunk_result.get("text", ""))

            final_result = json.loads(self.recognizer.FinalResult())
            results.append(final_result.get("text", ""))
            transcription = " ".join(results).strip()

            log.info(f"Transcription obtenue : {transcription}")
            return transcription
        except Exception as e:
            log.error(f"Erreur lors de la transcription audio : {e}")
            return None