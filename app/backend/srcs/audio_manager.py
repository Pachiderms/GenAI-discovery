from vosk import Model, KaldiRecognizer
import os

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
    
    async def transcribe_audio(self, audio_file_path):
        try:
            with open(audio_file_path, "rb") as audio:
                recognition_result = self.recognizer.AcceptWaveform(audio.read())
                transcription = recognition_result.get("text", "")
            return transcription
        except Exception as e:
            print(f"Erreur lors de la transcription audio : {e}")
            return "Désolé, une erreur est survenue lors de la transcription de l'audio."