const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');
const statusText = document.getElementById('status-text');

// 1. Charger l'historique au démarrage
window.addEventListener('load', async () => {
    try {
        const response = await fetch('http://localhost:8000/history');
        const data = await response.json();
        data.forEach(msg => {
            appendMessage('user', msg.user_query);
            appendMessage('bot', msg.ai_response);
        });
    } catch (e) {
        console.error("Erreur historique:", e);
    }
});

// 2. Envoyer un message texte
sendBtn.onclick = async () => {
    const text = userInput.value;
    if (!text) return;
    
    appendMessage('user', text);
    userInput.value = '';

    const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({text: text})
    });
    const data = await response.json();
    appendMessage('bot', data.answer);
};

// 3. Gestion simplifiée du Micro (Web Audio API)
let isRecording = false;
micBtn.onclick = () => {
    if (!isRecording) {
        startRecording();
        micBtn.className = 'mic-on';
        statusText.innerText = "Écoute en cours...";
    } else {
        stopRecording();
        micBtn.className = 'mic-off';
        statusText.innerText = "Traitement audio...";
    }
    isRecording = !isRecording;
};

function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.innerText = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}


function startRecording() {
    // Implémentation de l'enregistrement audio avec MediaRecorder
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            const audioChunks = [];
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            }
            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                // Envoi du Blob vers FastAPI
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.wav');
                fetch('http://localhost:8000/process-audio', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    appendMessage('bot', data.transcription);
                    // Optionnel: envoyer la transcription à /ask pour obtenir une réponse
                    return fetch('http://localhost:8000/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: data.transcription})
                    });
                })
                .then(response => response.json())
                .then(data => {
                    appendMessage('bot', data.answer);
                })
                .catch(e => console.error("Erreur audio:", e));
            }
            // Arrêter l'enregistrement après 5 secondes (ou selon ton besoin)
            setTimeout(() => {
                mediaRecorder.stop();
                stream.getTracks().forEach(track => track.stop());
            }, 5000);
        })
        .catch(e => console.error("Erreur accès micro:", e));
}

// Note: Pour startRecording/stopRecording, tu utiliseras MediaRecorder 
// et tu enverras le Blob vers ton endpoint FastAPI /process-audio