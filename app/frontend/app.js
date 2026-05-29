const chatBox = document.getElementById('chat-box');
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');
const micBtn = document.getElementById('mic-btn');

// Charger l'historique au démarrage
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

// Envoyer un message texte
sendBtn.onclick = async () => {
    const text = userInput.value;
    if (!text) return;
    
    appendMessage('user', text);
    userInput.value = '';

    try {
        const response = await fetch('http://localhost:8000/ask', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({question: text})
        });

        if (!response.ok) {
            throw new Error(`Erreur serveur: ${response.status}`);
        }

        // botMsg = appendMessage('bot', "...");
        const data = await response.json();
        appendMessage('bot', data.answer);
    } catch (e) {
        console.error("Erreur envoi message:", e);
        appendMessage('bot', "Désolé, une erreur est survenue lors de l'envoi de votre message.");
    }
};

// Gestion du Micro
let isRecording = false;
micBtn.onclick = () => {
    if (!isRecording) {
        startRecording();
        micBtn.className = 'mic-on';
    } else {
        micBtn.className = 'mic-off';
    }
    isRecording = !isRecording;
};

// Fonction pour afficher les messages dans le chat
function appendMessage(sender, text) {
    const div = document.createElement('div');
    div.className = `message ${sender}`;
    div.innerText = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
}

// Implémentation de l'enregistrement audio avec MediaRecorder
function startRecording() {
    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();
            let audioChunks = [];
            mediaRecorder.ondataavailable = event => {
                audioChunks.push(event.data);
            }
            mediaRecorder.onstop = () => {
                isRecording = false;
                micBtn.className = 'mic-off';
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                audioChunks = [];
                // Envoi du Blob vers FastAPI
                const formData = new FormData();
                formData.append('file', audioBlob, 'recording.wav');
                fetch('http://localhost:8000/transcribe', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (! data.transcription) {
                        appendMessage('user', 'Désolé une erreur est survenue pendant la transcription de votre message.');
                        return;
                    }
                    appendMessage('user', data.transcription);
                    return fetch('http://localhost:8000/ask', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({question: data.transcription})
                    });
                })
                .then(response => response.json())
                .then(data => {
                    appendMessage('bot', data.answer);
                })
                .catch(e => console.error("Erreur audio:", e));
            }
            // Arrêter l'enregistrement après 5 secondes
            setTimeout(() => {
                mediaRecorder.stop();
                isRecording = false;
                micBtn.className = 'mic-off';
                stream.getTracks().forEach(track => track.stop());
            }, 10000);
        })
        .catch(e => {
            console.error("Erreur accès micro:", e);
            isRecording = false;
            micBtn.className = 'mic-off';
        });
}

