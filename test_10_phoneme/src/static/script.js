let mediaRecorder;
let audioChunks = [];
let selectedAccent = 'American';
let selectedModel = '';

// Load available models
async function loadModels() {
    const response = await fetch('/models');
    const models = await response.json();
    const select = document.getElementById('model-select');
    select.innerHTML = '';
    models.forEach(m => {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name;
        select.appendChild(opt);
    });
    selectedModel = models[0]?.id || '';
}

loadModels();

// Accent selection
document.querySelectorAll('.accent-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.accent-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        selectedAccent = e.target.dataset.accent;
    });
});

document.getElementById('model-select').addEventListener('change', (e) => {
    selectedModel = e.target.value;
});

// Show/hide listen button when word is selected
document.getElementById('word-select').addEventListener('change', (e) => {
    const listenSection = document.getElementById('listen-section');
    if (e.target.value) {
        listenSection.style.display = 'block';
    } else {
        listenSection.style.display = 'none';
    }
});

// Listen button functionality
document.getElementById('listen-button').addEventListener('click', async () => {
    const word = document.getElementById('word-select').value;
    if (!word) {
        alert('Select a word first');
        return;
    }
    
    const button = document.getElementById('listen-button');
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '⏳ Generating...';
    
    try {
        const response = await fetch('/tts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: word,
                accent: selectedAccent
            })
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'TTS failed');
        }
        
        const audioBlob = await response.blob();
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = new Audio(audioUrl);
        
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
            button.disabled = false;
            button.textContent = originalText;
        };
        
        audio.onerror = () => {
            alert('Error playing audio');
            button.disabled = false;
            button.textContent = originalText;
        };
        
        await audio.play();
    } catch (error) {
        alert(`Error: ${error.message}`);
        button.disabled = false;
        button.textContent = originalText;
    }
});

document.getElementById('record-button').addEventListener('click', async () => {
    audioChunks = [];
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorder = new MediaRecorder(stream);
    
    mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
    mediaRecorder.onstop = sendAudio;
    
    mediaRecorder.start();
    document.getElementById('record-button').disabled = true;
    document.getElementById('stop-button').disabled = false;
});

document.getElementById('stop-button').addEventListener('click', () => {
    mediaRecorder.stop();
    document.getElementById('record-button').disabled = false;
    document.getElementById('stop-button').disabled = true;
});

function getScoreColor(score) {
    if (score >= 80) return '#4CAF50';  // green
    if (score >= 60) return '#FF9800';  // orange
    return '#f44336';                   // red
}

async function sendAudio() {
    const word = document.getElementById('word-select').value;
    if (!word) {
        alert('Select a word first');
        return;
    }
    
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('accent', selectedAccent);
    formData.append('word', word);
    formData.append('model', selectedModel);
    
    const response = await fetch('/analyze', { method: 'POST', body: formData });
    const data = await response.json();
    
    const resultDiv = document.getElementById('result');
    const scoreColor = getScoreColor(data.score);
    
    resultDiv.innerHTML = `
        <div class="result-box">
            <h3>Results</h3>
            <p><strong>Detected:</strong></p>
            <p>eSpeak: <code>${data.transcription}</code></p>
            <p>IPA: <code>${data.detected_ipa}</code></p>
            <hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;">
            <p><strong>Expected (${selectedAccent}):</strong></p>
            <p>eSpeak: <code>${data.expected_espeak}</code></p>
            <p>IPA: <code>${data.expected_ipa}</code></p>
            <hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;">
            <div class="score-display" style="background: ${scoreColor}; color: white; padding: 20px; border-radius: 8px; text-align: center;">
                <p style="font-size: 36px; font-weight: bold; margin: 0;">${data.score}%</p>
                <p style="margin: 5px 0; font-size: 14px;">${data.match ? '✓ Perfect!' : 'Keep practicing'}</p>
            </div>
        </div>
    `;
}
