let mediaRecorder;
let audioChunks = [];
let selectedAccent = 'en-US';
let selectedModel = '';
let selectedLevel = '';
let selectedCategory = '';
let selectedSoundId = null;
let selectedSound = null;
let selectedWord = '';
let referenceEspeak = '';

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

// Load levels
async function loadLevels() {
    const response = await fetch('/levels');
    const levels = await response.json();
    const select = document.getElementById('level-select');
    select.innerHTML = '<option value="">Choose a level...</option>';
    levels.forEach(level => {
        const opt = document.createElement('option');
        opt.value = level;
        opt.textContent = level;
        select.appendChild(opt);
    });
}

loadLevels();

// Level selection
document.getElementById('level-select').addEventListener('change', async (e) => {
    selectedLevel = e.target.value;
    if (selectedLevel) {
        await loadCategories();
        document.getElementById('category-heading').style.display = 'block';
        document.getElementById('category-select').style.display = 'block';
    } else {
        hideAllBelow('category-heading');
    }
});

// Load categories
async function loadCategories() {
    if (!selectedLevel) return;
    const response = await fetch(`/categories?level=${selectedLevel}`);
    const categories = await response.json();
    const select = document.getElementById('category-select');
    select.innerHTML = '<option value="">Choose a category...</option>';
    categories.forEach(category => {
        const opt = document.createElement('option');
        opt.value = category;
        opt.textContent = category;
        select.appendChild(opt);
    });
}

// Category selection
document.getElementById('category-select').addEventListener('change', async (e) => {
    selectedCategory = e.target.value;
    if (selectedCategory) {
        await loadSounds();
        document.getElementById('sound-heading').style.display = 'block';
        document.getElementById('sound-select').style.display = 'block';
    } else {
        hideAllBelow('sound-heading');
    }
});

// Load sounds
async function loadSounds() {
    if (!selectedLevel || !selectedCategory) return;
    const response = await fetch(`/sounds?level=${selectedLevel}&category=${selectedCategory}`);
    const sounds = await response.json();
    const select = document.getElementById('sound-select');
    select.innerHTML = '<option value="">Choose a sound...</option>';
    sounds.forEach(sound => {
        const opt = document.createElement('option');
        opt.value = sound.id;
        opt.textContent = `${sound.sound} (${sound.ipa}) [${sound.es}]`;
        select.appendChild(opt);
    });
}

// Sound selection
document.getElementById('sound-select').addEventListener('change', async (e) => {
    selectedSoundId = e.target.value ? parseInt(e.target.value) : null;
    if (selectedSoundId !== null) {
        await loadWords();
        document.getElementById('word-heading').style.display = 'block';
        document.getElementById('word-buttons').style.display = 'block';
    } else {
        hideAllBelow('word-heading');
    }
});

// Load words
async function loadWords() {
    if (selectedSoundId === null) return;
    const response = await fetch(`/sound/${selectedSoundId}/words?level=${selectedLevel}&category=${selectedCategory}&accent=${selectedAccent}`);
    const data = await response.json();
    selectedSound = data.sound;
    referenceEspeak = data.reference_es;
    
    const wordButtons = document.getElementById('word-buttons');
    wordButtons.innerHTML = '';
    
    data.words.forEach(word => {
        const btn = document.createElement('button');
        btn.className = 'example-btn';
        btn.textContent = word;
        btn.dataset.word = word;
        btn.addEventListener('click', () => {
            document.querySelectorAll('.example-btn').forEach(b => b.classList.remove('selected'));
            btn.classList.add('selected');
            selectedWord = word;
            document.getElementById('accent-heading').style.display = 'block';
            document.getElementById('accent-section').style.display = 'block';
            document.getElementById('listen-section').style.display = 'block';
        });
        wordButtons.appendChild(btn);
    });
}

// Accent selection
document.getElementById('accent-select').addEventListener('change', async (e) => {
    selectedAccent = e.target.value;
    if (selectedSoundId !== null) {
        await loadWords();
    }
});

// Hide all elements below a certain point
function hideAllBelow(elementId) {
    const elements = ['category-heading', 'category-select', 'sound-heading', 'sound-select', 
                     'word-heading', 'word-buttons', 'accent-heading', 'accent-section', 'listen-section'];
    const startIndex = elements.indexOf(elementId);
    for (let i = startIndex; i < elements.length; i++) {
        document.getElementById(elements[i]).style.display = 'none';
    }
    selectedWord = '';
}

document.getElementById('model-select').addEventListener('change', (e) => {
    selectedModel = e.target.value;
});

// Listen button - works for both sounds and words
document.getElementById('listen-button').addEventListener('click', async () => {
    if (!selectedWord) {
        alert('Select a word or sound first');
        return;
    }
    
    const button = document.getElementById('listen-button');
    const originalText = button.textContent;
    button.disabled = true;
    button.textContent = '⏳ Generating...';
    
    try {
        let response;
        // If selected word is the sound itself, use espeak phonemes
        // Otherwise use regular TTS (which uses espeak-ng)
        if (selectedWord === selectedSound && referenceEspeak) {
            response = await fetch('/tts-espeak', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({espeak: referenceEspeak, accent: selectedAccent})
            });
        } else {
            response = await fetch('/tts', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: selectedWord, accent: selectedAccent})
            });
        }
        
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
    const analyzingStatus = document.getElementById('analyzing-status');
    analyzingStatus.style.display = 'none';
    
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        
        mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
        mediaRecorder.onstop = sendAudio;
        
        mediaRecorder.start();
        document.getElementById('record-button').disabled = true;
        document.getElementById('stop-button').disabled = false;
    } catch (error) {
        alert('Error accessing microphone: ' + error.message);
    }
});

document.getElementById('stop-button').addEventListener('click', () => {
    mediaRecorder.stop();
    document.getElementById('record-button').disabled = false;
    document.getElementById('stop-button').disabled = true;
});

function getScoreColor(score) {
    if (score >= 80) return '#4CAF50';
    if (score >= 60) return '#FF9800';
    return '#f44336';
}

async function sendAudio() {
    if (!selectedWord) {
        alert('Select a word first');
        return;
    }
    
    const analyzingStatus = document.getElementById('analyzing-status');
    analyzingStatus.style.display = 'block';
    analyzingStatus.textContent = '⏳ Analyzing pronunciation...';
    
    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
    const formData = new FormData();
    formData.append('audio', audioBlob);
    formData.append('accent', selectedAccent);
    formData.append('word', selectedWord);
    formData.append('model', selectedModel);
    
    try {
        const response = await fetch('/analyze', { method: 'POST', body: formData });
        const data = await response.json();
        
        analyzingStatus.style.display = 'none';
        
        const resultDiv = document.getElementById('result');
        const scoreColor = getScoreColor(data.score);
        
        resultDiv.innerHTML = `
            <div class="result-box">
                <h3>Results</h3>
                <p><strong>Detected:</strong></p>
                <p>eSpeak: <code>${data.transcription}</code></p>
                <p>IPA: <code>${data.detected_ipa}</code></p>
                <hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;">
                <p><strong>Expected:</strong></p>
                <p>eSpeak: <code>${data.expected_espeak}</code></p>
                <p>IPA: <code>${data.expected_ipa}</code></p>
                <hr style="margin: 15px 0; border: none; border-top: 1px solid #ddd;">
                <div class="score-display" style="background: ${scoreColor}; color: white; padding: 20px; border-radius: 8px; text-align: center;">
                    <p style="font-size: 36px; font-weight: bold; margin: 0;">${data.score}%</p>
                    <p style="margin: 5px 0; font-size: 14px;">${data.match ? '✓ Perfect!' : 'Keep practicing'}</p>
                </div>
            </div>
        `;
    } catch (error) {
        analyzingStatus.style.display = 'none';
        alert('Error analyzing audio: ' + error.message);
    }
}
