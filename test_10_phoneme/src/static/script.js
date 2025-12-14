let mediaRecorder;
let audioChunks = [];
let selectedAccent = 'American';
let selectedModel = '';
let selectedUserMode = 'Native';

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

// User mode selection
document.querySelectorAll('.user-mode-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        document.querySelectorAll('.user-mode-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        selectedUserMode = e.target.dataset.mode;
        
        // Reset pattern selection and reload patterns
        document.getElementById('pattern-select').value = '';
        document.getElementById('pattern-select').dispatchEvent(new Event('change'));
        loadPatterns();
    });
});

// Accent selection (using event delegation)
document.getElementById('accent-section').addEventListener('click', (e) => {
    if (e.target.classList.contains('accent-btn')) {
        document.querySelectorAll('.accent-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        selectedAccent = e.target.dataset.accent;
        
        // Reload patterns when accent changes
        const currentPattern = document.getElementById('pattern-select').value;
        if (currentPattern) {
            loadPatterns();
            // Restore pattern selection after reload
            setTimeout(() => {
                document.getElementById('pattern-select').value = currentPattern;
                document.getElementById('pattern-select').dispatchEvent(new Event('change'));
            }, 100);
        }
    }
});

document.getElementById('model-select').addEventListener('change', (e) => {
    selectedModel = e.target.value;
});

let selectedWord = '';

// Load patterns based on user mode and accent
async function loadPatterns() {
    try {
        const response = await fetch(`/patterns?user_mode=${selectedUserMode}&accent=${selectedAccent}`);
        const patterns = await response.json();
        const select = document.getElementById('pattern-select');
        const currentValue = select.value;
        
        select.innerHTML = '<option value="">Choose a pattern...</option>';
        patterns.forEach(pattern => {
            const opt = document.createElement('option');
            opt.value = pattern.id;
            opt.textContent = `Pattern ${pattern.id}: ${pattern.name} - ${pattern.description}`;
            select.appendChild(opt);
        });
        
        // Restore selection if it still exists
        if (currentValue && patterns.some(p => p.id == currentValue)) {
            select.value = currentValue;
        }
    } catch (error) {
        console.error('Error loading patterns:', error);
    }
}

loadPatterns();

// Handle pattern selection
document.getElementById('pattern-select').addEventListener('change', async (e) => {
    const patternId = e.target.value;
    const exampleHeading = document.getElementById('example-heading');
    const exampleButtons = document.getElementById('example-buttons');
    const accentHeading = document.getElementById('accent-heading');
    const accentSection = document.getElementById('accent-section');
    const listenSection = document.getElementById('listen-section');
    
    if (patternId) {
        // Load pattern words
        try {
            const response = await fetch(`/pattern/${patternId}/words?user_mode=${selectedUserMode}&accent=${selectedAccent}`);
            if (response.ok) {
                const data = await response.json();
                
                // Show example heading and buttons
                exampleHeading.style.display = 'block';
                exampleButtons.style.display = 'block';
                
                // Clear and populate example buttons
                exampleButtons.innerHTML = '';
                data.words.forEach(word => {
                    const btn = document.createElement('button');
                    btn.className = 'example-btn';
                    btn.textContent = word;
                    btn.dataset.word = word;
                    btn.addEventListener('click', () => {
                        // Remove selected class from all buttons
                        document.querySelectorAll('.example-btn').forEach(b => b.classList.remove('selected'));
                        // Add selected class to clicked button
                        btn.classList.add('selected');
                        selectedWord = word;
                        
                        // Show accent selection and listen button
                        accentHeading.style.display = 'block';
                        accentSection.style.display = 'block';
                        listenSection.style.display = 'block';
                    });
                    exampleButtons.appendChild(btn);
                });
                
                // Hide accent and listen sections until word is selected
                accentHeading.style.display = 'none';
                accentSection.style.display = 'none';
                listenSection.style.display = 'none';
                selectedWord = '';
            }
        } catch (error) {
            console.error('Error loading pattern words:', error);
        }
    } else {
        exampleHeading.style.display = 'none';
        exampleButtons.style.display = 'none';
        accentHeading.style.display = 'none';
        accentSection.style.display = 'none';
        listenSection.style.display = 'none';
        selectedWord = '';
    }
});

// Listen button functionality
document.getElementById('listen-button').addEventListener('click', async () => {
    if (!selectedWord) {
        alert('Select an example first');
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
                text: selectedWord,
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
    if (score >= 80) return '#4CAF50';  // green
    if (score >= 60) return '#FF9800';  // orange
    return '#f44336';                   // red
}

async function sendAudio() {
    if (!selectedWord) {
        alert('Select an example first');
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
    } catch (error) {
        analyzingStatus.style.display = 'none';
        alert('Error analyzing audio: ' + error.message);
    }
}
