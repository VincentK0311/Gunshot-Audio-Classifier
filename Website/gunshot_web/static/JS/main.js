document.addEventListener('DOMContentLoaded', () => {
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('fileInput');
    const fileLabel = document.getElementById('fileLabel');
    const resetBtn = document.getElementById('resetBtn');
    const confirmBtn = document.querySelector('.confirm');
    const modelSelect = document.getElementById('model');
    const spinner = document.getElementById('spinner');
    const form = document.getElementById('uploadForm');

    // Click on drop area triggers file dialog
    dropArea.addEventListener('click', (e) => {
        if (e.target !== fileInput) {
            fileInput.click();
        }
    });

    // Highlight drop area on drag
    dropArea.addEventListener('dragover', e => {
        e.preventDefault();
        dropArea.classList.add('dragover');
    });

    dropArea.addEventListener('dragleave', () => {
        dropArea.classList.remove('dragover');
    });

    // File dropped into area
    dropArea.addEventListener('drop', e => {
        e.preventDefault();
        dropArea.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length !== 1 || !files[0].name.endsWith('.wav')) {
            alert('Please drop a single .wav file.');
            return;
        }

        const dt = new DataTransfer();
        dt.items.add(files[0]);
        fileInput.files = dt.files;

        checkAudioDuration(files[0]);
    });

    // File selected through browse
    fileInput.addEventListener('change', () => {
        const file = fileInput.files[0];
        if (!file || !file.name.endsWith('.wav')) {
            alert('Only .wav files are supported.');
            resetForm();
            return;
        }
        checkAudioDuration(file);
    });

    // Check audio duration (must be <= 2 seconds)
    function checkAudioDuration(file) {
        const audioEl = document.createElement('audio');
        audioEl.preload = 'metadata';
        audioEl.src = URL.createObjectURL(file);
        audioEl.onloadedmetadata = function() {
            URL.revokeObjectURL(audioEl.src);
            if (audioEl.duration > 2.5) { // allow up to 2.5 seconds for safety
                alert('Audio must be 2 seconds or shorter.');
                resetForm();
            } else {
                updateUIAfterFileSelect(file.name);
            }
        };
    }

    // Reset button
    resetBtn.addEventListener('click', resetForm);

    // Show loading spinner on submit
    form.addEventListener('submit', () => {
        spinner.style.display = 'flex';
    });

    // Check model selection before submit
    confirmBtn.addEventListener('click', e => {
        if (!fileInput.files.length) {
            e.preventDefault();
            alert('Please upload your audio file first.');
        } else if (!modelSelect.value) {
            e.preventDefault();
            alert('Please choose the model first.');
        }
    });

    // Enable form after valid file upload
    function updateUIAfterFileSelect(fileName) {
        fileLabel.textContent = fileName;
        modelSelect.disabled = false;
        confirmBtn.disabled = false;
        resetBtn.disabled = false;
    }

    // Reset all
    function resetForm() {
        form.reset();
        fileLabel.textContent = 'Click or Drop .wav file here';
        modelSelect.disabled = true;
        confirmBtn.disabled = true;
        resetBtn.disabled = true;
        spinner.style.display = 'none';
    }

    // Initial disable
    resetForm();
});
