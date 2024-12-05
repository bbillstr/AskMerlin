// Show/hide processing overlay
function showProcessing(message = 'Processing...') {
    const overlay = document.getElementById('processingOverlay');
    const status = document.getElementById('processingStatus');
    if (overlay && status) {
        status.textContent = message;
        overlay.style.display = 'flex';
    }
}

function hideProcessing() {
    const overlay = document.getElementById('processingOverlay');
    if (overlay) {
        overlay.style.display = 'none';
    }
}

// Toast notifications
function showToast(message, isError = false) {
    const toast = document.getElementById('toast');
    if (toast) {
        toast.textContent = message;
        toast.style.backgroundColor = isError ? 'var(--error-color)' : 'var(--success-color)';
        toast.style.opacity = '1';
        toast.style.transform = 'translateY(0)';
        
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateY(20px)';
        }, 3000);
    }
}

// Document Management
function loadDocuments() {
    const grid = document.querySelector('.documents-grid');
    if (!grid) return;

    fetch('/api/documents')
        .then(response => response.json())
        .then(documents => {
            console.log('Loaded documents:', documents);
            
            if (!documents || documents.length === 0) {
                grid.innerHTML = `
                    <div class="empty-state">
                        <i class="fas fa-folder-open"></i>
                        <p>No documents uploaded yet</p>
                    </div>
                `;
                return;
            }

            grid.innerHTML = '';
            documents.forEach(doc => {
                const card = document.createElement('div');
                card.className = 'document-card';
                card.innerHTML = `
                    <div class="document-info">
                        <h3 class="document-title">${doc.title || 'Untitled'}</h3>
                        <p class="document-preview">${doc.preview || ''}</p>
                        <div class="document-meta">
                            <span class="document-date">
                                <i class="fas fa-calendar"></i>
                                ${new Date(doc.uploaded_at).toLocaleDateString()}
                            </span>
                        </div>
                    </div>
                    <div class="document-actions">
                        <button class="icon-btn delete-btn" onclick="deleteDocument(${doc.id})" title="Delete Document">
                            <i class="fas fa-trash"></i>
                        </button>
                    </div>
                `;
                grid.appendChild(card);
            });
        })
        .catch(error => {
            console.error('Error loading documents:', error);
            showToast('Failed to load documents', true);
        });
}

function deleteDocument(id) {
    if (!confirm('Are you sure you want to delete this document?')) return;

    fetch(`/documents/${id}`, { method: 'DELETE' })
        .then(response => response.json())
        .then(data => {
            if (data.error) throw new Error(data.error);
            showToast('Document deleted successfully');
            loadDocuments();
        })
        .catch(error => {
            console.error('Error deleting document:', error);
            showToast('Failed to delete document', true);
        });
}

function showUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) modal.style.display = 'flex';
}

function closeUploadModal() {
    const modal = document.getElementById('uploadModal');
    if (modal) modal.style.display = 'none';
    // Reset form
    document.getElementById('documentFile').value = '';
    document.getElementById('documentText').value = '';
    document.getElementById('documentTitle').value = '';
    const fileInfo = document.querySelector('.file-info');
    const dropZoneContent = document.querySelector('.drop-zone-content');
    if (fileInfo) fileInfo.style.display = 'none';
    if (dropZoneContent) dropZoneContent.style.display = 'flex';
}

function uploadDocument() {
    const formData = new FormData();
    const fileInput = document.getElementById('documentFile');
    const textInput = document.getElementById('documentText');
    const titleInput = document.getElementById('documentTitle');

    if (fileInput.files[0]) {
        formData.append('document', fileInput.files[0]);
    } else if (textInput.value.trim()) {
        formData.append('content', textInput.value.trim());
    } else {
        showToast('Please provide a document or paste text', true);
        return;
    }

    if (titleInput.value.trim()) {
        formData.append('title', titleInput.value.trim());
    }

    showProcessing('Uploading document...');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        showToast('Document uploaded successfully');
        closeUploadModal();
        loadDocuments();
    })
    .catch(error => {
        showToast(error.message || 'Upload failed', true);
    })
    .finally(() => {
        hideProcessing();
    });
}

// Chat functionality
function addMessage(content, isQuestion = true) {
    const chatHistory = document.getElementById('chatHistory');
    if (!chatHistory) return;

    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${isQuestion ? 'question' : 'answer'}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (isQuestion) {
        contentDiv.textContent = content;
    } else {
        contentDiv.innerHTML = marked.parse(content);
        // Initialize code highlighting
        contentDiv.querySelectorAll('pre code').forEach((block) => {
            hljs.highlightBlock(block);
        });
    }
    
    messageDiv.appendChild(contentDiv);
    chatHistory.appendChild(messageDiv);
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Load documents if on documents page
    if (document.querySelector('.documents-grid')) {
        loadDocuments();
    }

    // Initialize chat if on chat page
    const questionInput = document.getElementById('questionInput');
    const askBtn = document.getElementById('askBtn');
    
    if (questionInput && askBtn) {
        // Auto-resize textarea
        questionInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 150) + 'px';
        });

        // Handle ask button click
        askBtn.addEventListener('click', async function() {
            const question = questionInput.value.trim();
            if (!question) return;

            // Show loading state
            const loadingIndicator = document.querySelector('.loading-indicator');
            loadingIndicator.style.display = 'block';
            askBtn.disabled = true;
            questionInput.disabled = true;

            // Add question to chat
            addMessage(question, true);
            questionInput.value = '';
            questionInput.style.height = 'auto';

            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ question })
                });

                const data = await response.json();
                if (response.ok) {
                    addMessage(data.answer, false);
                } else {
                    showToast(data.error || 'Failed to get answer', true);
                }
            } catch (error) {
                console.error('Error:', error);
                showToast('Error getting answer', true);
            } finally {
                loadingIndicator.style.display = 'none';
                askBtn.disabled = false;
                questionInput.disabled = false;
                questionInput.focus();
            }
        });

        // Handle enter key
        questionInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                askBtn.click();
            }
        });
    }

    // Initialize questionnaire file input
    const batchQuestionFile = document.getElementById('batchQuestionFile');
    if (batchQuestionFile) {
        batchQuestionFile.addEventListener('change', handleQuestionnaireFile);
    }

    // Initialize document file input
    const documentFile = document.getElementById('documentFile');
    if (documentFile) {
        documentFile.addEventListener('change', function(e) {
            if (this.files[0]) {
                handleFile(this.files[0]);
            }
        });
    }
});

// Add/Update these modal-related functions

function showQuestionnaireUpload() {
    const modal = document.getElementById('questionnaireModal');
    if (modal) {
        modal.style.display = 'flex';
        // Reset form
        document.getElementById('batchQuestionFile').value = '';
        document.getElementById('batchFileName').textContent = 'No file chosen';
        document.getElementById('columnSelectionArea').style.display = 'none';
    }
}

function closeQuestionnaireModal() {
    const modal = document.getElementById('questionnaireModal');
    if (modal) {
        modal.style.display = 'none';
    }
}

function handleQuestionnaireFile() {
    const fileInput = document.getElementById('batchQuestionFile');
    const fileNameSpan = document.getElementById('batchFileName');
    const columnSelectionArea = document.getElementById('columnSelectionArea');
    const processBtn = document.getElementById('processBtn');
    
    if (fileInput.files[0]) {
        const file = fileInput.files[0];
        fileNameSpan.textContent = file.name;
        processBtn.disabled = false;
        
        // Show file type specific UI
        if (file.name.endsWith('.xlsx') || file.name.endsWith('.xls')) {
            showProcessing('Reading Excel file...');
            const formData = new FormData();
            formData.append('document', file);
            
            fetch('/get-columns', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.columns) {
                    populateColumnSelects(data.columns);
                    columnSelectionArea.style.display = 'block';
                    columnSelectionArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                }
            })
            .catch(error => {
                console.error('Error getting columns:', error);
                showToast('Error reading Excel file', true);
            })
            .finally(() => {
                hideProcessing();
            });
        } else {
            columnSelectionArea.style.display = 'none';
        }
    }
}

function populateColumnSelects(columns) {
    const questionSelect = document.getElementById('questionColumn');
    const answerSelect = document.getElementById('answerColumn');
    
    if (questionSelect && answerSelect) {
        questionSelect.innerHTML = '';
        answerSelect.innerHTML = '';
        
        columns.forEach(column => {
            const qOption = document.createElement('option');
            const aOption = document.createElement('option');
            qOption.value = column;
            aOption.value = column;
            qOption.textContent = column;
            aOption.textContent = column;
            questionSelect.appendChild(qOption);
            answerSelect.appendChild(aOption.cloneNode(true));
        });
    }
}

function processQuestionnaire() {
    const fileInput = document.getElementById('batchQuestionFile');
    if (!fileInput.files[0]) {
        showToast('Please select a file', true);
        return;
    }

    const formData = new FormData();
    formData.append('document', fileInput.files[0]);

    // Add column information for Excel files
    if (fileInput.files[0].name.match(/\.xlsx?$/i)) {
        const questionCol = document.getElementById('questionColumn').value;
        const answerCol = document.getElementById('answerColumn').value;
        
        if (questionCol === answerCol) {
            showToast('Please select different columns for questions and answers', true);
            return;
        }
        
        formData.append('questionColumn', questionCol);
        formData.append('answerColumn', answerCol);
    }

    // Close modal and show processing
    closeQuestionnaireModal();
    showProcessing('Processing questions...');

    fetch('/batch-questions', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) throw new Error(data.error);
        
        if (data.results) {
            // First, quickly update the processing status
            updateProcessingStatus(`Processing ${data.results.length} questions...`);
            
            // Batch process all questions at once
            const processAllMessages = () => {
                // Add all questions and answers rapidly
                data.results.forEach(result => {
                    addMessage(result.question, true);
                    addMessage(result.answer, false);
                });

                // Scroll to the latest message
                const chatHistory = document.getElementById('chatHistory');
                if (chatHistory) {
                    chatHistory.scrollTop = chatHistory.scrollHeight;
                }
            };

            // Process everything with a minimal delay
            setTimeout(() => {
                processAllMessages();
                
                // Handle document download if available
                if (data.downloadUrl) {
                    updateProcessingStatus('Preparing document...');
                    setTimeout(() => {
                        const downloadLink = document.createElement('a');
                        downloadLink.href = data.downloadUrl;
                        downloadLink.download = '';
                        document.body.appendChild(downloadLink);
                        downloadLink.click();
                        document.body.removeChild(downloadLink);
                        
                        showToast('Processing complete - Document ready');
                        hideProcessing();
                    }, 500);
                } else {
                    showToast('Processing complete');
                    hideProcessing();
                }
            }, 300);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showToast(error.message || 'Error processing questions', true);
        hideProcessing();
    });
}

// Smoother status updates
function updateProcessingStatus(message) {
    const status = document.getElementById('processingStatus');
    if (status) {
        status.textContent = message;
    }
} 