// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    const chatDisplay = document.getElementById('chat-display');
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const listenButton = document.getElementById('listen-button');
    const stopButton = document.getElementById('stop-button');

    function addMessage(message, isUser = false) {
        const messageElement = document.createElement('div');
        messageElement.classList.add('message');
        messageElement.classList.add(isUser ? 'user-message' : 'assistant-message');
        messageElement.textContent = message;
        chatDisplay.appendChild(messageElement);
        chatDisplay.scrollTop = chatDisplay.scrollHeight;
    }

    async function sendMessage() {
        const message = userInput.value.trim();
        if (message) {
            addMessage(message, true);
            userInput.value = '';

            try {
                const response = await fetch('/process_command', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ command: message }),
                });
                const data = await response.json();
                addMessage(data.response);
            } catch (error) {
                console.error('Error:', error);
                addMessage('Sorry, an error occurred. Please try again.');
            }
        }
    }

    sendButton.addEventListener('click', sendMessage);
    userInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    listenButton.addEventListener('click', () => {
        addMessage('Listening...');
        // Implement speech recognition here
    });

    stopButton.addEventListener('click', () => {
        addMessage('Stopped listening.');
        // Implement stop listening functionality here
    });
});