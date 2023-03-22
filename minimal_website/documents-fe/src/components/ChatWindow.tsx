import {useState} from 'react';
import './ChatWindow.css';

const API_URL_BE: string = 'http://localhost:8000/api/chat/';

interface IMessage {
    message: string;
    // role true for user, false for bot
    is_user: boolean;
}

function ChatWindow(): JSX.Element {
    const [messages] = useState<IMessage[]>([]);
    const [inputValue, setInputValue] = useState<string>('');

    const handleMessageSend = async (): Promise<void> => {
        if (inputValue.trim() !== '') {
            messages.push({message: inputValue, is_user: true});
            // API call to the chatbot
            const response = await fetch(API_URL_BE, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': 'Bearer API_KEY_HERE'
                },
                body: JSON.stringify({messages: messages})
            });
            const data = await response.json();
            messages.push({message: data.message, is_user: false});
            setInputValue('');
        }
    };

    return (
        <div className="chat-window">
            <div className="message-container">
                {messages.map((message, index) => (
                    <div className={message.is_user ? "user-message" : "bot-message"} key={index}>
                        {message.message}
                    </div>
                ))}
            </div>
            <div className="input-container">
                <input
                    type="text"
                    className="message-input"
                    placeholder="Enter your message..."
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                />
                <button className="send-button" onClick={handleMessageSend}>
                    Send
                </button>
            </div>
        </div>
    );
}

export default ChatWindow;
