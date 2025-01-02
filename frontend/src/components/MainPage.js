import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import '../styles/MainPage.css';
import axios from 'axios';
import Header from './Header';

function MainPage() {
    // Use useLocation to get the passed state
    const location = useLocation();
    const { prediction, full_audio, conversations } = location.state || { };
    // conversations = [conversations]
    const [texts, setTexts] = useState([]);
    const [newWord, setNewWord] = useState("");
    const [showInput, setShowInput] = useState({ index: null, pos: null });
    const [inputPosition, setInputPosition] = useState({ top: 0, left: 0 });
    const [insertIndex, setInsertIndex] = useState({
        textIndex: null,
        wordIndex: null,
    });
    const inputRef = useRef(null);
    const [audioUrl, setAudioUrl] = useState(null); // To store the Blob URL for the full_audio
    
    // Convert byte array to Blob URL for audio playback
    useEffect(() => {
        if (full_audio) {
            // Assuming full_audio is base64 encoded, we first decode it
            const byteString = atob(full_audio); // Decode base64
            const byteArray = new Uint8Array(byteString.length);
            for (let i = 0; i < byteString.length; i++) {
                byteArray[i] = byteString.charCodeAt(i);
            }
            // Assuming full_audio is a byte array, convert it to a Blob
            const audioBlob = new Blob([byteArray], { type: 'audio/wav' }); // or audio/wav, depending on the file type
            const audioUrl = URL.createObjectURL(audioBlob);
            setAudioUrl(audioUrl);
        }

        // Initialize conversation texts
        const initialTextData = conversations.map((conv, index) => {
            const key = Object.keys(conv)[0];
            const conversation = conv[key];
            const textArray = conversation.original.text.split(" ").map((word) => ({
            word,
            isStriked: false,
            isNew: false,
            }));
            return {
            conversationKey: key, // Store the conversation key (e.g., 'conv_0')
            speaker: conversation.speaker,
            textArray,
            modified: conversation.modified,
            };
        });
        setTexts(initialTextData);
    }, [full_audio, conversations]);
    

    // Handle click to toggle strike-through
    const handleWordClick = (textIndex, wordIndex) => {
    const updatedTexts = [...texts];
    updatedTexts[textIndex].textArray[wordIndex].isStriked =
      !updatedTexts[textIndex].textArray[wordIndex].isStriked;
    setTexts(updatedTexts);
    };


    // Handle new word input and insertion
    const handleNewWordInput = (e) => {
        setNewWord(e.target.value);
    };

    const handleInsertWord = (e) => {
        if (e.key === "Enter" && newWord.trim() !== "") {
          const updatedTexts = [...texts];
          const { textIndex, wordIndex } = insertIndex;
          updatedTexts[textIndex].textArray.splice(wordIndex + 1, 0, {
            word: newWord,
            isStriked: false,
            isNew: true,
          });
          setTexts(updatedTexts);
          setNewWord("");
          setShowInput({ index: null, pos: null }); // Hide input box after insertion
        }
    };

    // Show input box where the cursor is
    const handleShowInput = (event, textIndex, wordIndex) => {
        const rect = event.target.getBoundingClientRect();
        setInputPosition({
        top: rect.top + window.scrollY - 30, // Position the input ABOVE the clicked word
        left: rect.left + window.scrollX,
        });
        setInsertIndex({ textIndex, wordIndex });
        setShowInput({ index: textIndex, pos: wordIndex });
        setTimeout(() => {
        inputRef.current.focus();
        }, 0);
    };

    // Process and save the updated text for each conversation
    const handleSubmit = () => {
        const updatedConversations = [...texts].map((conversation) => {
        // Build the updated text by excluding strike-through words
        const updatedText = conversation.textArray
            .filter((wordObj) => !wordObj.isStriked)
            .map((wordObj) => wordObj.word)
            .join(" ");

        // Save the updated text in the 'modified' dictionary
        conversation.modified.text = updatedText;
        return conversation;
    });

    // Return the modified dictionary exactly as input format
    const conversationsUpdated = updatedConversations.reduce((acc, conv) => {
        const originalConversation = conversations.find((c) => c[conv.conversationKey]); // Access the original conversation
    
        acc[conv.conversationKey] = {
          original: originalConversation ? originalConversation[conv.conversationKey].original : {}, // Get the original
          modified: conv.modified,
          speaker: conv.speaker,
        };
        return acc;
      }, {});
  
        console.log("Final updated dictionary format:", conversationsUpdated);

        // Create a FormData object to hold conversationsUpdated and the full_audio file
        const formData = new FormData();

        // Append the conversationsUpdated as a JSON blob
        const conversationsUpdatedBlob = new Blob([JSON.stringify(conversationsUpdated)], { type: "application/json" });
        formData.append("conversationsUpdated", conversationsUpdatedBlob);

        // Assuming full_audio is a Blob or File (binary data)
        if (full_audio) {
            const audioBlob = base64ToBlob(full_audio, 'audio/wav'); // Convert base64 to Blob
            formData.append("full_audio", audioBlob, "full_audio.wav"); // Adjust the name and file type as needed
        }

        // Send FormData to Flask backend using axios
        axios.post('http://127.0.0.1:5000/conversations_modified', formData, {
        headers: {
            'Content-Type': 'multipart/form-data'
        }
        })
        .then((response) => {
        console.log('Data and audio sent successfully:', response.modified_audio);
        })
        .catch((error) => {
        console.error('There was an error sending the data and audio!', error);
        });

    };

    // Utility function to convert base64 string to Blob
    const base64ToBlob = (base64Data, contentType) => {
        const byteCharacters = atob(base64Data); // Decode base64 string
        const byteArrays = [];
    
        for (let offset = 0; offset < byteCharacters.length; offset += 512) {
        const slice = byteCharacters.slice(offset, offset + 512);
        const byteNumbers = new Array(slice.length);
        for (let i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }
        const byteArray = new Uint8Array(byteNumbers);
        byteArrays.push(byteArray);
        }
    
        return new Blob(byteArrays, { type: contentType });
    };

    return (
        <div className="main-page">
            <Header />
            <div className="speech-results">
                <h2>Audio Analysis Result</h2>
                <p>The uploaded audio is classified as: <strong>{prediction}</strong></p>

                {/* Render the audio player for the full audio */}
                {audioUrl && (
                    <div className="audio-player">
                        <audio controls>
                            <source src={audioUrl} type="audio/wav" />
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                )}

                <p>Here's the audio transcript:</p>

                {texts.map((conversation, textIndex) => (
                <div key={textIndex} className="text-block">
                <div className="centered-text">
                    <span className="speaker-label">
                    Speaker {conversation.speaker}:
                    </span>
                    {conversation.textArray.map((item, wordIndex) => (
                    <span key={wordIndex} style={{ position: "relative" }}>
                        <span
                        onClick={() => handleWordClick(textIndex, wordIndex)}
                        style={{
                            textDecoration: item.isStriked ? "line-through" : "none",
                            color: item.isStriked
                            ? "red"
                            : item.isNew
                            ? "green"
                            : "black",
                            cursor: "pointer",
                        }}
                        >
                        {item.word}
                        </span>{""}
                        <span
                        onClick={(e) => handleShowInput(e, textIndex, wordIndex)}
                        style={{
                            display: "inline-block",
                            width: "10px", // Minimal space to detect click
                            height: "20px",
                            cursor: "text",
                        }}
                        ></span>
                    </span>
                    ))}
                </div>
                </div>
            ))}
            
            {/* Floating input box */}
            {showInput.index !== null && (
                <input
                ref={inputRef}
                type="text"
                value={newWord}
                onChange={handleNewWordInput}
                onKeyDown={handleInsertWord}
                style={{
                    backgroundColor: "black",
                    color: "white",
                    fontSize: "18px",
                    padding: "10px 20px",
                    width: "200px",
                    borderRadius: "25px",
                    border: "2px solid #fff",
                    textAlign: "center",
                    caretColor: "white", // Cursor color as white
                    position: "absolute",
                    top: `${inputPosition.top}px`,
                    left: `${inputPosition.left}px`,
                    zIndex: 10,
                }}
                placeholder="Enter new word"
                />
            )}

            {/* Submit button */}
            <button
                onClick={handleSubmit}
                style={{ marginTop: "20px", padding: "10px 20px", fontSize: "16px" }}
            >
                Submit
            </button>
            </div>

        </div>
    );
}

export default MainPage;