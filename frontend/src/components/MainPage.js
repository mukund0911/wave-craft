import React, { useState, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import '../styles/MainPage.css';
import axios from 'axios';
import Header from './Header';

function MainPage() {
    // Use useLocation to get the passed state
    const location = useLocation();
    const { prediction, conversations } = location.state || { };
    // conversations = [conversations]

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

    const [texts, setTexts] = useState(initialTextData);
    const [newWord, setNewWord] = useState("");
    const [showInput, setShowInput] = useState({ index: null, pos: null });
    const [inputPosition, setInputPosition] = useState({ top: 0, left: 0 });
    const [insertIndex, setInsertIndex] = useState({
        textIndex: null,
        wordIndex: null,
    });
    const inputRef = useRef(null);

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
    const finalStructure = updatedConversations.reduce((acc, conv) => {
        acc[conv.conversationKey] = {
          original: conversations.find((c) => c[conv.conversationKey]).original,
          modified: conv.modified,
          speaker: conv.speaker,
        };
        return acc;
      }, {});
  
        console.log("Final updated dictionary format:", finalStructure);

        // Send finalStructure to Flask backend using axios
        axios.post('http://127.0.0.1:5000/conversations_modified', finalStructure)
        .then((response) => {
            console.log('Data sent successfully:', response.data);
        })
        .catch((error) => {
            console.error('There was an error sending the data!', error);
        });

    };

    return (
        <div className="main-page">
            <Header />
            <div className="speech-results">
                <h2>Audio Analysis Result</h2>
                <p>The uploaded audio is classified as: <strong>{prediction}</strong></p>
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