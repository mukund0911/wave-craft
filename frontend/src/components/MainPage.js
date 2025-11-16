import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import '../styles/MainPage.css';
import axios from 'axios';
import Header from './Header';
import ArtificialSpeakerModal from './ArtificialSpeakerModal';

function MainPage() {
    const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:5000';

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
    
    // Modal states
    const [showArtificialSpeakerModal, setShowArtificialSpeakerModal] = useState(false);
    const [insertAfterIndex, setInsertAfterIndex] = useState(null);
    
    // Final audio preview states
    const [finalAudioUrl, setFinalAudioUrl] = useState(null);
    const [finalAudioData, setFinalAudioData] = useState(null);
    const [isProcessing, setIsProcessing] = useState(false);
    const [showFinalAudioPreview, setShowFinalAudioPreview] = useState(false);
    
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
    const handleSubmit = async () => {
        setIsProcessing(true);
        setShowFinalAudioPreview(false);
        
        try {
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

            // Prepare conversations data for the API - Include artificial speakers
            const conversationsForAPI = updatedConversations.map((conv, index) => {
                const conversationKey = conv.conversationKey || `conv_${index}`;
                
                // For artificial speakers, include their generated audio
                const conversationData = {
                    speaker: conv.speaker,
                    original: {
                        text: conv.textArray.map(w => w.word).join(" "),
                        start: 0,
                        end: 0
                    },
                    modified: {
                        text: conv.modified.text
                    }
                };

                // Include AI-generated audio if available
                if (conv.artificial && conv.audio_base64) {
                    conversationData.original.speaker_audio = conv.audio_base64;
                    conversationData.artificial = true;
                    conversationData.speaker_characteristics = conv.speaker_characteristics;
                } else if (conversations[index] && conversations[index][Object.keys(conversations[index])[0]]) {
                    // Use original transcribed audio
                    const originalConv = conversations[index][Object.keys(conversations[index])[0]];
                    conversationData.original = {
                        ...conversationData.original,
                        ...originalConv.original
                    };
                }

                return { [conversationKey]: conversationData };
            });

            // Create FormData for the request
            const formData = new FormData();
            const conversationsBlob = new Blob([JSON.stringify(conversationsForAPI)], { type: "application/json" });
            formData.append("conversationsUpdated", conversationsBlob);

            // Add full audio if available
            if (full_audio) {
                const audioBlob = base64ToBlob(full_audio, 'audio/wav');
                formData.append("full_audio", audioBlob, "full_audio.wav");
            }

            // Send request to backend
            const response = await axios.post(apiUrl + '/conversations_modified', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data'
                }
            });

            if (response.data && response.data.modified_audio) {
                // Create audio blob and URL for preview
                const finalAudioBlob = base64ToBlob(response.data.modified_audio, 'audio/wav');
                const finalAudioUrl = URL.createObjectURL(finalAudioBlob);
                
                setFinalAudioData({
                    audioBase64: response.data.modified_audio,
                    audioBlob: finalAudioBlob,
                    duration: response.data.duration_seconds,
                    sampleRate: response.data.sample_rate,
                    segmentsCloned: response.data.segments_cloned,
                    segmentsProcessed: response.data.segments_processed,
                    voiceCloningEnabled: response.data.voice_cloning_enabled
                });
                setFinalAudioUrl(finalAudioUrl);
                setShowFinalAudioPreview(true);
                
                console.log('Final audio generated successfully:', response.data);
            }
        } catch (error) {
            console.error('Error generating final audio:', error);
            alert('Failed to generate final audio. Please try again.');
        } finally {
            setIsProcessing(false);
        }
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

    // Download final audio function
    const handleDownloadFinalAudio = () => {
        if (finalAudioData && finalAudioData.audioBlob) {
            const downloadUrl = URL.createObjectURL(finalAudioData.audioBlob);
            const link = document.createElement('a');
            link.href = downloadUrl;
            link.download = `final_audio_${new Date().getTime()}.wav`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(downloadUrl);
        }
    };

    // Format duration for display
    const formatDuration = (seconds) => {
        if (!seconds) return '0:00';
        const minutes = Math.floor(seconds / 60);
        const remainingSeconds = Math.floor(seconds % 60);
        return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
    };

    // Handler for adding conversations
    const handleAddConversation = (index) => {
        const updatedTexts = [...texts];
        const newConversation = {
            conversationKey: `conv_${texts.length}`,
            speaker: "New",
            textArray: [{
                word: "New conversation",
                isStriked: false,
                isNew: true
            }],
            modified: { text: "New conversation" }
        };
        updatedTexts.splice(index + 1, 0, newConversation);
        setTexts(updatedTexts);
    };

    // Handler for adding artificial speaker
    const handleAddArtificialSpeaker = (index) => {
        setInsertAfterIndex(index);
        setShowArtificialSpeakerModal(true);
    };

    // Submit artificial speaker request
    const handleArtificialSpeakerSubmit = async (requestData) => {
        try {
            const response = await axios.post(`${apiUrl}/add_artificial_speaker`, requestData, {
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.data && response.data.conversation_entry) {
                const newConversation = response.data.conversation_entry;
                
                // Convert conversation entry to the format expected by the UI
                const formattedConversation = {
                    conversationKey: `conv_ai_${Date.now()}`,
                    speaker: newConversation.speaker,
                    textArray: [{
                        word: newConversation.original.text,
                        isStriked: false,
                        isNew: true
                    }],
                    modified: newConversation.modified,
                    artificial: true,
                    speaker_characteristics: newConversation.speaker_characteristics,
                    audio_base64: newConversation.original.speaker_audio
                };

                // Insert the new conversation after the specified index
                const updatedTexts = [...texts];
                const insertIndex = insertAfterIndex !== null ? insertAfterIndex + 1 : texts.length;
                updatedTexts.splice(insertIndex, 0, formattedConversation);
                setTexts(updatedTexts);

                console.log('Artificial speaker created:', response.data);
            }
        } catch (error) {
            console.error('Error creating artificial speaker:', error);
            throw error;
        }
    };

     return (
        <div className="main-page">
            <Header />
            <div className="speech-results">
                <h2>Audio Analysis Result</h2>
                <p>The uploaded audio is classified as: <strong>{prediction}</strong></p>

                {audioUrl && (
                    <div className="audio-player">
                        <audio controls>
                            <source src={audioUrl} type="audio/wav" />
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                )}

                <p>Here's the audio transcript:</p>

                <div className="conversation-container">
                    {texts.map((conversation, textIndex) => (
                        <React.Fragment key={textIndex}>
                            <div className="conversation-block">
                                <div className="text-block">
                                    <div className="centered-text">
                                        <span className={`speaker-label ${conversation.artificial ? 'artificial-speaker' : ''}`}>
                                            Speaker {conversation.speaker}:
                                            {conversation.artificial && <span className="ai-badge">AI</span>}
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
                                                </span>{" "}
                                                <span
                                                    onClick={(e) => handleShowInput(e, textIndex, wordIndex)}
                                                    style={{
                                                        display: "inline-block",
                                                        width: "10px",
                                                        height: "20px",
                                                        cursor: "text",
                                                    }}
                                                ></span>
                                            </span>
                                        ))}
                                    </div>
                                </div>
                                <div className="conversation-actions">
                                    {conversation.audio_base64 && (
                                        <audio 
                                            controls 
                                            className="conversation-audio"
                                            src={`data:audio/wav;base64,${conversation.audio_base64}`}
                                        />
                                    )}
                                </div>
                            </div>
                            <div className="conversation-controls">
                                <button
                                    className="add-conversation-button regular"
                                    onClick={() => handleAddConversation(textIndex)}
                                    title="Add new conversation"
                                >
                                    + Add Conversation
                                </button>
                                <button
                                    className="add-conversation-button ai"
                                    onClick={() => handleAddArtificialSpeaker(textIndex)}
                                    title="Add AI speaker"
                                >
                                    🤖 Add AI Speaker
                                </button>
                            </div>
                        </React.Fragment>
                    ))}
                </div>

                {/* Floating input box */}
                {showInput.index !== null && (
                    <input
                        ref={inputRef}
                        type="text"
                        value={newWord}
                        onChange={handleNewWordInput}
                        onKeyDown={handleInsertWord}
                        style={{
                            position: "absolute",
                            top: `${inputPosition.top}px`,
                            left: `${inputPosition.left}px`,
                            zIndex: 10,
                        }}
                        placeholder="Enter new word"
                    />
                )}

                <button
                    onClick={handleSubmit}
                    disabled={isProcessing}
                    style={{
                        marginTop: "20px",
                        padding: "10px 20px",
                        fontSize: "16px",
                        backgroundColor: isProcessing ? "#666" : "#000",
                        color: "#fff",
                        border: "none",
                        borderRadius: "5px",
                        cursor: isProcessing ? "not-allowed" : "pointer"
                    }}
                >
                    {isProcessing ? "Processing..." : "Submit"}
                </button>

                {/* Final Audio Preview */}
                {showFinalAudioPreview && finalAudioUrl && (
                    <div className="final-audio-preview" style={{
                        marginTop: "30px",
                        padding: "20px",
                        border: "2px solid #4CAF50",
                        borderRadius: "10px",
                        backgroundColor: "#f9f9f9"
                    }}>
                        <h3 style={{ color: "#4CAF50", marginBottom: "15px" }}>
                            🎉 Final Audio Ready!
                        </h3>
                        
                        <div className="audio-info" style={{ marginBottom: "15px" }}>
                            <p><strong>Duration:</strong> {formatDuration(finalAudioData?.duration)}</p>
                            <p><strong>Quality:</strong> {finalAudioData?.sampleRate || 22050} Hz</p>
                            {finalAudioData?.segmentsProcessed && (
                                <p><strong>Segments:</strong> {finalAudioData.segmentsProcessed} processed</p>
                            )}
                            {finalAudioData?.voiceCloningEnabled && finalAudioData?.segmentsCloned > 0 && (
                                <p><strong>Voice Cloning:</strong> ✅ {finalAudioData.segmentsCloned} segment(s) regenerated</p>
                            )}
                        </div>

                        <div className="audio-controls" style={{ 
                            display: "flex", 
                            alignItems: "center", 
                            gap: "15px",
                            flexWrap: "wrap"
                        }}>
                            <audio 
                                controls 
                                style={{ flex: "1", minWidth: "300px" }}
                                src={finalAudioUrl}
                            >
                                Your browser does not support the audio element.
                            </audio>
                            
                            <button
                                onClick={handleDownloadFinalAudio}
                                style={{
                                    padding: "10px 20px",
                                    fontSize: "14px",
                                    backgroundColor: "#4CAF50",
                                    color: "#fff",
                                    border: "none",
                                    borderRadius: "5px",
                                    cursor: "pointer",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "8px"
                                }}
                            >
                                📥 Download Audio
                            </button>
                        </div>

                        <div className="processing-info" style={{
                            marginTop: "15px",
                            fontSize: "12px",
                            color: "#666",
                            fontStyle: "italic"
                        }}>
                            <p>✅ Audio processed with style preservation and quality enhancement</p>
                        </div>
                    </div>
                )}
            </div>

            <ArtificialSpeakerModal
                isOpen={showArtificialSpeakerModal}
                onClose={() => setShowArtificialSpeakerModal(false)}
                onSubmit={handleArtificialSpeakerSubmit}
                conversationHistory={conversations}
            />
        </div>
    );
}

export default MainPage;