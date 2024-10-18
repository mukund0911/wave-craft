import React, { useState, useRef, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import '../styles/MainPage.css';
import Header from './Header';

function MainPage() {
    // Use useLocation to get the passed state
    const location = useLocation();
    const { prediction, converted_text } = location.state || { };

    // Set the initial state with the passed text
    const [editedText, setEditedText] = useState(converted_text);
    const [previousText, setPreviousText] = useState(editedText);

    const contentEditableRef = useRef(null);

     // Function to handle user edits in the textarea
     const handleTextChange = (e) => {
        setPreviousText(converted_text);
        setEditedText(e.target.innerText); // Update the state with the new text
    };

    // Function to handle saving the text (submit back to server or local storage)
    const handleSave = () => {
        console.log("Saving modified text:", editedText);
        // You can send this modified text back to the server using fetch or axios
        // For example, sending it back to your Flask backend:
        // axios.post('http://localhost:5000/save_text', { modified_text: editedText });
    };

    // Detect added or deleted characters
    const detectChanges = () => {
        const currentText = editedText.split('');
        const previous = previousText.split('');

        let result = "";
        let i = 0;

        // Compare current and previous text character by character
        while (i < currentText.length || i < previous.length) {
            if (i >= currentText.length) {
                // If character exists in previous text but not in current
                result += `<span class="deleted">${previous[i]}</span>`;
            } else if (i >= previous.length) {
                // If character exists in current text but not in previous
                result += `<span class="added">${currentText[i]}</span>`;
            } else if (currentText[i] !== previous[i]) {
                // Detect differences (either addition or replacement)
                result += `<span class="added">${currentText[i]}</span>`;
                result += `<span class="deleted">${previous[i]}</span>`;
            } else {
                result += currentText[i];
            }
            i++;
        }
        return result;
    };

    useEffect(() => {
        // Update the contentEditable div with the result of detectChanges
        const contentEditable = contentEditableRef.current;
        contentEditable.innerHTML = detectChanges();
    }, [editedText]);  // Re-run when the text changes

    

    return (
        <div className="main-page">
            <Header />
            <h2>Audio Analysis Result</h2>
            <p>The uploaded audio is classified as: <strong>{prediction}</strong></p>
            <p>Here's the audio transcript:</p>

            <div
                ref={contentEditableRef}
                contentEditable
                onInput={handleTextChange}
                className="editable"
                style={{ border: '1px solid #ccc', padding: '10px', fontSize: '16px', minHeight: '150px', width: '80%', margin: '0 auto' }}
            ></div>

            {/* <textarea
                value={editedText}
                onChange={handleTextChange}
                rows="10"
                cols="80"
                style={{ fontSize: '16px', padding: '10px' }}
            /> */}

            {/* Save Button */}
            <button onClick={handleSave} style={{ marginTop: '10px' }}>
                Save Changes
            </button>
        </div>
    );
}

export default MainPage;
