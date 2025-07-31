# Wave-Craft MCP Extension Setup Guide

This guide will help you set up the new MCP (Model Context Protocol) based features for artificial speaker generation and background music.

## New Features Added

1. **Artificial Speaker Generation**: Add AI-generated speakers with customizable characteristics
2. **Background Music Integration**: Add background music to conversations or entire audio
3. **Modular MCP Architecture**: Individual agents for different AI capabilities

## Prerequisites

1. Python 3.8+
2. Node.js 14+
3. OpenAI API Key with sufficient credits

## Setup Instructions

### 1. Backend Setup

#### Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

#### Configure OpenAI API Key
Edit `backend/models/config.py`:
```python
OPENAI_API_KEY = "your-actual-openai-api-key-here"
```

**Important**: Ensure your OpenAI account has sufficient credits. The system uses:
- GPT-3.5-turbo for dialogue generation
- OpenAI TTS API for speech synthesis

#### Directory Structure
The new MCP agents are located in:
```
backend/
├── mcp_agents/
│   ├── __init__.py
│   ├── base_agent.py              # Base MCP agent class
│   ├── dialogue_generator_agent.py # OpenAI dialogue generation
│   ├── tts_agent.py              # OpenAI text-to-speech
│   ├── music_agent.py            # Background music processing
│   └── agent_coordinator.py      # Orchestrates all agents
├── music_library/                # Directory for background music files
└── routes.py                     # Updated with new endpoints
```

### 2. Frontend Setup

#### Install Dependencies
```bash
cd frontend
npm install
```

#### New Components Added
- `ArtificialSpeakerModal.js` - Modal for creating AI speakers
- `BackgroundMusicModal.js` - Modal for adding background music
- Updated `MainPage.js` with new features

### 3. Running the Application

#### Start Backend
```bash
cd backend
python run.py
```

#### Start Frontend
```bash
cd frontend
npm start
```

## API Endpoints

### New Endpoints Added

#### 1. Add Artificial Speaker
```
POST /add_artificial_speaker
Content-Type: application/json

{
  "conversation_history": [...],
  "speaker_prompt": "A confident male speaker discussing technology",
  "add_background_music": true,
  "music_type": "corporate",
  "music_volume": 0.3
}
```

#### 2. Add Background Music
```
POST /add_background_music
Content-Type: application/json

{
  "audio_base64": "base64_encoded_audio",
  "music_type": "calm",
  "volume_level": 0.3,
  "target_speaker": "A"
}
```

#### 3. Get Agent Capabilities
```
GET /agent_capabilities
```

## Using the Features

### Adding Artificial Speakers

1. After transcribing audio, click "🤖 Add AI Speaker" between conversations
2. In the modal, describe the speaker characteristics:
   - Gender (male/female)
   - Personality (calm, energetic, professional)
   - Emotion (happy, serious, excited)
   - Topic expertise
3. Optionally enable background music
4. The AI will generate appropriate dialogue and convert it to speech

### Adding Background Music

1. Click the music button (🎵) next to any conversation or on the main audio player
2. Choose music type:
   - **Calm & Ambient**: Relaxing background tones
   - **Upbeat & Energetic**: Positive, motivating music
   - **Dramatic & Orchestral**: Cinematic and intense
   - **Corporate & Professional**: Business-appropriate
   - **Nature Sounds**: Natural ambient sounds
3. Adjust volume level
4. Choose scope (single conversation or entire audio)

## MCP Agent Architecture

### Base Agent (`base_agent.py`)
- Abstract base class for all MCP agents
- Provides common functionality like request validation and error handling

### Dialogue Generator Agent (`dialogue_generator_agent.py`)
- Uses OpenAI GPT-4 to generate contextual dialogue
- Analyzes conversation history for context
- Generates speaker-appropriate responses

### Text-to-Speech Agent (`tts_agent.py`)
- Uses OpenAI TTS API to convert text to speech
- Maps speaker characteristics to appropriate voices
- Adjusts speech speed based on personality

### Background Music Agent (`music_agent.py`)
- Mixes audio with background music
- Supports multiple music types
- Adjustable volume levels
- Can target specific conversations or full audio

### Agent Coordinator (`agent_coordinator.py`)
- Orchestrates communication between agents
- Manages the complete artificial speaker workflow
- Handles error propagation and recovery

## Music Library Setup

Create background music files in `backend/music_library/`:
- `calm_ambient.mp3`
- `upbeat_background.mp3`
- `dramatic_orchestral.mp3`
- `corporate_background.mp3`
- `nature_sounds.mp3`

If music files are not available, the system will generate simple demo tones.

## Troubleshooting

### Common Issues

1. **OpenAI API Errors**
   - Verify API key is correctly set in `config.py`
   - Check API usage limits
   - Ensure internet connectivity

2. **Audio Processing Errors**
   - Verify ffmpeg is installed
   - Check audio file formats are supported
   - Ensure sufficient disk space

3. **Frontend Errors**
   - Check console for JavaScript errors
   - Verify API endpoints are accessible
   - Ensure CORS is properly configured

### Error Handling

The MCP agents include comprehensive error handling:
- Failed requests return detailed error messages
- Automatic fallbacks for missing music files
- Graceful degradation when services are unavailable

## Performance Considerations

- **OpenAI API**: Requests may take 5-10 seconds for complex operations
- **Audio Processing**: Large audio files may require more processing time
- **Memory Usage**: Background music mixing increases memory usage
- **Concurrent Requests**: Multiple AI operations may hit API rate limits

## Security Notes

- Store OpenAI API key securely
- Validate all user inputs
- Implement proper authentication for production use
- Consider implementing request rate limiting

## Future Enhancements

Potential improvements for the MCP architecture:
1. Voice cloning capabilities
2. Multiple language support
3. Real-time audio processing
4. Advanced music composition
5. Speaker emotion analysis
6. Conversation summarization
7. Audio quality enhancement

## Support

For issues or questions:
1. Check the console logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Test with simple examples first
4. Check OpenAI API status and quotas