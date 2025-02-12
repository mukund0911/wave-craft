# WaveCrafter

WaveCrafter is a web application that enables users to edit speech audio by providing speaker-wise transcripts. Users can modify the transcripts, and the app generates new audio while preserving the original speaker's style and emotions.



## Features

- Upload multi-speaker audio files
- Automatic speaker diarization and transcription
- Speaker-wise transcript editing
- Text-to-speech conversion maintaining speaker characteristics
- Real-time 3D wave visualization background
- Responsive web design

## Tech Stack

### Frontend
- React.js
- Three.js (for 3D wave visualization)
- Axios (for API communication)
- React Router (for navigation)
- CSS3 for styling

### Backend
- Flask (Python web framework)
- Socket.IO (for real-time communication)
- AssemblyAI API (for speech-to-text conversion)
- Pydub (for audio processing)

### Deployment
- Frontend: GitHub Pages
- Backend: Heroku
- CI/CD: GitHub Actions

## Prerequisites

Before running the application, make sure you have the following installed:
- Node.js (v16 or higher)
- Python (v3.11.5)
- npm (Node Package Manager)
- pip (Python Package Manager)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/wave-crafter.git
cd wave-crafter
```

2. Set up the frontend:
```bash
cd frontend
npm install
```

3. Set up the backend:
```bash
cd ../backend
pip install -r requirements.txt
```

## API Configuration

1. Sign up for an AssemblyAI account at [https://www.assemblyai.com](https://www.assemblyai.com)
2. Get your API key from the dashboard
3. Create a `.env` file in the backend directory:
```
assembly_ai_key=your_api_key_here
```

## Running Locally

1. Start the backend server:
```bash
cd backend
python app.py
```

2. Start the frontend development server:
```bash
cd frontend
npm start
```

3. Open your browser and navigate to `http://localhost:3000`

## Deployment

### Frontend Deployment (GitHub Pages)

The frontend is automatically deployed to GitHub Pages using GitHub Actions when changes are pushed to the main branch. The workflow is defined in `frontend_deploy.yml`.

To configure:
1. Enable GitHub Pages in your repository settings
2. Set up the custom domain (if needed) in the repository settings
3. The CNAME record is automatically configured through the workflow

### Backend Deployment (Heroku)

The backend is automatically deployed to Heroku using GitHub Actions when changes are pushed to the main branch. The workflow is defined in `heroku_deploy.yml`.

To configure:
1. Create a new Heroku app
2. Add the following secrets to your GitHub repository:
   - `HEROKU_API_KEY`
   - `HEROKU_APP_NAME`
   - `HEROKU_EMAIL`
3. Add your AssemblyAI API key to Heroku's environment variables

## Environment Variables

### Frontend
Create a `.env` file in the frontend directory:
```
REACT_APP_API_URL=your_backend_url_here
```

### Backend
Create a `.env` file in the backend directory:
```
assembly_ai_key=your_assemblyai_api_key
```

## Project Structure

```
wave-crafter/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── styles/
│   │   └── App.js
│   ├── package.json
│   └── README.md
├── backend/
│   ├── models/
│   │   ├── speech_music_classifier/
│   │   └── speech_edit/
│   ├── app.py
│   ├── routes.py
│   └── requirements.txt
├── .github/
│   └── workflows/
│       ├── frontend_deploy.yml
│       └── heroku_deploy.yml
└── README.md
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

