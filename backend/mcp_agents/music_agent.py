import base64
import requests
from pydub import AudioSegment
from io import BytesIO
from typing import Dict, Any, Optional
from .base_agent import MCPAgent
import os
import tempfile

class BackgroundMusicAgent(MCPAgent):
    def __init__(self, music_library_path: str = "backend/music_library"):
        super().__init__("background_music", ["add_background_music", "list_music", "mix_audio", "download_music"])
        self.music_library_path = music_library_path
        os.makedirs(music_library_path, exist_ok=True)
        
        # Jamendo API base URL (free, no API key required for basic access)
        self.jamendo_api_base = "https://api.jamendo.com/v3.0"
        
        # Music type to Jamendo tag mapping
        self.music_type_tags = {
            "calm": ["ambient", "chillout", "relaxing"],
            "upbeat": ["energetic", "upbeat", "electronic"],
            "dramatic": ["orchestral", "cinematic", "dramatic"],
            "corporate": ["corporate", "motivational", "background"],
            "nature": ["ambient", "nature", "meditation"]
        }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        action = request.get("action")
        
        if action == "add_background_music":
            return await self._add_background_music(request)
        elif action == "list_music":
            return await self._list_music(request)
        elif action == "mix_audio":
            return await self._mix_audio(request)
        elif action == "download_music":
            return await self._download_music(request)
        else:
            return self.create_response(False, error=f"Unknown action: {action}")
    
    async def _add_background_music(self, request: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["audio_base64", "music_type", "volume_level"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        try:
            audio_base64 = request["audio_base64"]
            music_type = request["music_type"]
            volume_level = float(request["volume_level"])
            target_speaker = request.get("target_speaker", None)
            
            # Convert base64 audio to AudioSegment
            audio_bytes = base64.b64decode(audio_base64)
            audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
            
            # Get background music (download if not available)
            background_music = await self._get_background_music(music_type, len(audio))
            if not background_music:
                return self.create_response(False, error=f"Failed to get background music for type: {music_type}")
            
            # Adjust volume
            background_music = background_music - (60 - volume_level * 60)  # Convert 0-1 to dB adjustment
            
            # Mix audio with background music
            mixed_audio = audio.overlay(background_music)
            
            # Convert back to base64
            buffer = BytesIO()
            mixed_audio.export(buffer, format="wav")
            mixed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return self.create_response(True, {
                "mixed_audio_base64": mixed_base64,
                "music_type": music_type,
                "volume_level": volume_level,
                "target_speaker": target_speaker
            })
            
        except Exception as e:
            self.logger.error(f"Music mixing error: {str(e)}")
            return self.create_response(False, error=f"Failed to add background music: {str(e)}")
    
    async def _list_music(self, request: Dict[str, Any]) -> Dict[str, Any]:
        return self.create_response(True, {
            "available_music_types": list(self.music_tracks.keys()),
            "music_library_path": self.music_library_path
        })
    
    async def _mix_audio(self, request: Dict[str, Any]) -> Dict[str, Any]:
        required_fields = ["audio_segments"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        try:
            audio_segments = request["audio_segments"]
            mixed_audio = AudioSegment.silent(duration=0)
            
            for segment in audio_segments:
                audio_bytes = base64.b64decode(segment["audio_base64"])
                audio = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
                mixed_audio += audio
            
            buffer = BytesIO()
            mixed_audio.export(buffer, format="wav")
            mixed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return self.create_response(True, {
                "mixed_audio_base64": mixed_base64,
                "segment_count": len(audio_segments)
            })
            
        except Exception as e:
            self.logger.error(f"Audio mixing error: {str(e)}")
            return self.create_response(False, error=f"Failed to mix audio: {str(e)}")
    
    def _generate_demo_music(self, music_type: str, duration_ms: int) -> AudioSegment:
        """Generate demo background music tones when actual music files aren't available"""
        from pydub.generators import Sine
        
        tone_mapping = {
            "calm": 220,      # A3 - calming low tone
            "upbeat": 440,    # A4 - more energetic
            "dramatic": 330,  # E4 - dramatic middle tone
            "corporate": 349, # F4 - professional tone
            "nature": 196     # G3 - natural, earthy tone
        }
        
        frequency = tone_mapping.get(music_type, 220)
        tone = Sine(frequency).to_audio_segment(duration=duration_ms)
        
        # Make it very quiet for background
        return tone - 40  # Reduce volume by 40dB
    
    async def _download_music(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Download music from Jamendo API"""
        required_fields = ["music_type"]
        if not self.validate_request(request, required_fields):
            return self.create_response(False, error="Missing required fields")
        
        try:
            music_type = request["music_type"]
            duration_limit = request.get("duration_limit", 300)  # 5 minutes max
            
            # Search for music on Jamendo
            track_info = await self._search_jamendo_music(music_type, duration_limit)
            if not track_info:
                return self.create_response(False, error=f"No suitable music found for type: {music_type}")
            
            # Download the track
            music_audio = await self._download_jamendo_track(track_info)
            if not music_audio:
                return self.create_response(False, error="Failed to download music track")
            
            # Convert to base64
            buffer = BytesIO()
            music_audio.export(buffer, format="mp3")
            music_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return self.create_response(True, {
                "music_base64": music_base64,
                "track_info": track_info,
                "duration": len(music_audio)
            })
            
        except Exception as e:
            self.logger.error(f"Music download error: {str(e)}")
            return self.create_response(False, error=f"Failed to download music: {str(e)}")
    
    async def _get_background_music(self, music_type: str, target_duration: int) -> Optional[AudioSegment]:
        """Get background music, downloading if necessary"""
        try:
            # First check if we have cached music
            cache_file = os.path.join(self.music_library_path, f"{music_type}_cached.mp3")
            
            if os.path.exists(cache_file):
                background_music = AudioSegment.from_file(cache_file)
            else:
                # Download from Jamendo
                download_request = {
                    "action": "download_music",
                    "music_type": music_type,
                    "duration_limit": min(target_duration // 1000 + 60, 300)  # Add buffer, max 5min
                }
                
                result = await self._download_music(download_request)
                if not result["success"]:
                    # Fallback to demo music
                    return self._generate_demo_music(music_type, target_duration)
                
                # Save to cache and load
                music_bytes = base64.b64decode(result["data"]["music_base64"])
                with open(cache_file, 'wb') as f:
                    f.write(music_bytes)
                
                background_music = AudioSegment.from_file(BytesIO(music_bytes))
            
            # Adjust length to match target duration
            if len(background_music) < target_duration:
                loops_needed = (target_duration // len(background_music)) + 1
                background_music = background_music * loops_needed
            
            return background_music[:target_duration]
            
        except Exception as e:
            self.logger.error(f"Error getting background music: {str(e)}")
            return self._generate_demo_music(music_type, target_duration)
    
    async def _search_jamendo_music(self, music_type: str, duration_limit: int) -> Optional[Dict]:
        """Search for suitable music on Jamendo"""
        try:
            tags = self.music_type_tags.get(music_type, ["background"])
            
            # Search parameters
            params = {
                "client_id": "background_music_app",  # Free tier client ID
                "format": "json",
                "limit": 10,
                "tags": "+".join(tags),
                "duration_min": 30,
                "duration_max": duration_limit,
                "ccmixter": "false",  # Ensure Creative Commons
                "include": "musicinfo"
            }
            
            # Make API request
            response = requests.get(f"{self.jamendo_api_base}/tracks/", params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                tracks = data.get("results", [])
                
                if tracks:
                    # Select the first suitable track
                    track = tracks[0]
                    return {
                        "id": track["id"],
                        "name": track["name"],
                        "artist": track["artist_name"],
                        "duration": track["duration"],
                        "audio_download": track["audiodownload"],
                        "license": track.get("license_ccurl", "https://creativecommons.org/licenses/by-sa/3.0/")
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Jamendo search error: {str(e)}")
            return None
    
    async def _download_jamendo_track(self, track_info: Dict) -> Optional[AudioSegment]:
        """Download a track from Jamendo"""
        try:
            download_url = track_info["audio_download"]
            
            # Download the audio file
            response = requests.get(download_url, timeout=30)
            
            if response.status_code == 200:
                # Load audio from bytes
                audio_bytes = BytesIO(response.content)
                return AudioSegment.from_file(audio_bytes)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Track download error: {str(e)}")
            return None