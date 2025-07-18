from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import whisper
import numpy as np
import asyncio
import json
import base64
import wave
import io
import tempfile
import os
import logging
from datetime import datetime
import uuid
import glob
import threading
import time
from collections import defaultdict

# Configure logging
def setup_logging():
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure logging format
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # Configure file handler for all logs
    file_handler = logging.FileHandler('logs/audio_processing.log', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler],
        format=log_format,
        datefmt=date_format
    )

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def _cleanup_session_files(session_id: str, session_audio_dir: str):
    """Clean up session files and directories"""
    try:
        import shutil
        if os.path.exists(session_audio_dir):
            shutil.rmtree(session_audio_dir)
            logger.info(f"[SESSION {session_id}] Session directory cleaned up: {session_audio_dir}")
        
        # Also clean up any JSON files in audio_files folder for this session
        audio_files_dir = "audio_files"
        if os.path.exists(audio_files_dir):
            session_json_files = glob.glob(os.path.join(audio_files_dir, f"*_{session_id}_*.json"))
            for json_file in session_json_files:
                try:
                    os.remove(json_file)
                    logger.info(f"[SESSION {session_id}] Cleaned up JSON file: {os.path.basename(json_file)}")
                except Exception as e:
                    logger.warning(f"[SESSION {session_id}] Failed to delete JSON file {json_file}: {str(e)}")
        
    except Exception as cleanup_error:
        logger.error(f"[SESSION {session_id}] Error during cleanup: {str(cleanup_error)}")

app = FastAPI(
    title="Whisper Real-time Transcription API",
    description="WebSocket API for real-time audio transcription using OpenAI's Whisper model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logger.info("Loading Whisper model: medium.en")
model = whisper.load_model("medium.en")
logger.info("Whisper model loaded successfully")



@app.get("/health")
async def health_check():
    """Health check endpoint for the server"""
    return {
        "status": "running",
        "model_loaded": True,
        "model_name": "large-v3",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/sessions/status")
async def get_sessions_status():
    """Get status of all active sessions"""
    try:
        with audio_processor.session_lock:
            sessions_info = []
            for session_id, session_info in audio_processor.sessions.items():
                sessions_info.append({
                    "session_id": session_id,
                    "total_chunks": session_info['total_chunks'],
                    "processed_chunks": session_info['processed_chunks'],
                    "complete": session_info['complete'],
                    "websocket_active": session_info['websocket_active'],
                    "pending_messages": len(session_info.get('pending_messages', [])),
                    "created_at": session_info['created_at'].strftime("%Y-%m-%d %H:%M:%S"),
                    "progress_percentage": round((session_info['processed_chunks'] / max(session_info['total_chunks'], 1)) * 100, 1)
                })
            
            return {
                "success": True,
                "active_sessions": len(sessions_info),
                "processing_queue_size": len(audio_processor.processing_queue),
                "sessions": sessions_info
            }
    except Exception as e:
        logger.error(f"Error getting sessions status: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/test-websocket")
async def test_websocket():
    """Test endpoint to send a test message to all connected websockets"""
    try:
        # Send a test message to all active sessions
        test_message = {
            "type": "transcription",
            "chunk_id": 999,
            "text": "This is a test transcription message from the server",
            "confidence": 0.95,
            "language": "en",
            "timestamp": int(datetime.now().timestamp() * 1000)
        }
        
        with audio_processor.session_lock:
            for session_id, session_info in audio_processor.sessions.items():
                if session_info['websocket_active']:
                    try:
                        websocket = session_info['websocket']
                        await websocket.send_json(test_message)
                        logger.info(f"Test message sent to session {session_id}")
                    except Exception as e:
                        logger.error(f"Failed to send test message to session {session_id}: {str(e)}")
        
        return {
            "success": True,
            "message": "Test message sent to all active sessions",
            "active_sessions": len(audio_processor.sessions)
        }
    except Exception as e:
        logger.error(f"Error in test websocket: {str(e)}")
        return {
            "success": False,
            "error": str(e)
        }

class AudioBuffer:
    def __init__(self):
        self.buffer = []
        self.sample_rate = 16000  # Whisper expects 16kHz audio
        self.channels = 1  # Whisper expects mono audio

    def add_audio(self, audio_data):
        self.buffer.extend(audio_data)

    def get_audio(self):
        return np.array(self.buffer)

    def clear(self):
        self.buffer = []

def get_session_audio_files(session_id):
    """Get all audio files for a specific session, sorted by chunk number"""
    session_dir = f"audio/session_{session_id}"
    if not os.path.exists(session_dir):
        return []
    
    # Get all .wav files in the session directory
    audio_files = glob.glob(os.path.join(session_dir, "chunk_*.wav"))
    
    # Sort by chunk number (extract number from filename)
    def extract_chunk_number(filename):
        basename = os.path.basename(filename)
        try:
            return int(basename.split('_')[1])
        except (IndexError, ValueError):
            return 0
    
    audio_files.sort(key=extract_chunk_number)
    return audio_files

@app.get("/process_session/{session_id}")
async def process_session_audio(session_id: str):
    """Process all audio files in a session and return transcriptions"""
    logger.info(f"Starting processing for session: {session_id}")
    
    audio_files = get_session_audio_files(session_id)
    if not audio_files:
        logger.warning(f"No audio files found for session: {session_id}")
        return {"error": "No audio files found for this session"}
    
    logger.info(f"Found {len(audio_files)} audio files for session: {session_id}")
    
    transcriptions = []
    total_files = len(audio_files)
    
    for i, audio_file in enumerate(audio_files, 1):
        try:
            chunk_number = int(os.path.basename(audio_file).split('_')[1])
            logger.info(f"[SESSION {session_id}] Processing chunk {chunk_number}/{total_files} - File: {os.path.basename(audio_file)}")
            
            # Process with Whisper model
            result = model.transcribe(audio_file)
            transcription_text = result["text"].strip()
            
            transcription_data = {
                "chunk": chunk_number,
                "filename": os.path.basename(audio_file),
                "text": transcription_text,
                "confidence": result.get("confidence", 0.0),
                "language": result.get("language", "en")
            }
            
            transcriptions.append(transcription_data)
            logger.info(f"[SESSION {session_id}] Chunk {chunk_number} processed - Text: '{transcription_text}'")
            
        except Exception as e:
            logger.error(f"[SESSION {session_id}] Error processing chunk {chunk_number}: {str(e)}")
            transcriptions.append({
                "chunk": chunk_number,
                "filename": os.path.basename(audio_file),
                "text": "",
                "error": str(e)
            })
    
    # Sort transcriptions by chunk number
    transcriptions.sort(key=lambda x: x["chunk"])
    
    # Create complete transcription text
    complete_text = " ".join([t["text"] for t in transcriptions if t["text"]])
    
    result = {
        "session_id": session_id,
        "total_chunks": total_files,
        "processed_chunks": len(transcriptions),
        "transcriptions": transcriptions,
        "complete_text": complete_text,
        "processing_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    logger.info(f"[SESSION {session_id}] Processing complete - {len(transcriptions)} chunks processed")
    return result

@app.get("/sessions")
async def list_sessions():
    """List all available sessions"""
    audio_dir = "audio"
    if not os.path.exists(audio_dir):
        return {"sessions": []}
    
    sessions = []
    for session_dir in os.listdir(audio_dir):
        if session_dir.startswith("session_"):
            session_id = session_dir.replace("session_", "")
            session_path = os.path.join(audio_dir, session_dir)
            
            # Count audio files
            audio_files = glob.glob(os.path.join(session_path, "chunk_*.wav"))
            
            # Get creation time
            creation_time = datetime.fromtimestamp(os.path.getctime(session_path))
            
            sessions.append({
                "session_id": session_id,
                "audio_files_count": len(audio_files),
                "created": creation_time.strftime("%Y-%m-%d %H:%M:%S"),
                "processed": False  # You can add logic to track if session was processed
            })
    
    # Sort by creation time (newest first)
    sessions.sort(key=lambda x: x["created"], reverse=True)
    
    return {"sessions": sessions}

@app.get("/transcription-files")
async def get_transcription_files():
    """Get all transcription text files from the transcriptions directory"""
    try:
        transcriptions_dir = "transcriptions"
        if not os.path.exists(transcriptions_dir):
            return {
                "success": True,
                "files": [],
                "total_files": 0,
                "directory": transcriptions_dir
            }
        
        files = []
        for filename in os.listdir(transcriptions_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(transcriptions_dir, filename)
                stat = os.stat(filepath)
                
                # Read file content
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                except Exception as e:
                    logger.warning(f"Could not read file {filename}: {e}")
                    content = ""
                
                files.append({
                    "name": filename,
                    "size": f"{stat.st_size / 1024:.1f} KB",
                    "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                    "content": content,
                    "content_length": len(content),
                    "filepath": filepath
                })
        
        # Sort by date (newest first)
        files.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "success": True,
            "files": files,
            "total_files": len(files),
            "directory": transcriptions_dir
        }
    except Exception as e:
        logger.error(f"Error listing transcription files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/transcription-file/{filename}")
async def get_transcription_file_content(filename: str):
    """Get content of a specific transcription file"""
    try:
        transcriptions_dir = "transcriptions"
        file_path = os.path.join(transcriptions_dir, filename)
        
        # Security check to prevent directory traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(transcriptions_dir)):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        stat = os.stat(file_path)
        
        return {
            "success": True,
            "filename": filename,
            "content": content,
            "size": f"{stat.st_size / 1024:.1f} KB",
            "date": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
            "lines": len(content.split('\n'))
        }
    except Exception as e:
        logger.error(f"Error reading transcription file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/delete-transcription/{filename}")
async def delete_transcription(filename: str):
    """Delete a specific transcription file"""
    try:
        transcriptions_dir = "transcriptions"
        file_path = os.path.join(transcriptions_dir, filename)
        
        # Security check to prevent directory traversal
        if not os.path.abspath(file_path).startswith(os.path.abspath(transcriptions_dir)):
            raise HTTPException(status_code=400, detail="Invalid filename")
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Delete the file
        os.remove(file_path)
        logger.info(f"Deleted transcription file: {filename}")
        
        return {
            "success": True,
            "message": f"Transcription file '{filename}' deleted successfully",
            "deleted_file": filename
        }
    except Exception as e:
        logger.error(f"Error deleting transcription: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())[:8]
    chunk_counter = 0
    total_messages = 0
    websocket_connection = websocket  # Store reference to websocket
    
    logger.info(f"=== NEW SESSION STARTED: {session_id} ===")
    
    # Create audio directory for this session
    session_audio_dir = f"audio/session_{session_id}"
    os.makedirs(session_audio_dir, exist_ok=True)
    logger.info(f"[SESSION {session_id}] Audio directory created: {session_audio_dir}")
    
    # Register this session with the audio processor
    audio_processor.register_session(session_id, session_audio_dir, websocket_connection)
    
    try:
        while True:
            # Check for pending transcription messages first
            pending_messages = []
            with audio_processor.session_lock:
                if session_id in audio_processor.sessions:
                    session_info = audio_processor.sessions[session_id]
                    if 'pending_messages' in session_info and session_info['pending_messages']:
                        pending_messages = session_info['pending_messages'].copy()
                        session_info['pending_messages'] = []
            
            # Send pending messages outside the lock
            for msg in pending_messages:
                try:
                    await websocket.send_json(msg)
                    logger.info(f"[SESSION {session_id}] Sent pending message: {msg['text']}")
                except Exception as e:
                    logger.error(f"[SESSION {session_id}] Failed to send pending message: {str(e)}")
            
            # Receive new data with timeout
            try:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=1.0)  # Increased timeout
            except asyncio.TimeoutError:
                # No new data, continue to check for pending messages
                continue
                
            total_messages += 1
            message = json.loads(data)
            
            logger.info(f"[SESSION {session_id}] MESSAGE {total_messages} RECEIVED - Type: {message.get('type', 'unknown')}")
            
            if message["type"] == "audio":
                chunk_counter += 1
                # Log audio reception
                audio_bytes = base64.b64decode(message["data"])
                audio_size = len(audio_bytes)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
                logger.info(f"[SESSION {session_id}] AUDIO CHUNK {chunk_counter} RECEIVED - Size: {audio_size} bytes, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                # Save audio chunk to file immediately
                chunk_filename = f"chunk_{chunk_counter}_{timestamp}.wav"
                chunk_filepath = os.path.join(session_audio_dir, chunk_filename)
                
                try:
                    with open(chunk_filepath, 'wb') as audio_file:
                        audio_file.write(audio_bytes)
                    
                    logger.info(f"[SESSION {session_id}] AUDIO CHUNK {chunk_counter} SAVED - File: {chunk_filename}")
                    
                    # Send acknowledgment back to client
                    await websocket.send_json({
                        "type": "audio_received",
                        "chunk": chunk_counter,
                        "filename": chunk_filename
                    })
                    
                    # Add to processing queue (background processing)
                    audio_processor.add_chunk_to_queue(session_id, chunk_filepath, chunk_counter)
                    
                except Exception as save_error:
                    logger.error(f"[SESSION {session_id}] FAILED TO SAVE AUDIO CHUNK {chunk_counter}: {str(save_error)}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Failed to save chunk {chunk_counter}",
                        "chunk": chunk_counter
                    })

            elif message["type"] == "end":
                logger.info(f"[SESSION {session_id}] SESSION ENDED - Total messages: {total_messages}, Total chunks: {chunk_counter}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                
                # Send final acknowledgment
                await websocket.send_json({
                    "type": "session_complete",
                    "total_chunks": chunk_counter,
                    "session_id": session_id
                })
                
                # Mark session as complete for background processing
                audio_processor.mark_session_complete(session_id)
                
                break
            elif message["type"] == "ping":
                # Handle ping messages for connection keep-alive
                await websocket.send_json({
                    "type": "pong",
                    "chunk_id": message.get("chunk_id", 0),
                    "timestamp": int(datetime.now().timestamp() * 1000)
                })
            else:
                logger.warning(f"[SESSION {session_id}] UNKNOWN MESSAGE TYPE: {message.get('type', 'unknown')}")

    except WebSocketDisconnect:
        logger.warning(f"[SESSION {session_id}] CLIENT DISCONNECTED - Total messages: {total_messages}, Processed chunks: {chunk_counter}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Mark websocket as disconnected but continue processing
        audio_processor.mark_websocket_disconnected(session_id)
    except Exception as e:
        logger.error(f"[SESSION {session_id}] ERROR: {str(e)} - Total messages: {total_messages}, Processed chunks: {chunk_counter}, Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        # Mark websocket as disconnected but continue processing
        audio_processor.mark_websocket_disconnected(session_id)
        await websocket.close()

# Global variables for WebSocket connections and processing
active_connections = []
processing_lock = threading.Lock()
processed_files = set()

class AudioProcessor:
    def __init__(self):
        self.running = False
        self.thread = None
        self.processed_files = set()
        self.sessions = {}  # Store session info: {session_id: {dir, websocket, chunks, complete}}
        self.processing_queue = []  # Queue of chunks to process
        self.session_lock = threading.Lock()
        self.queue_lock = threading.Lock()
        
    def start(self):
        """Start the audio processing thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_loop, daemon=True)
            self.thread.start()
            logger.info("Audio processing thread started")
    
    def stop(self):
        """Stop the audio processing thread"""
        self.running = False
        if self.thread:
            self.thread.join()
            logger.info("Audio processing thread stopped")
    
    def register_session(self, session_id: str, session_dir: str, websocket):
        """Register a new session for processing"""
        with self.session_lock:
            self.sessions[session_id] = {
                'dir': session_dir,
                'websocket': websocket,
                'chunks': [],
                'complete': False,
                'websocket_active': True,
                'pending_messages': [],
                'total_chunks': 0,
                'processed_chunks': 0,
                'created_at': datetime.now()
            }
            logger.info(f"[PROCESSOR] Registered session {session_id} - Total active sessions: {len(self.sessions)}")
    
    def add_chunk_to_queue(self, session_id: str, chunk_filepath: str, chunk_number: int):
        """Add a chunk to the processing queue"""
        with self.queue_lock:
            self.processing_queue.append({
                'session_id': session_id,
                'filepath': chunk_filepath,
                'chunk_number': chunk_number,
                'timestamp': datetime.now()
            })
            
            # Update session info
            with self.session_lock:
                if session_id in self.sessions:
                    self.sessions[session_id]['total_chunks'] = max(self.sessions[session_id]['total_chunks'], chunk_number)
            
            logger.info(f"[PROCESSOR] Added chunk {chunk_number} to queue for session {session_id} - Queue size: {len(self.processing_queue)}")
    
    def mark_session_complete(self, session_id: str):
        """Mark a session as complete"""
        with self.session_lock:
            if session_id in self.sessions:
                self.sessions[session_id]['complete'] = True
                total_chunks = self.sessions[session_id]['total_chunks']
                logger.info(f"[PROCESSOR] Marked session {session_id} as complete - Total chunks: {total_chunks}")
    
    def mark_websocket_disconnected(self, session_id: str):
        """Mark websocket as disconnected for a session"""
        with self.session_lock:
            if session_id in self.sessions:
                self.sessions[session_id]['websocket_active'] = False
                logger.info(f"[PROCESSOR] Marked websocket as disconnected for session {session_id}")
    
    def _process_loop(self):
        """Main processing loop that processes queued audio chunks"""
        while self.running:
            try:
                self._process_queue()
                self._cleanup_completed_sessions()
                time.sleep(0.5)  # Check every 500ms for better responsiveness
            except Exception as e:
                logger.error(f"Error in audio processing loop: {str(e)}")
                time.sleep(5)  # Wait longer on error
    
    def _process_queue(self):
        """Process audio chunks from the queue"""
        with self.queue_lock:
            if not self.processing_queue:
                return
            
            # Get next chunk to process
            chunk_info = self.processing_queue.pop(0)
        
        session_id = chunk_info['session_id']
        filepath = chunk_info['filepath']
        chunk_number = chunk_info['chunk_number']
        
        try:
            logger.info(f"[PROCESSOR] Processing chunk {chunk_number} for session {session_id}")
            
            # Process with Whisper model
            result = model.transcribe(filepath)
            transcription_text = result["text"].strip()
            
            if transcription_text:  # Only process if there's actual text
                # Create output data
                output_data = {
                    "session_id": session_id,
                    "chunk": chunk_number,
                    "filename": os.path.basename(filepath),
                    "text": transcription_text,
                    "confidence": result.get("confidence", 0.0),
                    "language": result.get("language", "en"),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Save transcription to file
                self._save_transcription_output(session_id, chunk_number, output_data)
                
                # Send message directly to websocket
                self._send_websocket_message_immediate(session_id, chunk_number, transcription_text, result)
                
                logger.info(f"[PROCESSOR] Chunk {chunk_number} processed - Text: '{transcription_text}'")
            else:
                logger.info(f"[PROCESSOR] No transcription text for chunk {chunk_number} (likely silence)")
            
            # Mark as processed and update session info
            file_key = f"{session_id}_{os.path.basename(filepath)}"
            self.processed_files.add(file_key)
            
            # Update session processed count
            with self.session_lock:
                if session_id in self.sessions:
                    self.sessions[session_id]['processed_chunks'] += 1
                    processed = self.sessions[session_id]['processed_chunks']
                    total = self.sessions[session_id]['total_chunks']
                    logger.info(f"[PROCESSOR] Session {session_id} progress: {processed}/{total} chunks processed")
            
        except Exception as e:
            logger.error(f"[PROCESSOR] Error processing {filepath}: {str(e)}")
            # Mark as processed to avoid retry loops
            file_key = f"{session_id}_{os.path.basename(filepath)}"
            self.processed_files.add(file_key)
    

    
    def _save_transcription_output(self, session_id, chunk_number, output_data):
        """Save transcription output to audio_files folder"""
        try:
            # Create audio_files directory if it doesn't exist
            output_dir = "audio_files"
            os.makedirs(output_dir, exist_ok=True)
            
            # Save JSON output for individual chunk
            json_filename = f"chunk_{chunk_number}_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.json"
            json_filepath = os.path.join(output_dir, json_filename)
            
            with open(json_filepath, 'w', encoding='utf-8') as json_file:
                json.dump(output_data, json_file, indent=2, ensure_ascii=False)
            
            # Save/append to single session transcription file in transcriptions folder
            transcriptions_dir = "transcriptions"
            os.makedirs(transcriptions_dir, exist_ok=True)
            
            # Use single filename per session (no timestamp in filename)
            session_txt_filename = f"transcription_session_{session_id}.txt"
            session_txt_filepath = os.path.join(transcriptions_dir, session_txt_filename)
            
            # Append transcription to session file (only content)
            with open(session_txt_filepath, 'a', encoding='utf-8') as session_file:
                session_file.write(f"{output_data['text']}\n")
            
            logger.info(f"[BACKGROUND] Output saved - JSON: {json_filename}, Transcription appended to: {session_txt_filename}")
            
        except Exception as e:
            logger.error(f"[BACKGROUND] Error saving output: {str(e)}")
    
    def _send_websocket_message_immediate(self, session_id: str, chunk_number: int, transcription_text: str, result):
        """Send transcription result to websocket immediately"""
        with self.session_lock:
            if session_id not in self.sessions:
                logger.warning(f"[PROCESSOR] Session {session_id} not found in sessions")
                return
            
            session_info = self.sessions[session_id]
            if not session_info['websocket_active']:
                logger.info(f"[PROCESSOR] Websocket not active for session {session_id}, skipping send")
                return
            
            websocket = session_info['websocket']
            
            try:
                transcription_message = {
                    "type": "transcription",
                    "chunk_id": chunk_number,
                    "text": transcription_text,
                    "confidence": result.get("confidence", 0.0),
                    "language": result.get("language", "en"),
                    "timestamp": int(datetime.now().timestamp() * 1000)  # Unix timestamp in milliseconds
                }
                
                # Store the message to be sent by the main WebSocket handler
                if 'pending_messages' not in session_info:
                    session_info['pending_messages'] = []
                session_info['pending_messages'].append(transcription_message)
                
                logger.info(f"[PROCESSOR] Message queued for session {session_id}, chunk {chunk_number}")
                    
            except Exception as e:
                logger.warning(f"[PROCESSOR] Failed to queue message for session {session_id}: {str(e)}")
                # Mark websocket as inactive
                session_info['websocket_active'] = False
    
    def _cleanup_completed_sessions(self):
        """Clean up sessions that are complete and all chunks processed"""
        with self.session_lock:
            sessions_to_remove = []
            
            for session_id, session_info in self.sessions.items():
                if session_info['complete']:
                    # Check if all chunks for this session have been processed
                    session_dir = session_info['dir']
                    if os.path.exists(session_dir):
                        audio_files = glob.glob(os.path.join(session_dir, "chunk_*.wav"))
                        all_processed = True
                        
                        for audio_file in audio_files:
                            file_key = f"{session_id}_{os.path.basename(audio_file)}"
                            if file_key not in self.processed_files:
                                all_processed = False
                                break
                        
                        if all_processed:
                            logger.info(f"[PROCESSOR] All chunks processed for session {session_id}, cleaning up")
                            self._cleanup_session_files(session_id, session_dir)
                            sessions_to_remove.append(session_id)
            
            # Remove completed sessions
            for session_id in sessions_to_remove:
                del self.sessions[session_id]
                logger.info(f"[PROCESSOR] Removed completed session {session_id}")
    
    def _cleanup_session_files(self, session_id: str, session_dir: str):
        """Clean up session files and directories"""
        try:
            import shutil
            if os.path.exists(session_dir):
                shutil.rmtree(session_dir)
                logger.info(f"[PROCESSOR] Session directory cleaned up: {session_dir}")
            
            # Also clean up any JSON files in audio_files folder for this session
            audio_files_dir = "audio_files"
            if os.path.exists(audio_files_dir):
                session_json_files = glob.glob(os.path.join(audio_files_dir, f"*_{session_id}_*.json"))
                for json_file in session_json_files:
                    try:
                        os.remove(json_file)
                        logger.info(f"[PROCESSOR] Cleaned up JSON file: {os.path.basename(json_file)}")
                    except Exception as e:
                        logger.warning(f"[PROCESSOR] Failed to delete JSON file {json_file}: {str(e)}")
            
        except Exception as cleanup_error:
            logger.error(f"[PROCESSOR] Error during cleanup: {str(cleanup_error)}")
    
    def _notify_ui_clients(self, output_data):
        """Send transcription results to connected UI clients"""
        if not active_connections:
            return
        
        message = {
            "type": "transcription_result",
            "data": output_data
        }
        
        # Send to all connected clients
        disconnected = []
        for websocket in active_connections:
            try:
                asyncio.create_task(websocket.send_json(message))
            except Exception as e:
                logger.warning(f"Failed to send to client: {str(e)}")
                disconnected.append(websocket)
        
        # Remove disconnected clients
        for websocket in disconnected:
            try:
                active_connections.remove(websocket)
            except ValueError:
                pass

# Initialize audio processor
audio_processor = AudioProcessor()

@app.on_event("startup")
async def startup_event():
    """Start the audio processing thread when the app starts"""
    audio_processor.start()

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the audio processing thread when the app shuts down"""
    audio_processor.stop()

@app.websocket("/ws/ui")
async def ui_websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for UI clients to receive real-time transcription results"""
    await websocket.accept()
    active_connections.append(websocket)
    logger.info(f"UI client connected. Total connections: {len(active_connections)}")
    
    try:
        while True:
            # Keep connection alive and handle any UI messages
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            elif message.get("type") == "get_status":
                await websocket.send_json({
                    "type": "status",
                    "processed_files": len(audio_processor.processed_files),
                    "active_connections": len(active_connections)
                })
                
    except WebSocketDisconnect:
        logger.info("UI client disconnected")
    except Exception as e:
        logger.error(f"Error in UI WebSocket: {str(e)}")
    finally:
        try:
            active_connections.remove(websocket)
        except ValueError:
            pass
        logger.info(f"UI client removed. Total connections: {len(active_connections)}")

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Whisper Real-time Transcription API server")
    uvicorn.run(app, host="192.168.1.13", port=8111) 