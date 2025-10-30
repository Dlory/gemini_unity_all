# -*- coding: utf-8 -*-
# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
## Setup

To install the dependencies for this script, run:

```

pip install google-genai opencv-python pyaudio pillow mss taskgroup websockets

```

Before running this script, ensure the `GOOGLE_API_KEY` environment
variable is set to the api-key you obtained from Google AI Studio.

Important: **Use headphones**. This script uses the system default audio
input and output, which often won't include echo cancellation. So to prevent
the model from interrupting itself it is important that you use headphones.

"""

import asyncio
import base64
import io
import os
import sys
import traceback
import base64
import json
import cv2
import pyaudio
import PIL.Image
import mss
import argparse
import wave
from google import genai
from google.genai import types
from google.genai.errors import APIError
import websockets

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup

    asyncio.TaskGroup = taskgroup.TaskGroup
    ExceptionGroup = exceptiongroup.ExceptionGroup
else:
    ExceptionGroup = ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 1024

MODEL = "models/gemini-live-2.5-flash-preview"
PDF_FILE_NAME = "knowledge_base.pdf"
DEFAULT_MODE = "screen"

# Initialize global client and pyaudio instance
client = genai.Client(http_options={"api_version": "v1beta"})
pya = pyaudio.PyAudio()

YOUR_SERVER_HOST = "0.0.0.0"
YOUR_SERVER_PORT = 8765

# --- File Utility Functions (Remain the same) ---

async def check_and_upload_file(client_async, pdf_file_path, display_name):
    file_uri = None
    try:
        async for uploaded_file in await client_async.files.list():
            if uploaded_file.display_name == display_name:
                print(f"file already exist: {uploaded_file.name},{uploaded_file.uri}")
                file_uri = uploaded_file.uri
                return file_uri
        if file_uri is None:
            # 1. Asynchronously upload the PDF file and get the file reference (file.name)
            uploaded_file = await client_async.files.upload(file=pdf_file_path, config={"display_name": display_name})
            file_uri = uploaded_file.uri
            print(f"File uploaded successfully. uri: {file_uri}")
            return file_uri
    except APIError as e:
        print(f"File upload failed: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during file upload: {e}")
        return False


async def setup_live_session_with_pdf_context(
    live_session,
    pdf_file_path: str,
) -> bool:
    file_part = types.Part.from_uri(file_uri=pdf_file_path, mime_type="application/pdf")

    initial_context_turn = types.Content(
        role="user",
        parts=[file_part]
    )

    print("--- Step 3: Sending context to Live API session ---")
    try:
        await live_session.send_client_content(
            turns=[initial_context_turn],
            turn_complete=True
        )
        print("Initial PDF knowledge base context successfully sent to Live API.")
        return True

    except Exception as e:
        print(f"Failed to send Live API context: {e}")
        return False
    finally:
        pass


# --- Client Handler Class (New/Refactored) ---

class ClientHandler:
    """
    Handles a single client's websocket connection and Live API session.
    All state (session, queues, handle) is local to this instance.
    """
    def __init__(self, client_ws, initial_handle=None):
        self.client_ws = client_ws
        self.session = None

        # Queues are specific to this handler instance
        self.audio_in_queue = asyncio.Queue()  # Queue for audio chunks from Gemini
        self.out_queue = asyncio.Queue(maxsize=50) # Queue for media/text to Gemini
        self.current_handle = initial_handle

        # Debug flag/file for client audio (optional)
        self.debug_wave_file = None

    async def send_text(self):
        """Allows server admin to send text via console."""
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            # Send message to the current client's session
            await self.session.send(input=text or ".", end_of_turn=True)

    async def send_realtime(self):
        """Forwards media/image data from client queue to Gemini session."""
        while True:
            # Waits for media from receive_client_message
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def receive_client_message(self):
        """Reads messages (audio, image, text) from the client websocket."""
        try:
            async for message in self.client_ws:
                if isinstance(message, bytes):
                    # Raw audio data from client
                    await self.session.send_realtime_input(audio=types.Blob(data=message, mime_type="audio/pcm"))

                elif isinstance(message, str):
                    # JSON message from client (text or image metadata)
                    try:
                        data = json.loads(message)
                        data_type = data.get('type')

                        if data_type == 'image':
                            # Client sent an image base64 string
                            image_data = base64.b64decode(data['data'])
                            img = PIL.Image.open(io.BytesIO(image_data))

                            # Save image locally for debug
                            img.save(f"debug_screen_{self.client_ws.remote_address[1]}.jpg", format="JPEG")

                            image_io = io.BytesIO()
                            img.save(image_io, format="jpeg")
                            image_io.seek(0)

                            image_bytes = image_io.read()

                            # Put image into out_queue to be sent by send_realtime
                            #await self.out_queue.put({"data": image_bytes, "mime_type": "image/jpeg"})
                            await self.session.send_realtime_input(media=types.Blob(data=image_bytes, mime_type="image/jpeg"))

                        elif data_type == 'text':
                            # Client sent text message
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": data['content']}]}, turn_complete=True
                            )

                    except json.JSONDecodeError:
                        print(f"Client {self.client_ws.remote_address} sent invalid json message.")

        except websockets.exceptions.ConnectionClosed:
            print(f"Client {self.client_ws.remote_address} disconnected.")
        except Exception as e:
            print(f"receive_client_message error for {self.client_ws.remote_address}: {e}")
        finally:
            # Raise CancelledError to shut down other tasks in the group gracefully
            raise asyncio.CancelledError("Client disconnected or error occurred.")


    async def receive_audio(self):
        """Reads responses from the Live API session and handles audio/text/updates."""
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    # Received audio chunk
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    # Received text response (streaming)
                    print(f"[{self.client_ws.remote_address[1]}] {text}", end="", flush=True)

                if response.session_resumption_update:
                    # Session handle updated
                    new_handle = response.session_resumption_update.new_handle
                    if new_handle != self.current_handle:
                        print(f"\n[{self.client_ws.remote_address[1]}] Session handle updated: {new_handle}\n")
                        self.current_handle = new_handle
                        # Notify client of the new handle (client must implement receiving this)
                        await self.client_ws.send(json.dumps({"type": "update_handle", "content": new_handle}))

                if response.go_away is not None:
                    # The connection will soon be terminated by the server
                    print(f"\n[{self.client_ws.remote_address[1]}] Live API connection closing in {response.go_away.time_left}s.")
                    await self.client_ws.send(json.dumps({"type": "reconnect", "content": response.go_away.time_left}))
                if response.server_content is not None and response.server_content.generation_complete is True:
                    # The generation is complete
                    print(f"[{self.client_ws.remote_address[1]}] Generation complete.")


            # If an interruption occurs (turn_complete is sent), we empty the audio queue
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
        """Plays audio chunks received from the Live API session."""
        stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=RECEIVE_SAMPLE_RATE,
            output=True,
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)


    async def run(self):
        """Initializes and runs the Gemini Live API session for this client."""
        # SYSTEM_INSTRUCTION = "You are an expert assistant. Please answer all user questions strictly based on the content of the uploaded PDF knowledge document. If the answer is not in the document, say 'I cannot find that information in the provided document.'"
        try:
            CONFIG = types.LiveConnectConfig.model_validate({
                "response_modalities": ["AUDIO"],
                "session_resumption": {
                    'handle': self.current_handle, # Use the handle specific to this client
                },
                #"system_instruction":SYSTEM_INSTRUCTION
            })
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session # This session is local to the ClientHandler instance

                # Optional: Send PDF context (example is commented out)
                # success = await setup_live_session_with_pdf_context(
                #         live_session=session,
                #         pdf_file_path="https://storage.googleapis.com/gin_bucket/pdfs/P800R%20V5%20user%20manual%20-V3.2-20231221.pdf"
                #     )
                # if not success:
                #     print("Failed to set up PDF context. Proceeding without knowledge base.")

                # Start the tasks for this client's session
                tg.create_task(self.send_realtime())
                tg.create_task(self.receive_client_message())
                tg.create_task(self.receive_audio())
                tg.create_task(self.play_audio())

                # Task for server admin text input (optional)
                admin_task = tg.create_task(self.send_text())
                await admin_task
                raise asyncio.CancelledError("Admin requested exit") # If admin exits, close all connections

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            traceback.print_exception(EG)
        finally:
            if self.debug_wave_file:
                self.debug_wave_file.close()


# --- Server Manager Class (Replaces original AudioLoop's server role) ---

class ServerManager:
    """Manages the overall websocket server and spawns ClientHandlers."""
    def __init__(self, video_mode):
        self.video_mode = video_mode
        self.active_handlers = {}

    async def handle_client_stream(self, client_ws, *args):
        """Called by websockets.serve for every new client connection."""
        client_address = client_ws.remote_address
        # Get session handle from the path if provided (e.g., ws://host:port/handle_xyz)
        path_handle = client_ws.request.path[1:] if client_ws.request and client_ws.request.path else None

        print(f"New client connected: {client_address}, initial handle: {path_handle or 'None'}")

        # 1. Create a new, independent handler for THIS client
        handler = ClientHandler(client_ws, initial_handle=path_handle)
        self.active_handlers[client_address] = handler

        try:
            # 2. Run the client's session logic
            await handler.run()
        except asyncio.CancelledError:
            # This is the expected way to exit on client disconnect or admin exit
            pass
        except Exception as e:
            print(f"An error occurred in ClientHandler for {client_address}: {e}")
            traceback.print_exc()
        finally:
            # 3. Clean up when the client disconnects
            if client_address in self.active_handlers:
                del self.active_handlers[client_address]
            print(f"Client disconnected and handler cleaned up: {client_address}")

    async def create_websocket_server(self):
        """Starts the main websocket server."""
        async with websockets.serve(
            self.handle_client_stream, YOUR_SERVER_HOST, YOUR_SERVER_PORT
        ):
            print(f"WebSocket server started, address: ws://{YOUR_SERVER_HOST}:{YOUR_SERVER_PORT}")
            print(f"Wait for client's connect...")
            await asyncio.Future()


# --- Main Execution ---

async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from (Note: Client-side streaming is currently implemented)",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    # Create the ServerManager instance
    server_manager = ServerManager(video_mode=args.mode)

    # Use asyncio TaskGroup to run both tasks concurrently
    async with asyncio.TaskGroup() as tg:
        tg.create_task(server_manager.create_websocket_server())


if __name__ == "__main__":
    try:
        asyncio.run(main())  # Start the main function
    except KeyboardInterrupt:
        print("Server shutdown by user.")
    except Exception as e:
        print(f"Fatal error: {e}")