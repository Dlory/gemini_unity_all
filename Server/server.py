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
pip install google-genai opencv-python pyaudio pillow mss taskgroup
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

client = genai.Client(http_options={"api_version": "v1beta"})

pya = pyaudio.PyAudio()

YOUR_SERVER_HOST = "0.0.0.0" 
YOUR_SERVER_PORT = 8765

async def check_and_upload_file(client_async,pdf_file_path,display_name):
    file_uri=None
    try:
        async for uploaded_file in await client_async.files.list():
            if uploaded_file.display_name ==display_name:
                print(f"file alreay exist: {uploaded_file.name},{uploaded_file.uri}")
                file_uri=uploaded_file.uri
                return file_uri
        if(file_uri is None):
            # 1. Asynchronously upload the PDF file and get the file reference (file.name)
            uploaded_file = await client_async.files.upload(file=pdf_file_path,config={"display_name": display_name})
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


class AudioLoop:
    def __init__(self, video_mode=DEFAULT_MODE):
        self.video_mode = video_mode

        self.audio_in_queue = None
        self.out_queue = None
        self.session = None

        self.send_text_task = None
        self.receive_audio_task = None
        self.play_audio_task = None
        self.debug_wave_file = None
        self.last_handle = None

    async def create_websocket_server(self):
        async with websockets.serve(
            self.handle_client_stream, YOUR_SERVER_HOST, YOUR_SERVER_PORT
        ):
            print(f"websocket server started，address: ws://{YOUR_SERVER_HOST}:{YOUR_SERVER_PORT}")
            print(f"wait for client's connect...")
            await asyncio.Future()

    async def handle_client_stream(self, client_ws, *args):
        try:
            async with (
                asyncio.TaskGroup() as tg,
            ):
                last_handle = client_ws.request.path if client_ws.request else None
                if(last_handle is not None and last_handle!="/"):
                    self.last_handle= last_handle[1:]  # remove leading '/'
                    print(f"handle_client_stream: {client_ws.remote_address}, last_handle: {last_handle}")

                tg.create_task(self.run(client_ws))

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            traceback.print_exception(EG)

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(
                input,
                "message > ",
            )
            if text.lower() == "q":
                break
            await self.session.send(input=text or ".", end_of_turn=True)

    def _get_frame(self, cap):
        # Read the frameq
        ret, frame = cap.read()
        # Check if the frame was read successfully
        if not ret:
            return None
        # Fix: Convert BGR to RGB color space
        # OpenCV captures in BGR but PIL expects RGB format
        # This prevents the blue tint in the video feed
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(frame_rgb)  # Now using RGB frame
        img.thumbnail([1024, 1024])

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        # This takes about a second, and will block the whole program
        # causing the audio pipeline to overflow if you don't to_thread it.
        cap = await asyncio.to_thread(
            cv2.VideoCapture, 0
        )  # 0 represents the default camera

        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

        # Release the VideoCapture object
        cap.release()

    def _get_screen(self):
        sct = mss.mss()
        monitor = sct.monitors[0]

        i = sct.grab(monitor)

        mime_type = "image/jpeg"
        image_bytes = mss.tools.to_png(i.rgb, i.size)
        img = PIL.Image.open(io.BytesIO(image_bytes))

        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)

        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_screen(self):

        while True:
            frame = await asyncio.to_thread(self._get_screen)
            if frame is None:
                break

            await asyncio.sleep(1.0)

            await self.out_queue.put(frame)

    async def send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            await self.session.send(input=msg)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        self.audio_stream = await asyncio.to_thread(
            pya.open,
            format=FORMAT,
            channels=CHANNELS,
            rate=SEND_SAMPLE_RATE,
            input=True,
            input_device_index=mic_info["index"],
            frames_per_buffer=CHUNK_SIZE,
        )
        if __debug__:
            kwargs = {"exception_on_overflow": False}
        else:
            kwargs = {}
        while True:
            data = await asyncio.to_thread(self.audio_stream.read, CHUNK_SIZE, **kwargs)
            if self.debug_wave_file:
                await asyncio.to_thread(self.debug_wave_file.writeframes, data)
            await self.out_queue.put({"data": data, "mime_type": "audio/pcm"})

    async def receive_audio(self,client_ws):
        new_handle = None
        "Background task to reads from the websocket and write pcm chunks to the output queue"
        while True:
            turn = self.session.receive()
            async for response in turn:
                if data := response.data:
                    self.audio_in_queue.put_nowait(data)
                    continue
                if text := response.text:
                    print(text, end="")
                if response.session_resumption_update:
                    new_handle = response.session_resumption_update.new_handle
                    if new_handle != self.last_handle:
                        print(f"\nSession handle updated: {new_handle}\n")
                        self.last_handle = new_handle
                        client_ws.send(f'{"type":"update_handle","content":{new_handle}}')
                if response.go_away is not None:
                    # The connection will soon be terminated
                    print(response.go_away.time_left)
                    client_ws.send(f'{"type":"reconnect","content":{response.go_away.time_left}}')
                if response.server_content is not None and response.server_content.generation_complete is True:
                    # The generation is complete
                    print("response.server_content.generation_complete")

            # If you interrupt the model, it sends a turn_complete.
            # For interruptions to work, we need to stop playback.
            # So empty out the audio queue because it may have loaded
            # much more audio than has played yet.
            while not self.audio_in_queue.empty():
                self.audio_in_queue.get_nowait()

    async def play_audio(self):
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


    async def receive_client_message(self, client_ws):
            try:
                async for message in client_ws:
                    if isinstance(message, bytes):
                        #  if self.debug_wave_file:
                        #     await asyncio.to_thread(self.debug_wave_file.writeframes, message)
                        await self.session.send_realtime_input(media=types.Blob(data=message, mime_type="audio/pcm"))
                        
                    elif isinstance(message, str):
                        try:
                            data = json.loads(message)
                            data_type = data.get('type')
                            
                            if data_type == 'image':
                                image_data = base64.b64decode(data['data'])
                                img = PIL.Image.open(io.BytesIO(image_data))

                                # Save image locally for debug
                                img.save("debug_screen.jpg", format="JPEG")

                                image_io = io.BytesIO()
                                img.save(image_io, format="jpeg")
                                image_io.seek(0)

                                image_bytes = image_io.read()

                                await self.out_queue.put({"data": image_bytes, "mime_type": "image/jpeg"})
                            elif data_type == 'text':
                                await self.session.send(input=data['content'] or ".", end_of_turn=True)
                                
                        except json.JSONDecodeError:
                            print("invalid json message received from client")
                            
            except websockets.exceptions.ConnectionClosed:
                print("client disconnected")
            except Exception as e:
                print(f"receive_client_message error: {e}")
            finally:
                pass

    async def run(self,client_ws):
        #SYSTEM_INSTRUCTION = "You are an expert assistant. Please answer all user questions strictly based on the content of the uploaded PDF knowledge document. If the answer is not in the document, say 'I cannot find that information in the provided document.'"
        try:
            self.debug_wave_file = wave.open("client_audio_debug.wav", 'wb')
            self.debug_wave_file.setnchannels(CHANNELS)
            self.debug_wave_file.setsampwidth(pya.get_sample_size(FORMAT))
            self.debug_wave_file.setframerate(SEND_SAMPLE_RATE)
            CONFIG = types.LiveConnectConfig.model_validate({
                "response_modalities": ["AUDIO"],
                "session_resumption": {
                    'handle': self.last_handle,
                },
                #"system_instruction":SYSTEM_INSTRUCTION
            })
            async with (
                client.aio.live.connect(model=MODEL, config=CONFIG) as session,
                asyncio.TaskGroup() as tg,
            ):
                self.session = session
                # success = await setup_live_session_with_pdf_context(
                #         live_session=session,
                #         pdf_file_path="https://storage.googleapis.com/gin_bucket/pdfs/P800R%20V5%20user%20manual%20-V3.2-20231221.pdf"
                #     )
                # if not success:
                #     print("Failed to set up PDF context. Proceeding without knowledge base.")
                self.audio_in_queue = asyncio.Queue()
                self.out_queue = asyncio.Queue(maxsize=50)

                send_text_task = tg.create_task(self.send_text())
                tg.create_task(self.send_realtime())
                #tg.create_task(self.listen_audio())
                # if self.video_mode == "camera":
                #     tg.create_task(self.get_frames())
                # elif self.video_mode == "screen":
                #     tg.create_task(self.get_screen())

                tg.create_task(self.receive_client_message(client_ws))
                tg.create_task(self.receive_audio(client_ws))
                tg.create_task(self.play_audio())

                await send_text_task
                raise asyncio.CancelledError("User requested exit")

        except asyncio.CancelledError:
            pass
        except ExceptionGroup as EG:
            if(hasattr(self, 'audio_stream') and self.audio_stream is not None):
                self.audio_stream.close()
            traceback.print_exception(EG)
        finally:
            if self.debug_wave_file:
                self.debug_wave_file.close()


async def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default=DEFAULT_MODE,
        help="pixels to stream from",
        choices=["camera", "screen", "none"],
    )
    args = parser.parse_args()

    # Create the AudioLoop instance
    main_instance = AudioLoop(video_mode=args.mode)

    # Use asyncio TaskGroup to run both tasks concurrently
    async with asyncio.TaskGroup() as tg:
        tg.create_task(main_instance.create_websocket_server())  # Run websocket server


if __name__ == "__main__":
    asyncio.run(main())  # Start the main function
