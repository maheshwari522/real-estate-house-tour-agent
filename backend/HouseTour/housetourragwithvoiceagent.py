# house_tour_voice_rag.py

# Install required packages first:
# pip install langchain openai faiss-cpu sentence-transformers tiktoken langchain-community speechrecognition pyaudio elevenlabs pydub numpy python-dotenv

import os
import wave
import pyaudio
import numpy as np
from io import BytesIO
from time import time
from typing import Optional
from dotenv import load_dotenv
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from openai import OpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from groq import Groq
from config import (
    ELEVENLABS_API_KEY,
    GROQ_API_KEY,
    OPENAI_API_KEY,
    FORMAT,
    CHANNELS,
    RATE,
    CHUNK,
    SILENCE_THRESHOLD,
    SILENCE_DURATION,
    PRE_SPEECH_BUFFER_DURATION,
    Voices
)

# Load environment variables
load_dotenv()

# Settings
openai_api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 2
PRE_SPEECH_BUFFER_DURATION = 1

# 1. Setup your House Tour RAG
folders = ["/Users/truptaditya/Documents/GitHub/HouseTour/Images"]

text_loader_kwargs = {'encoding': 'utf-8'}
documents = []

for folder in folders:
    doc_type = os.path.basename(folder.rstrip("/"))
    loader = DirectoryLoader(folder, glob="*.txt", loader_cls=TextLoader, loader_kwargs=text_loader_kwargs)
    folder_docs = loader.load()
    for doc in folder_docs:
        doc.metadata["doc_type"] = doc_type
        documents.append(doc)

print(f"✅ Loaded {len(documents)} documents.")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

llm = ChatOpenAI(
    temperature=0,
    model_name="gpt-4",
    openai_api_key=openai_api_key,
)

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 2. Voice Assistant Class

class VoiceAssistant:
    def __init__(
        self,
        voice_id: Optional[str] = Voices.ADAM):
        self.audio = pyaudio.PyAudio()
        #self.agent = Agent()
        self.voice_id = voice_id
        self.xi_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
        self.oai_client = OpenAI(api_key=OPENAI_API_KEY)
        self.g_client = Groq(api_key=GROQ_API_KEY)

    def is_silence(self, data):
        """
        Detect if the provided audio data is silence.

        Args:
            data (bytes): Audio data.

        Returns:
            bool: True if the data is considered silence, False otherwise.
        """
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        return rms < SILENCE_THRESHOLD

    def listen_for_speech(self):
        """
        Continuously detect silence and start recording when speech is detected.
        
        Returns:
            BytesIO: The recorded audio bytes.
        """
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        print("Listening for speech...")
        pre_speech_buffer = []
        pre_speech_chunks = int(PRE_SPEECH_BUFFER_DURATION * RATE / CHUNK)

        while True:
            data = stream.read(CHUNK)
            pre_speech_buffer.append(data)
            if len(pre_speech_buffer) > pre_speech_chunks:
                pre_speech_buffer.pop(0)

            if not self.is_silence(data):
                print("Speech detected, start recording...")
                stream.stop_stream()
                stream.close()
                return self.record_audio(pre_speech_buffer)

    def record_audio(self, pre_speech_buffer):
        stream = self.audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = pre_speech_buffer.copy()

        silent_chunks = 0
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            if self.is_silence(data):
                silent_chunks += 1
            else:
                silent_chunks = 0
            if silent_chunks > int(RATE / CHUNK * SILENCE_DURATION):
                break

        stream.stop_stream()
        stream.close()

        audio_bytes = BytesIO()
        with wave.open(audio_bytes, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
        audio_bytes.seek(0)
        return audio_bytes

    def speech_to_text(self, audio_bytes):
        """
        Transcribe speech to text using OpenAI.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        audio_bytes.seek(0)
        transcription = self.oai_client.audio.transcriptions.create(
            file=("temp.wav", audio_bytes.read()),
            model="whisper-1",
        )
        return transcription.text
    
    def speech_to_text_g(self, audio_bytes):
        """
        Transcribe speech to text using OpenAI.

        Args:
            audio_bytes (BytesIO): The audio bytes to transcribe.

        Returns:
            str: The transcribed text.
        """
        start = time()
        audio_bytes.seek(0)
        transcription = self.g_client.audio.transcriptions.create(
            file=("temp.wav", audio_bytes.read()),
            model="whisper-large-v3",
        )
        end = time()
        print(transcription)
        return transcription.text


    def text_to_speech(self, text, voice_id: Optional[str] = None):
        """
        Convert text to speech and return an audio stream.

        Args:
            text (str): The text to convert to speech.

        Returns:
            BytesIO: The audio stream.
        """
        voice_id = voice_id or self.voice_id
        response = self.xi_client.text_to_speech.convert(
            voice_id=voice_id,
            optimize_streaming_latency="0",
            output_format="mp3_22050_32",
            text=text,
            model_id="eleven_multilingual_v2",
        )

        audio_stream = BytesIO()

        for chunk in response:
            if chunk:
                audio_stream.write(chunk)

        audio_stream.seek(0)
        return audio_stream

    def audio_stream_to_iterator(self, audio_stream, format='mp3'):
        """
        Convert audio stream to an iterator of raw PCM audio bytes.

        Args:
            audio_stream (BytesIO): The audio stream.
            format (str): The format of the audio stream.

        Returns:
            bytes: The raw PCM audio bytes.
        """
        audio = AudioSegment.from_file(audio_stream, format=format)
        audio = audio.set_frame_rate(22050).set_channels(2).set_sample_width(2)  # Ensure the format matches pyaudio parameters
        raw_data = audio.raw_data

        chunk_size = 1024  # Adjust as necessary
        for i in range(0, len(raw_data), chunk_size):
            yield raw_data[i:i + chunk_size]

    def stream_audio(self, audio_bytes_iterator, rate=22050, channels=2, format=pyaudio.paInt16):
        """
        Stream audio in real-time.

        Args:
            audio_bytes_iterator (bytes): The raw PCM audio bytes.
            rate (int): The sample rate of the audio.
            channels (int): The number of audio channels.
            format (pyaudio format): The format of the audio.
        """
        stream = self.audio.open(format=format,
                                 channels=channels,
                                 rate=rate,
                                 output=True)

        try:
            for audio_chunk in audio_bytes_iterator:
                stream.write(audio_chunk)
        finally:
            stream.stop_stream()
            stream.close()
    

    def chat(self, query: str) -> str:
        start = time()
        response = qa_chain.invoke(query)
        end = time()
        print(f"🤖 Response: {response['result']}\n⏱️ Response Time: {end - start:.2f} seconds")
        return response['result']

    def run(self):
        while True:
            audio_bytes = self.listen_for_speech()
            text = self.speech_to_text(audio_bytes)
            print(f"📝 You said: {text}")

            response_text = self.chat(text)
            audio_stream = self.text_to_speech(response_text)
            audio_iterator = self.audio_stream_to_iterator(audio_stream)
            self.stream_audio(audio_iterator)

# 3. Run the Voice Assistant

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
