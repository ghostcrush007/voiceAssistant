import streamlit as st
import os
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import soundfile as sf
from groq import Groq
import pyttsx3
import pyaudio
import wave
from dotenv import load_dotenv

# Load the Whisper model
stt_model_name = "openai/whisper-tiny"
stt_model = WhisperForConditionalGeneration.from_pretrained(stt_model_name)
stt_processor = WhisperProcessor.from_pretrained(stt_model_name)

# Initialize session state variables if not already set
if 'assistant_response' not in st.session_state:
    st.session_state.assistant_response = ""

# Function: Record Audio
def record_audio(output_path, duration=5, sample_rate=16000, channels=1):
    """Record audio from the microphone and save it as a .wav file."""
    chunk = 1024  # Record in chunks of 1024 samples
    format = pyaudio.paInt16  # 16-bit audio format
    p = pyaudio.PyAudio()

    st.write("Recording... Please wait.")
    stream = p.open(format=format, channels=channels, rate=sample_rate, input=True, frames_per_buffer=chunk)
    frames = []

    for _ in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(format))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
    st.success(f"Audio recorded and saved as {output_path}")

# Function: Speech-to-Text
def speech_to_text(audio_file_path):
    try:
        audio_input, sample_rate = sf.read(audio_file_path)
        input_features = stt_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").input_features
        predicted_ids = stt_model.generate(input_features, language='en')  # Ensure English transcription
        transcription = stt_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription.strip()
    except Exception as e:
        return f"Error during speech-to-text: {e}"

# Function: Generate Response
def generate_response(user_input):
    load_dotenv()
    client = Groq()  # Replace with your Groq API key
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_input}
    ]
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
    return response.strip()

# Function: Text-to-Speech
def text_to_speech(text):
    tts_engine = pyttsx3.init()
    rate = tts_engine.getProperty('rate')
    tts_engine.setProperty('rate', rate - 50)  # Adjust speech rate
    voices = tts_engine.getProperty('voices')
    tts_engine.setProperty('voice', voices[1].id)  # Change to a different voice
    tts_engine.say(text)
    tts_engine.runAndWait()

# Streamlit App
st.title("Voice Assistant with Streamlit")

# Button to Record Audio
if st.button("Record Audio"):
    recorded_audio_path = "recorded_audio.wav"
    record_audio(recorded_audio_path, duration=5)  # Record audio for 5 seconds

    # Process the recorded audio
    st.write("Processing the recorded audio...")
    user_query = speech_to_text(recorded_audio_path)
    st.write(f"You said: {user_query}")

    if user_query:
        # Generate response and store in session state
        st.write("Generating response...")
        st.session_state.assistant_response = generate_response(user_query)
        st.write(f"Assistant: {st.session_state.assistant_response}")

# Speak the output
if st.button("Speak the Output"):
    if st.session_state.assistant_response:
        st.write("Speaking the output...")
        text_to_speech(st.session_state.assistant_response)
    else:
        st.warning("No response to speak. Please record and generate a response first.")
