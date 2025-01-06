import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
st.title("Voice Assistant using GROQ")
start_recording_button = st.button("Start Recording")
stop_recording_button = st.button("Stop Recording")
stop_flag = False
transcribed_text = ""
duration = 10
sample_rate = 44100
audio_file = "audio.wav"
recorded_audio = None
client = Groq()

if 'recorded_audio' not in st.session_state:
    st.session_state.recorded_audio = None

def output_based_on_generated_audio_text(transcribed_text):
    completion = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role":"system","content":"you are a helpful assistant"},
              {"role":"user","content":transcribed_text}],
    temperature=1,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
    )
    output = ""
    for chunk in completion:
        if chunk.choices[0].delta.content:
            output+=chunk.choices[0].delta.content
    st.write(output.strip())


def show_recorded_audio_as_text():
    
    filename = r"C:\Gen AI Project\Test for voice models\audio.wav"
    with open(filename, "rb") as file:
        transcription = client.audio.transcriptions.create(
        file=(filename, file.read()),
        model="distil-whisper-large-v3-en",
        response_format="verbose_json",
    )
        st.write(transcription.text)
        transcribed_text = transcription.text
        output_based_on_generated_audio_text(transcribed_text) 

def write_audio_file(recorded_audio):
    write(audio_file,sample_rate,recorded_audio)
    st.write("recoding ended...")
    show_recorded_audio_as_text()

if start_recording_button:
    st.write("Recording started...")
    st.session_state.recorded_audio = sd.rec(int(duration*sample_rate),samplerate=sample_rate,channels=1,dtype='int16')
    #sd.wait()
    if stop_flag==False:
        sd.wait()
        write_audio_file(st.session_state.recorded_audio)


if stop_recording_button:
    if st.session_state.recorded_audio is not None:
        sd.stop()
        stop_flag = True
        write_audio_file(st.session_state.recorded_audio)
    else:
        st.write("No audio is recorded yet")
    
