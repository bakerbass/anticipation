import sys
import time
import midi2audio
import transformers
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from transformers import AutoModelForCausalLM
from mido import MidiFile, tempo2bpm

from IPython.display import Audio

from anticipation import ops
from anticipation.sample import generate
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi, midi_to_events
from anticipation.visuals import visualize
from anticipation.config import *
from anticipation.vocab import *

SMALL_MODEL = 'stanford-crfm/music-small-800k'     # faster inference, worse sample quality
MEDIUM_MODEL = 'stanford-crfm/music-medium-800k'   # slower inference, better sample quality
LARGE_MODEL = 'stanford-crfm/music-large-800k'     # slowest inference, best sample quality

# load an anticipatory music transformer
start = time.time()
model = AutoModelForCausalLM.from_pretrained(LARGE_MODEL)
end = time.time()
print("Model loaded in ", end-start, " seconds")
# a MIDI synthesizer
fs = midi2audio.FluidSynth('./8bitsf.sf2')

# the MIDI synthesis script
def synthesize(fs, tokens):
    mid = events_to_midi(tokens)
    mid.save('tmp.mid')
    fs.midi_to_audio('tmp.mid', 'tmp.wav')
    return 'tmp.wav'

def normalize_wav(file_path):
    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Normalize the data
    data = data / np.max(np.abs(data))

    # Write the normalized WAV file
    wavfile.write(file_path, sample_rate, data.astype(np.float32))
    return file_path

def plot_wav(file_path, bpm, start_bar, end_bar):
    # Read the WAV file
    sample_rate, data = wavfile.read(file_path)

    # Create a time array in seconds
    time = np.linspace(0, len(data) / sample_rate, num=len(data))

    # Plot the waveform
    plt.figure(figsize=(12, 6))
    plt.plot(time, data)
    plt.title("Waveform of " + file_path)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")

    # Calculate the positions of the vertical bars
    beats_per_bar = 4  # Assuming 4/4 time signature
    seconds_per_beat = 60 / bpm
    seconds_per_bar = beats_per_bar * seconds_per_beat

    for bar in range(start_bar, end_bar + 1):
        plt.axvline(x=bar * seconds_per_bar, color='r', linestyle='--')

    plt.show()

def detect_bpm(midi_file_path):
    midi = MidiFile(midi_file_path)
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                bpm = tempo2bpm(tempo)
                return bpm
    return None

def process_midi_file(midi_file_path, start_bar, end_bar):
    bpm = detect_bpm(midi_file_path)
    if bpm is None:
        print("BPM not detected. Using default values.")
        bpm = 120  # Default BPM

    beats_per_bar = 4  # Assuming 4/4 time signature
    seconds_per_beat = 60 / bpm
    seconds_per_bar = beats_per_bar * seconds_per_beat

    start_time = start_bar * seconds_per_bar
    end_time = end_bar * seconds_per_bar

    start = time.time()
    events = midi_to_events(midi_file_path)
    end = time.time()
    print("MIDI converted in ", end-start, " seconds")

    segment = events
    segment = ops.translate(segment, -ops.min_time(segment, seconds=False))

    history = ops.clip(segment, 0, start_time + .01, clip_duration=False)
    anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, start_time + .01, end_time, clip_duration=False)]

    start = time.time()
    inpainted = generate(model, start_time, end_time, inputs=history, controls=anticipated, top_p=.95)
    end = time.time()
    print("Generated in ", end-start, " seconds")

    out = synthesize(fs, ops.combine(inpainted, anticipated))
    out = normalize_wav(out)
    plot_wav(out, bpm, start_bar, end_bar)
    Audio(out)

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python anticipation_inputs.py <midi_file_path> <start_bar> <end_bar>")
        sys.exit(1)
    midi_file_path = sys.argv[1]
    start_bar = int(sys.argv[2])
    end_bar = int(sys.argv[3])
    fs.midi_to_audio(midi_file_path, 'before.wav')
    process_midi_file(midi_file_path, start_bar, end_bar)
