import sys,time
import midi2audio
import transformers
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from transformers import AutoModelForCausalLM

from IPython.display import Audio

from anticipation import ops
from anticipation.sample import generate
from anticipation.tokenize import extract_instruments
from anticipation.convert import events_to_midi,midi_to_events
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
fs = midi2audio.FluidSynth('./sf.sf2')

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

def plot_wav(file_path):
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
    plt.show()

start = time.time()
events = midi_to_events('./midi/Marley, Bob - No Woman No Cry (arranged for one acoustic guitar).mid')
end = time.time()
print("MIDI converted in ", end-start, " seconds")
#Audio(synthesize(fs, ops.clip(events, 0, 30)))
segment = ops.clip(events, 0, 48)
segment = ops.translate(segment, -ops.min_time(segment, seconds=False))

#Audio(synthesize(fs, segment))

history = ops.clip(segment, 0, 24, clip_duration=False)
anticipated = [CONTROL_OFFSET + tok for tok in ops.clip(segment, 25, 35, clip_duration=False)]
#Audio(synthesize(fs, ops.combine(history, anticipated)))
start = time.time()
inpainted = generate(model, 24, 35, inputs=history, controls=anticipated, top_p=.95)
end = time.time()
print("Generated in ", end-start, " seconds")
out = synthesize(fs, ops.combine(inpainted, anticipated))
out = normalize_wav(out)
plot_wav(out)
Audio(out)
