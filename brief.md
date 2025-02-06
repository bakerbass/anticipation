# Project Brief
midi_processor/
├── main.py                   # Entry point of the application
├── watcher.py                # Directory watcher logic
├── processor.py              # Functions for model processing
├── synth.py                  # FluidSynth initialization and playback
├── audio_utils.py            # Audio conversion, normalization, and plotting
├── models/
│   └── model_loader.py       # Model loading logic
├── tests/                    # Tests for individual modules
└── data/
    ├── input/                # Directory to watch for MIDI files
    ├── processed/            # Processed MIDI files
    └── audio/                # Generated audio files

a) main.py
Purpose: Acts as the controller to tie everything together. Instantiates the watcher, loads the model, initializes FluidSynth, and orchestrates the processing pipeline.
b) watcher.py
Responsibility: Watches a specific directory for new MIDI files.
Implementation: Uses a library like watchdog to monitor the directory and trigger events when a file is added.
c) processor.py
Responsibility: Contains the main logic for handling MIDI files and processing them with the pre-loaded model.
Structure:
def process_midi(midi_path, model):
Any transformations or processing that need to be applied.
d) synth.py
Responsibility: Manages FluidSynth initialization and MIDI playback.
Structure:
def initialize_fluidsynth() -> FluidSynth:
def play_midi(midi_path, fluidsynth):
e) audio_utils.py
Responsibility: Handles audio conversion, normalization, and plotting.
Functions:
def convert_midi_to_audio(midi_path, fluidsynth):
def normalize_audio(audio_path):
def plot_waveforms(audio_path_1, audio_path_2):
f) models/model_loader.py
Responsibility: Encapsulates model initialization logic.
Structure:
def load_model(model_path):
g) tests/
Add unit tests for:
Watching directories
Processing MIDI files
Synth playback
Audio utilities
3. Workflow
Initialize Environment:

Load the model via models/model_loader.py.
Initialize FluidSynth via synth.py.
Start Watching Directory:

Use watcher.py to monitor data/input/ for new files.
Process MIDI:

On detecting a new MIDI file:
Call processor.process_midi() with the file and model.
Pre-process and process MIDI for comparison.
Convert to Audio:

Use audio_utils.convert_midi_to_audio() to convert both versions.
Normalize and Plot:

Normalize audio via audio_utils.normalize_audio().
Compare waveforms using audio_utils.plot_waveforms().
Playback (Optional for Testing):

Play MIDI files using synth.play_midi().