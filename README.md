# coral_amici

---
AI Music Iterative Composer Interface for Google Coral Dev Board
 
## Directory Structure
- [resources](resources "resources directory"): directory containing resources for README
    - [doc](resources/doc "doc directory"): directory containing supplemental documents
    - [img](resources/img "img directory"): directory containing images for README  
- [src](src "src directory"): directory containing all library files and scripts  
    - [examples](src/examples "examples directory"): directory containing scripts to demonstrate library functionality
        - [***train_lstm_accompaniment_model.py***](src/examples/train_lstm_accompaniment_model.py "train_lstm_accompaniment_model.py")*: builds, trains, saves, and evaluates a LstmAccompaniment model
    - [models](src/models "models directory"): directory containing model library files
        - [basic.py](src/models/basic.py "basic.py"): very basic model to test conversion to TFLite for Coral Dev Board
        - [***lstm_accompaniment.py***](src/models/lstm_accompaniment.py "lstm_accompaniment.py")*: implements the main accompaniment model
        - [lstm_autoencoder.py](src/models/lstm_autoencoder.py "lstm_autoencoder.py"): basic LSTM autoencoder designed to replicate a given MIDI track
    - [util](src/util "util directory"): directory containing miscellaneous files for data handling, data visualization, inference, etc...
        - [basic_inference_test.py](src/util/basic_inference_test.py "basic_inference_test.py"): runs inference of a converted model from basic.py on the Coral
        - [***coral_converter.py***](src/util/coral_converter.py "coral_converter.py")*: serializes the given model, then converts it TFLite format using the Coral delegate
        - [***data_handling.py***](src/util/data_handling.py "data_handling.py")*: creates dataset from POP909 data for accompaniment generation
        - [midi_test.py](src/util/midi_test.py "midi_test.py"): tests various functions of the pretty_midi library. Trains, saves, and converts a test model to TFLite
        - [pop_visualization.py](src/util/pop_visualization.py "pop_visualization.py"): various functions to visualize the POP909 data
        - [synth_test.py](src/util/synth_test.py "synth_test.py"): tests the pyfluidsynth library
 
\* : indicates a file directly involved in this project's research that will most-likely be used in the final interface

# Edge ML with the Google Coral Dev Board

---
## Overview
- Capabilities with Edge TPU
## Constraints 
- Tensorflow Lite format
- Limited operation compatibility
- Quantization-Aware Training still in development
## Procedure
- Model creation and training
- Quantization and Optimization
- Conversion to TFLite format using Coral delegate
- Inference on the Coral
## Considerations
- While any model in a quantized TFLite format can run on the Coral, only the layers with supported ops up to the first 
layer with an unsupported operation will be run on the TPU. The subsequent layers will be run on the board's CPU. Thus, 
considerations should be made regarding the performance tradeoffs between running a larger model composed of simple
operations on the TPU vs running a smaller model with more advanced operation on the CPU.
- More research to determine if the model can be split up to maximize the amount of TPU processing.  
- Available libraries and development has been focused on computer vision applications
### Comparison with other edge ML platforms  
- Platform / Application suitability
    - GPU-based platforms (Nvidia Jetson, ...) -> much more versatile
- Performance on Coral-supported applications (MobileNet, ...)
- Ability to train on the edge device currently only supported by GPU/CPU 
- Ability to reuse TFLite models on mobile

# AI Music Iterative Composer Interface (AMICI)

---
## Background
- Provides a tool/interface to compose and perform music alongside AI and iteratively improve upon the generated sequences
## UI design flow
- Live MIDI instrument recording
- Sequence selection
- Accompaniment generation
- Iterative improvement
- Voicing selection
- Arrangement
- Playback
## SW architecture
- Accompaniment model
- Iteration model
- DDSP model server for MIDI-to-WAV
- Touchscreen
- Interface framework/language selection
## Accompaniment Generation

---
### Dataset Selection
- POP909 Dataset contains multi-track MIDI files labeled with melodies, accompaniments, key changes, ...
- Tensorflow Dataset creation
- Data handling and visualization
    - Process songs separately to avoid combining the end/start of two into a sequence
    - Don't add data if either window is empty
### Feature Representation
- Flexibility
    - Non-varying i/o note densities
    - Varying lengths via quantization/sampling
    - Multiple simultaneous notes (chords)
- Sample rate/combining 
### Model Training and Exploration
- Generating melody by sequentially predicting next note
    ![LSTM Gen1](resources/img/LSTM_gen1.png?raw=true "Title")
- LSTM autoencoder
    - Training (epochs, #files, batch size, learning rate, ...)
    - 99.5% training accuracy in recreating a melody
    - Evaluation results
    - Demonstrates that a simple LSTM layer is capable of learning how to represent a 25 note sequence
- LSTM-based Encoder-Decoder
    - Training hyperparameters
    - Transfer learning vs retraining decoder for accompaniment
        - Freezing encoder weights
    - Training results
    - Select generated accompaniments
### Inference Evaluation
- Conversion
- Sample melody
- Performance results
    - Memory usage
    - Inference latency
### Demo Samples
# References
1. [https://www.tensorflow.org/tutorials/audio/music_generation](https://www.tensorflow.org/tutorials/audio/music_generation)
2. [https://github.com/craffel/pretty-midi](https://github.com/craffel/pretty-midi)
3. [https://magenta.tensorflow.org/music-vae](https://magenta.tensorflow.org/music-vae)
4. [https://arxiv.org/pdf/1803.05428.pdf](https://arxiv.org/pdf/1803.05428.pdf)
5. [https://github.com/music-x-lab/POP909-Dataset](https://github.com/music-x-lab/POP909-Dataset)
6. [https://machinelearningmastery.com/lstm-autoencoders/](https://machinelearningmastery.com/lstm-autoencoders/)
