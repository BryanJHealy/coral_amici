# coral_amici

---
AI Music Iterative Composer Interface for Google Coral Dev Board
 
## Directory Structure   

---
- [resources](resources "resources directory"): directory containing resources for README  
    - [img](resources/img "img directory"): directory containing images for README  
- [src](src "src directory"): directory containing all library files and scripts  
    - [examples](src/examples "examples directory"): directory containing scripts to demonstrate library functionality
        - [train_lstm_accompaniment_model.py](src/examples/train_lstm_accompaniment_model.py ""): builds, trains, saves, and evaluates a LstmAccompaniment model
    - [models](src/models "models directory"): directory containing model library files
        - [](src/models/ ""):
        - [](src/models/ ""):
        - [](src/models/ ""):
        - [](src/models/ ""):
        - [](src/models/ ""):
    - [util](src/util "util directory"): directory containing miscellaneous files for data handling, data visualization, inference, etc...
 
- TFlite quant and optim
- coral conversion and inference
- SW design
- dataset selection
- data handling and visualization
    - don't add data if either window is empty
## Feature Representation
    - flexibility
    - non-varying I/O note densities
    - sample rate/combining 
- model training and exploration
    - preliminary results
    
    ![LSTM Gen1](resources/img/LSTM_gen1.png?raw=true "Title")
- varying lengths via quantization/sampling
- chords (multiple simultaneous notes)
#Demo