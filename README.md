# Voice-Recognition-System

### Overview
The Voice Recognition System is a Python-based project designed to recognize speakers and convert audio recordings to text. It integrates audio processing, machine learning, and a graphical user interface (GUI) using the Tkinter library. The system employs features such as MFCC extraction and GMMs for speaker identification.

### Project Structure

Initialization and GUI Setup :
The code initializes a Tkinter application, creating a user-friendly interface with buttons for critical functionalities.

#### Model Training :
The "Train Model" button triggers model training, including speaker name writing, audio file reading, and GMM training.

#### Recording for Testing:
The "Record for Test" button initiates audio recording for a specified duration, allowing users to test the system.

#### Testing the Model:
The "Test Model" button applies trained GMM models to recognize speakers in selected test files.

#### File Handling :
Functions for writing speaker names to text files and selecting test files for evaluation.

#### Audio Processing and Feature Extraction:
Comprehensive feature extraction using MFCC for capturing relevant audio characteristics.

#### Speech to Text Conversion :
Utilizes Google's speech recognition service for converting recorded audio to text.

#### Displaying Results in GUI:
Results of speaker recognition and recognized text are displayed in Tkinter GUI text widgets.


### Output of Project:
![Screenshot (314)](https://github.com/maham1033/Voice-Recognition-System/assets/109579257/9eae5a45-a05e-4a95-876d-d9674c216657)

![Screenshot (315)](https://github.com/maham1033/Voice-Recognition-System/assets/109579257/acd07f0d-4a96-4dc7-9523-04532774d067)


