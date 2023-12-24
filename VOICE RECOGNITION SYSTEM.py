"""
Created on  25th November 2023  18:56:34

                    Project Title:
                           Voice Recognition System

                    Submitted by:

                    Ebaa Haq              2021-CE-22
                    Faiza Riaz            2021-CE-20
                    Maham Nadeem          2021-CE-10

                    Submitted to:
                           Raja Muzammil Munir

                    Course:
                           CMPE-341L Artificial Intelligence

                    Semester:
                           Fall 2023 (5th)


"""
# import libraries
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from tkinter import messagebox
import os
import time
import pickle
import numpy as np
from scipy.io.wavfile import read, write
import sounddevice as sd
import speech_recognition as sr
import python_speech_features as mfcc
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

# Define models and speakers globally
models = []
speakers = []

class VoiceRecognitionApp:
    def __init__(self, root):
        # Initialize the GUI window
        self.root = root
        self.root.title("Voice Recognition System")
        self.root.geometry("600x500")  # Set the window size

        # Configure a style for colored buttons
        self.style = ttk.Style()
        self.style.configure("TButton", background="#4CAF50", foreground="black")

        # Buttons for training, recording, and testing
        self.train_button = ttk.Button(root, text="Train Model", command=self.train_model)
        self.train_button.pack(pady=10)

        self.record_button = ttk.Button(root, text="Record for Test", command=self.record_for_test)
        self.record_button.pack(pady=10)

        # Entry widget for user to enter the name of the recorded audio file
        self.recorded_audio_name_var = tk.StringVar()
        self.recorded_audio_name_entry = ttk.Entry(root, textvariable=self.recorded_audio_name_var)
        self.recorded_audio_name_entry.pack(pady=5, padx=10)

        # Button to browse for a test audio file
        self.browse_button = ttk.Button(root, text="Browse", command=self.browse_test_file)
        self.browse_button.pack(pady=5)

        # Entry widget to display the selected test file path
        self.test_file_path_var = tk.StringVar()
        self.test_file_entry = ttk.Entry(root, textvariable=self.test_file_path_var, state="readonly")
        self.test_file_entry.pack(pady=10)

        # Text widget with vertical scrollbar for displaying detected speaker
        self.detected_speaker_frame = ttk.Frame(root)
        self.detected_speaker_frame.pack(pady=10)

        self.detected_speaker_scrollbar = ttk.Scrollbar(self.detected_speaker_frame, orient="vertical")
        self.detected_speaker_scrollbar.pack(side="right", fill="y")

        self.detected_speaker = tk.Text(self.detected_speaker_frame, height=1, width=50,
                                        yscrollcommand=self.detected_speaker_scrollbar.set)
        self.detected_speaker.pack(side="left")

        self.detected_speaker_scrollbar.config(command=self.detected_speaker.yview)

        # Text box with vertical scrollbar for displaying recognized text
        self.text_display_frame = ttk.Frame(root)
        self.text_display_frame.pack(pady=10)

        self.text_display_scrollbar = ttk.Scrollbar(self.text_display_frame, orient="vertical")
        self.text_display_scrollbar.pack(side="right", fill="y")

        self.text_display = tk.Text(self.text_display_frame, height=4, width=50,
                                    yscrollcommand=self.text_display_scrollbar.set)
        self.text_display.pack(side="left")

        self.text_display_scrollbar.config(command=self.text_display.yview)

        # Button to test the model
        self.test_button = ttk.Button(root, text="Test Model", command=self.test_model, style="TButton")
        self.test_button.pack(pady=10)
    def train_model(self):
        # Train the voice recognition model
        write_names()
        train_model()
        messagebox.showinfo("Training Completed", "Model training completed successfully.")

    def record_for_test(self):
        # Record audio for testing
        self.record_button.config(state="disabled")
        self.test_button.config(state="disabled")

        # Set the sampling rate and duration for the recording
        fs = 44100
        duration = 10
        print("Recording Started...")

        # Initialize a global variable to store the recorded audio
        global test_audio

        # Start recording using the sounddevice library
        test_audio = sd.rec(frames=duration * fs, samplerate=fs, channels=2)
        sd.wait()  # Wait for the recording to finish
        print("Recording Ended...")

        # Get the entered name from the Entry widget
        audio_name = self.recorded_audio_name_var.get()

        # Check if the name is provided, otherwise use a default name
        if not audio_name:
            audio_name = "default_recording"

        # Create the 'recorded_audio' folder if it does not exist
        audio_folder = "recorded_audio"
        if not os.path.exists(audio_folder):
            os.makedirs(audio_folder)

        # Save the recorded audio with the entered name
        audio_path = os.path.join(audio_folder, f"{audio_name}.wav")
        write(audio_path, 44100, test_audio)

        # Re-enable buttons for future actions
        self.record_button.config(state="normal")
        self.test_button.config(state="normal")

    def browse_test_file(self):
        # Browse for a test audio file
        file_path = filedialog.askopenfilename(filetypes=[('WAV Files', '*.wav')])

        # Check if a file path is selected
        if file_path:
            # Set the selected file path to the corresponding Entry widget variable
            self.test_file_path_var.set(file_path)

    def test_model(self):
        # Test the trained models on the selected test audio file
        file_path = self.test_file_path_var.get()

        # Check if a test file is selected
        if not file_path:
            messagebox.showerror("Error", "Please select a test file.")
            return

        try:
            # Read the test audio file
            sr, audio = read(file_path)
            vector = extract_features(audio, sr)

            # Initialize an array for log likelihood scores
            log_likelihood = np.zeros(len(models))

            # Calculate the log likelihood for each model
            for i in range(len(models)):
                gmm = models[i]
                scores = np.array(gmm.score(vector))
                log_likelihood[i] = scores.sum()

            # Select the speaker with the highest likelihood
            winner = np.argmax(log_likelihood)
            recognized_speaker = speakers[winner]
            print("\n>> detected as - ", recognized_speaker)

            # Display the detected speaker in the text widget
            self.detected_speaker.delete(1.0, tk.END)  # Clear previous text
            self.detected_speaker.insert(tk.END, f"Detected as: {recognized_speaker}")

            # Convert audio to text using Google Speech Recognition
            text = convert_voice_to_text(file_path)
            print(f"\n>> Recognized text: {text}")

            # Display the recognized text in the text widget
            self.text_display.delete(1.0, tk.END)  # Clear previous text
            self.text_display.insert(tk.END, f"Recognized text: {text}")

            # Pause for a moment to display results
            time.sleep(1.0)

        except Exception as e:
            # Handle exceptions and show an error message
            messagebox.showerror("Error", f"An error occurred during testing: {str(e)}")


def write_names():
    # Write names of speakers in the training set to a text file
    source_dir_train = "./training_set/"
    train_file = "./training_set_addition.txt"

    # Open the text file for writing
    with open(train_file, "w") as file:
        # Iterate through the list of files in the training set directory
        for filename in os.listdir(source_dir_train):
            # Write the name of each speaker to the text file followed by a newline character
            file.writelines(filename + '\n')

    # Write names of speakers in the testing set to a text file
    source_dir_test = "./testing_set/"
    test_file = "./testing_set_addition.txt"

    # Open the text file for writing
    with open(test_file, "w") as file:
        # Iterate through the list of files in the testing set directory
        for filename in os.listdir(source_dir_test):
            # Write the name of each speaker to the text file followed by a newline character
            file.writelines(filename + '\n')


def calculate_delta(array):
    # Calculate delta coefficients from the MFCC features
    rows, cols = array.shape

    # Initialize an array to store the delta coefficients
    deltas = np.zeros((rows, 20))

    # Set the window size for delta calculation
    N = 2

    # Iterate through the rows (frames) of the MFCC features
    for i in range(rows):
        index = []
        j = 1

        # Create indices for the neighboring frames within the window
        while j <= N:
            if i - j < 0:
                first = 0
            else:
                first = i - j
            if i + j > rows - 1:
                second = rows - 1
            else:
                second = i + j
            index.append((second, first))
            j += 1

        # Calculate the delta coefficients using the neighboring frames
        deltas[i] = (array[index[0][0]] - array[index[0][1]] + (2 * (array[index[1][0]] - array[index[1][1]]))) / 10

    return deltas


def extract_features(audio, rate):
    # Extract MFCC features from audio
    mfcc_feature = mfcc.mfcc(audio, rate, winlen=0.025, winstep=0.01, numcep=20, nfft=1200, appendEnergy=True)

    # Scale the MFCC features
    mfcc_feature = preprocessing.scale(mfcc_feature)

    # Calculate delta coefficients using the previously defined calculate_delta function
    delta = calculate_delta(mfcc_feature)

    # Combine the original MFCC features and delta coefficients
    combined = np.hstack((mfcc_feature, delta))

    return combined


def train_model():
    # Train the voice recognition model using Gaussian Mixture Models (GMM)

    # Step 1: Write Names
    write_names()

    # Step 2: Set Source and Destination Directories
    source = "./training_set/"
    dest = "./trained_models/"

    # Step 3: Specify the Training File
    train_file = "./training_set_addition.txt"

    # Step 4: Open the Training File
    file_paths = open(train_file, 'r')

    # Step 5: Initialize Count and Features
    count = 1
    features = np.asarray(())

    # Step 6: Iterate through Paths in the Training File
    for path in file_paths:
        # Clean up path by removing leading/trailing whitespaces
        path = path.strip()

        # Read the audio file
        sr, audio = read(source + path)

        # Extract features using the extract_features function
        vector = extract_features(audio, sr)

        # Step 7: Build Feature Matrix
        if features.size == 0:
            features = vector
        else:
            features = np.vstack((features, vector))

        # Step 8: Train GMM for Each Speaker
        if count == 1:
            gmm = GaussianMixture(n_components=7, max_iter=200, covariance_type='diag', n_init=3)
            gmm.fit(features)

            # Save the trained model using pickle
            picklefile = path.split("-")[0] + ".gmm"
            pickle.dump(gmm, open(dest + picklefile, 'wb'))

            # Print information about the completed modeling
            print('>> Modeling completed for speaker:', picklefile, " with data point = ", features.shape)

            # Reset features for the next speaker
            features = np.asarray(())
            count = 0

        # Increment count for each speaker
        count = count + 1


def convert_voice_to_text(audio_path):
    # Convert audio to text using Google Speech Recognition
    recognizer = sr.Recognizer()

    # Open the audio file using the AudioFile class from SpeechRecognition library
    with sr.AudioFile(audio_path) as source_audio:
        # Record the audio from the file
        audio_data = recognizer.record(source_audio)

    try:
        # Attempt to recognize the speech using Google Speech Recognition
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        # Handle the case where speech cannot be understood
        print("Speech Recognition could not understand audio")
        return ""
    except sr.RequestError as e:
        # Handle the case where a request error occurs (e.g., no internet connection)
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return ""


if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()

    # Instantiate the VoiceRecognitionApp class with the main window as its parent
    app = VoiceRecognitionApp(root)

    # Load the pre-trained models and corresponding speakers
    modelpath = "./trained_models/"
    gmm_files = [os.path.join(modelpath, fname) for fname in os.listdir(modelpath) if fname.endswith('.gmm')]

    # Load the GMM models using pickle and store them in the 'models' list
    models = [pickle.load(open(fname, 'rb')) for fname in gmm_files]

    # Extract speaker names from file paths and store them in the 'speakers' list
    speakers = [fname.split("\\")[-1].split(".gmm")[0] for fname in gmm_files]

    # Start the Tkinter event loop
    root.mainloop()

