VisionID 🖼️🔊

VisionID is a simple face recognition and verification system built with OpenCV, Tkinter, and pyttsx3.
It allows you to add new face images and verify identities using the LBPH Face Recognizer.

✨ Features
📸 Add New Images: Capture face samples from your webcam and save them for training.

🧠 Train Recognizer: Uses LBPH (Local Binary Patterns Histograms) for robust face recognition.

🔍 Face Verification: Detects and verifies faces in real-time.

🗣️ Voice Feedback: Provides audio prompts using pyttsx3.

🖥️ GUI Interface: Simple Tkinter-based interface with buttons for scanning and adding images.

📂 Project Structure
Code
VisionID/
│── VisionID.py        # Main application script
│── Faces/             # Folder where captured face images are stored
⚙️ Requirements
Install the following dependencies before running:

bash
pip install opencv-python opencv-contrib-python numpy pyttsx3
🚀 Usage
Run the application:

bash
python VisionID.py
Add New Image:

Click ADD NEW IMAGE.

Enter a name in the text box.

The system captures 10 face samples and saves them in the Faces/ directory.

Scan & Verify:

Click SCAN.

The system trains the recognizer with available images.

If a face is recognized, it displays the name and announces "Image verified".

If not recognized, it labels the face as Unknown.

🛠️ How It Works
Face Detection: Uses Haar Cascade (haarcascade_frontalface_default.xml).

Face Recognition: LBPH algorithm (cv2.face.LBPHFaceRecognizer_create()).

Training Data: Images stored in Faces/ are used to train the recognizer.

Confidence Threshold: Faces with confidence < 80 are considered verified.
