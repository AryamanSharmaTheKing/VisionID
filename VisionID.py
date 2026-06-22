import cv2
import os
import numpy as np
import pyttsx3
import tkinter as tk
from tkinter import messagebox

root = tk.Tk()
root.title("VisionID")
engine = pyttsx3.init()
path = "Faces"

if not os.path.exists(path):
    os.makedirs(path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples, ids, names = [], [], {}
    current_id = 0
    for image_path in image_paths:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        label = os.path.splitext(os.path.basename(image_path))[0].split("_")[0] 
        if label not in names:
            current_id += 1
            names[label] = current_id
        id = names[label]
        faces = face_cascade.detectMultiScale(img, 1.3, 5)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(id)
    return face_samples, ids, names

faces, ids, names = get_images_and_labels(path)
if len(faces) > 1:
    recognizer.train(faces, np.array(ids))
else:
    engine.say("Not enough training data! Please add more images.")
    engine.runAndWait()

def on_click():
    entry = tk.Entry(root)
    entry.pack()
    def save_image():
        name_text = entry.get().strip()
        cap = cv2.VideoCapture(0)
        if name_text:
            count = 0
            while count < 10:
                ret, frame = cap.read()
                if not ret:
                    print("Camera did not return a frame")
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=30)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    filename = os.path.join(path, f"{name_text}_{count+1}.jpg")
                    cv2.imwrite(filename, gray[y:y+h, x:x+w])
                    print(f"Image saved as {filename}")
                    count += 1
                    break
                cv2.waitKey(1000)  
        else:
            print("No name entered")
        cap.release()
        cv2.destroyAllWindows()
        entry.destroy()
        save_button.destroy()
    save_button = tk.Button(root, text="Save", command=save_image)
    save_button.pack()

def on_click_1():
    faces, ids, names = get_images_and_labels(path)
    if len(faces) > 1:
        recognizer.train(faces, np.array(ids))
    else:
        label = tk.Label(root, text="Not enough training data!")
        label.pack()
        return
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=30)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 80:
                name = [k for k,v in names.items() if v == id_][0]
                engine.say("Image verified")
                messagebox.showinfo("", f"Name: {name}")
                engine.runAndWait()
            else:
                name = "Unknown"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow("Face Recognition", frame)
        cv2.waitKey(0)
    else:
        engine.say("Camera did not return a frame")
        engine.runAndWait()
    cap.release()
    cv2.destroyAllWindows()

tk.Button(text="SCAN", command=on_click_1).pack()
tk.Button(text="ADD NEW IMAGE", command=on_click).pack()
root.mainloop()
