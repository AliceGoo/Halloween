import cv2
import dlib
import numpy as np
import random
import time
import math
import os
import pyttsx3  # For text-to-speech
import speech_recognition as sr  # For voice recognition
from pygame import mixer  # For playing scary music
from threading import Thread
from PIL import ImageFont, ImageDraw, Image, ImageSequence  # For Gothic font rendering

# Define the base directory where the images are stored
base_path = '/Users/lovelyyuer/Documents/0.QuincyCollege/Fall/halloweenProject'

# Initialize Pygame Mixer for music playback
mixer.init()

# Voice Recognition and Text-to-Speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('/Users/lovelyyuer/Documents/0.QuincyCollege/Fall/halloweenProject/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat')

# Define the paths to individual image files using the base path
horns_path = os.path.join(base_path, 'horns.png')
witch_hat_path = os.path.join(base_path, 'witch_hat.png')
pumpkin_path = os.path.join(base_path, 'pumpkin.png')
skeleton_path = os.path.join(base_path, 'skeleton.png')
jason_path = os.path.join(base_path, 'jason_vorhees.png')
future_head_path = os.path.join(base_path, 'future_head.gif')

# Load the Halloween-themed decoration images with transparency (RGBA)
horns_img = cv2.imread(horns_path, cv2.IMREAD_UNCHANGED)
witch_hat_img = cv2.imread(witch_hat_path, cv2.IMREAD_UNCHANGED)
skeleton_img = cv2.imread(skeleton_path, cv2.IMREAD_UNCHANGED) 
pumpkin_img = cv2.imread(pumpkin_path, cv2.IMREAD_UNCHANGED) 
jason_img = cv2.imread(jason_path, cv2.IMREAD_UNCHANGED)

# Load Gothic Font (TrueType Font)
gothic_font_path = '/Users/lovelyyuer/Documents/0.QuincyCollege/Fall/halloweenProject//Blackletter.ttf'  # Replace with the path to your font
font = ImageFont.truetype(gothic_font_path, 70)  # Load the Gothic font with size 70

# Global state variables
fortune_active = False
fortune_start_time = 0
fortune_duration = 10  # Show the fortune for 10 seconds

# Function to make pumpkins jump in a bumpy way and wiggle
def animated_pumpkin_position(base_pos, frame_width, frame_height):
    current_time = time.time()
    
    # Vertical jumping with a bumpy effect (sudden changes in position)
    jump_amount = (math.sin(current_time * 3 * math.pi / 5) ** 3) * (frame_height // 30)
    
    # Horizontal wiggling every 8 seconds
    wiggle_amount = math.sin(current_time * 2 * math.pi / 8) * (frame_width // 50)
    
    # Adjust base position with jump and wiggle
    x = base_pos[0] + int(wiggle_amount)
    y = base_pos[1] + int(jump_amount)
    
    return x, y

# Function to resize the overlay image while maintaining the aspect ratio
def resize_overlay_image(image, target_width):
    aspect_ratio = image.shape[0] / image.shape[1]  # Height / Width
    target_height = int(aspect_ratio * target_width)
    return cv2.resize(image, (target_width, target_height))

# Function to overlay transparent image over another image (with alpha channel)
def overlay_image_alpha(img, img_overlay, pos, alpha=1.0):
    x, y = pos
    h, w = img_overlay.shape[0], img_overlay.shape[1]

    # Ensure the overlay fits within the frame's dimensions
    if x < 0:
        w += x  # Reduce width to fit the image
        img_overlay = img_overlay[:, -x:]  # Clip the overlay
        x = 0  # Reset the x position
    if y < 0:
        h += y  # Reduce height to fit the image
        img_overlay = img_overlay[-y:, :]  # Clip the overlay
        y = 0  # Reset the y position
    if x + w > img.shape[1]:
        w = img.shape[1] - x
        img_overlay = img_overlay[:, :w]
    if y + h > img.shape[0]:
        h = img.shape[0] - y
        img_overlay = img_overlay[:h, :]

    # Check if the overlay still has valid dimensions after clipping
    if h <= 0 or w <= 0:
        return img  # If the overlay is fully out of bounds, skip drawing

    # Extract the region of interest (ROI) from the background image
    roi = img[y:y+h, x:x+w]

    # Extract the alpha channel from the overlay
    img_overlay = img_overlay[:h, :w]
    alpha_mask = (img_overlay[:, :, 3] / 255.0) * alpha  # Normalize alpha channel to range 0-1 and apply alpha factor

    # Blend the overlay image and the background image using the alpha mask
    for c in range(0, 3):
        roi[:, :, c] = roi[:, :, c] * (1 - alpha_mask) + img_overlay[:, :, c] * alpha_mask

    img[y:y+h, x:x+w] = roi
    return img

# Function to morph a person's face (simple warping effect)
def morph_face(frame, landmarks):
    # Define a basic face warping by stretching around the nose and eyes
    center_x = (landmarks.part(36).x + landmarks.part(45).x) // 2  # Between the eyes
    center_y = (landmarks.part(27).y)  # Around the nose bridge

    morph_radius = 50  # Radius for the morph effect
    frame = cv2.circle(frame, (center_x, center_y), morph_radius, (0, 255, 0), 3)  # Example: green circle for effect

    # We can expand this function to create more sophisticated morphs, like stretching or distortion.

    return frame

# Function to place decorations based on facial landmarks and number of people
def place_decorations(frame, landmarks_list):
    num_people = len(landmarks_list)
    
    if num_people == 1:
        # Single person gets the witch hat
        landmarks = landmarks_list[0]
        face_width = int((landmarks.part(16).x - landmarks.part(0).x) * 1.2)
        witch_hat_resized = resize_overlay_image(witch_hat_img, face_width)
        witch_hat_pos = (landmarks.part(27).x - witch_hat_resized.shape[1] // 2, landmarks.part(19).y - witch_hat_resized.shape[0] - 20)
        frame = overlay_image_alpha(frame, witch_hat_resized, witch_hat_pos)
        
    elif num_people == 2:
        # First person gets witch hat, second person gets horns
        landmarks_left = landmarks_list[0]
        landmarks_right = landmarks_list[1]
        
        face_width_left = int((landmarks_left.part(16).x - landmarks_left.part(0).x) * 1.2)
        face_width_right = int((landmarks_right.part(16).x - landmarks_right.part(0).x) * 1.2)
        
        # Witch Hat for Left Person
        witch_hat_resized = resize_overlay_image(witch_hat_img, face_width_left)
        witch_hat_pos = (landmarks_left.part(27).x - witch_hat_resized.shape[1] // 2, landmarks_left.part(19).y - witch_hat_resized.shape[0] - 20)
        frame = overlay_image_alpha(frame, witch_hat_resized, witch_hat_pos)
        
        # Horns for Right Person
        horns_resized = resize_overlay_image(horns_img, face_width_right)
        horns_pos = (landmarks_right.part(27).x - horns_resized.shape[1] // 2, landmarks_right.part(24).y - horns_resized.shape[0])
        frame = overlay_image_alpha(frame, horns_resized, horns_pos)

    elif num_people >= 3:
        # Distribute hats, horns, Jason mask, and skeleton mask
        landmarks_left = landmarks_list[0]
        landmarks_middle = landmarks_list[1]
        landmarks_right = landmarks_list[2]
        
        face_width_left = int((landmarks_left.part(16).x - landmarks_left.part(0).x) * 1.2)
        face_width_middle = int((landmarks_middle.part(16).x - landmarks_middle.part(0).x) * 1.2)
        face_width_right = int((landmarks_right.part(16).x - landmarks_right.part(0).x) * 1.2)
        
        # Left Person - Horns
        horns_resized = resize_overlay_image(horns_img, face_width_left)
        horns_pos = (landmarks_left.part(27).x - horns_resized.shape[1] // 2, landmarks_left.part(24).y - horns_resized.shape[0])
        frame = overlay_image_alpha(frame, horns_resized, horns_pos)
        
        # Middle Person - Witch Hat
        witch_hat_resized = resize_overlay_image(witch_hat_img, face_width_middle)
        witch_hat_pos = (landmarks_middle.part(27).x - witch_hat_resized.shape[1] // 2, landmarks_middle.part(19).y - witch_hat_resized.shape[0] - 20)
        frame = overlay_image_alpha(frame, witch_hat_resized, witch_hat_pos)
        
        # Right Person - Jason Mask
        jason_resized = resize_overlay_image(jason_img, face_width_right)
        jason_pos = (landmarks_right.part(27).x - jason_resized.shape[1] // 2, landmarks_right.part(24).y - jason_resized.shape[0] // 2)
        frame = overlay_image_alpha(frame, jason_resized, jason_pos)
        
        # Add the Skeleton mask for additional people if num_people > 3
        if num_people > 3:
            for i in range(3, num_people):
                landmarks_extra = landmarks_list[i]
                face_width_extra = int((landmarks_extra.part(16).x - landmarks_extra.part(0).x) * 1.2)
                skeleton_resized = resize_overlay_image(skeleton_img, face_width_extra)
                skeleton_pos = (landmarks_extra.part(27).x - skeleton_resized.shape[1] // 2, landmarks_extra.part(27).y - skeleton_resized.shape[0] // 2)
                frame = overlay_image_alpha(frame, skeleton_resized, skeleton_pos)

    return frame

# Add pumpkins to the four corners of the frame with bumpy jumping and wiggling effects
def add_pumpkins_in_corners(frame):
    pumpkin_resized = resize_overlay_image(pumpkin_img, 320)  # Resize pumpkin to 4x larger
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Top-left corner
    pos = animated_pumpkin_position((10, 10), frame_width, frame_height)
    frame = overlay_image_alpha(frame, pumpkin_resized, pos)

    # Top-right corner
    pos = animated_pumpkin_position((frame.shape[1] - pumpkin_resized.shape[1] - 10, 10), frame_width, frame_height)
    frame = overlay_image_alpha(frame, pumpkin_resized, pos)

    # Bottom-left corner
    pos = animated_pumpkin_position((10, frame.shape[0] - pumpkin_resized.shape[0] - 10), frame_width, frame_height)
    frame = overlay_image_alpha(frame, pumpkin_resized, pos)

    # Bottom-right corner
    pos = animated_pumpkin_position((frame.shape[1] - pumpkin_resized.shape[1] - 10, frame.shape[0] - pumpkin_resized.shape[0] - 10), frame_width, frame_height)
    frame = overlay_image_alpha(frame, pumpkin_resized, pos)

    return frame

# Function to display slogans or jokes with animations
def display_slogans_and_jokes(frame):
    # Skip if the fortune is active
    if fortune_active:
        return frame

    slogans = [
        "Eat, drink, and be scary!",
        "Wishing you a spooky, kooky, and happily haunted Halloween!",
        "Here's to a spooky season filled with tricks, treats, and lots of laughs!",
        "Hope your Halloween is full of chills, thrills, and candy spills!"
    ]

    jokes = [
        ("Q: Why don't mummies take vacations?", "A: They are afraid they'll relax and unwind!"),
        ("Q: What's a ghost's favorite dessert?", "A: I scream!"),
        ("Q: Why did the skeleton stay home from the party?", "A: He had no body to go with!"),
        ("Q: How do vampires start their letters?", "A: Tomb it may concern..."),
        ("Q: Why don't skeletons fight each other?", "A: They don't have the guts!")
    ]
    
    current_time = int(time.time())
    cycle_time = current_time // 8  # Change slogan or joke every 8 seconds

    font = cv2.FONT_HERSHEY_SIMPLEX
    frame_height, frame_width = frame.shape[:2]

    # First half of the cycle: Display slogans
    if cycle_time % 2 == 0:
        slogan_index = cycle_time % len(slogans)
        slogan = slogans[slogan_index]

        # Slide in and explode out effect
        time_in_cycle = current_time % 8
        if time_in_cycle < 4:
            # Slide in from the left
            x_pos = int((time_in_cycle / 4.0) * frame_width) - frame_width
        else:
            # Explode out
            x_pos = int(frame_width * 1.5 * (time_in_cycle - 4) / 4)

        # Display the slogan in the middle and closer to the top with 80pt font
        text_size = cv2.getTextSize(slogan, font, 2.2, 5)[0]  # 80pt = scale 2.2 in OpenCV
        text_x = (frame_width - text_size[0]) // 2 + x_pos
        text_y = int(frame_height * 0.25)  # 1/4th from the top
        cv2.putText(frame, slogan, (text_x, text_y), font, 2.2, (0, 165, 255), 5, cv2.LINE_AA)  # Orange color in BGR

    # Second half of the cycle: Display jokes (question first, then answer after 4 seconds)
    else:
        joke_index = cycle_time % len(jokes)
        question, answer = jokes[joke_index]

        # Display question first, then answer after 4 seconds
        time_in_cycle = current_time % 8
        question_size = cv2.getTextSize(question, font, 2.2, 5)[0]  # 80pt = scale 2.2 in OpenCV
        answer_size = cv2.getTextSize(answer, font, 2.2, 5)[0]  # 80pt = scale 2.2 in OpenCV

        question_x = (frame_width - question_size[0]) // 2
        answer_x = (frame_width - answer_size[0]) // 2

        # Display the question at 1/4th of the screen height
        question_y = int(frame_height * 0.25)
        cv2.putText(frame, question, (question_x, question_y), font, 2.2, (255, 255, 255), 5, cv2.LINE_AA)  # White color

        if time_in_cycle >= 4:
            # Display the answer below the question
            answer_y = question_y + 80  # 80px below the question
            cv2.putText(frame, answer, (answer_x, answer_y), font, 2.2, (0, 165, 255), 5, cv2.LINE_AA)  # Orange color

    return frame

# Fortune phrases for when the 'fortune' keyword is spoken
fortunes = [
    "Beware, something wicked this way comes!",
    "Your future holds... shadows and mystery.",
    "The full moon calls... prepare for a transformation!",
    "A dark figure looms in your path. Tread carefully...",
    "The spirits whisper your name... will you answer?"
]

# Function to recognize voice commands asynchronously
def voice_recognition_thread():
    global recognized_command
    while True:
        with sr.Microphone() as source:
            print("Listening for 'fortune' command...")
            audio = recognizer.listen(source)
            try:
                command = recognizer.recognize_google(audio).lower()
                if "fortune" in command:
                    recognized_command = "fortune"
            except sr.UnknownValueError:
                pass
        time.sleep(1)  # Sleep for 1 second between recognitions to avoid continuous listening

# # Start the voice recognition thread
# voice_thread = Thread(target=voice_recognition_thread)
# voice_thread.daemon = True
# voice_thread.start()

# Function to draw text with Gothic font
def draw_gothic_text(image, text, position, font, color=(255, 0, 0)):
    """Draw text on an OpenCV image using PIL for Gothic fonts."""
    # Convert OpenCV image to PIL format
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    draw.text(position, text, font=font, fill=color)
    # Convert PIL image back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# Function to load and display the future_head.gif frames on a person's face for 10 seconds
def show_future_head(frame, landmarks):
    try:
        gif = Image.open(future_head_path)
    except Exception as e:
        print(f"Error loading GIF: {e}")
        return frame
    
    face_width = int((landmarks.part(16).x - landmarks.part(0).x) * 1.2)  # Rescale GIF to match face width
    pos = (landmarks.part(27).x - face_width // 2, landmarks.part(24).y - face_width // 2)  # Position over face

    start_time = time.time()
    while time.time() - start_time < 10:  # Show for 10 seconds
        for gif_frame in ImageSequence.Iterator(gif):
            gif_frame = gif_frame.convert('RGBA')
            frame_np = np.array(gif_frame)  # Convert PIL image to NumPy array
            gif_overlay = cv2.cvtColor(frame_np, cv2.COLOR_RGBA2BGRA)  # Convert RGBA to BGRA for OpenCV
            
            # Resize the GIF overlay to match the face size
            gif_overlay = resize_overlay_image(gif_overlay, face_width)  # Resize as needed
            
            # Overlay the GIF on the person's face
            frame = overlay_image_alpha(frame, gif_overlay, pos)
            
            # Show the current frame with the GIF
            cv2.imshow('Halloween Magic Mirror', frame)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # Delay for GIF frame duration
                break

    return frame

# Function to trigger fortunes when voice command is detected
def show_fortune(frame, landmarks):
    global fortune_active, fortune_start_time
    
    # Activate fortune and set the start time
    fortune_active = True
    fortune_start_time = time.time()
    
    random_fortune = random.choice(fortunes)
    
    # Play music for the fortune
    try:
        mixer.music.load('scary_music.mp3')
        mixer.music.play()
    except Exception as e:
        print(f"Error playing music: {e}")
        
    # Draw fortune text on the frame
    frame = draw_gothic_text(frame, random_fortune, (50, 200), font)

    # Show the GIF on the first person's face for 10 seconds
    frame = show_future_head(frame, landmarks)

    # Morph the face
    frame = morph_face(frame, landmarks)

    return frame

# Main program
cap = cv2.VideoCapture(0)

# Global variable to store the recognized command
recognized_command = None

# Start the voice recognition thread
voice_thread = Thread(target=voice_recognition_thread)
voice_thread.daemon = True
voice_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Collect landmarks for all detected faces
    landmarks_list = [predictor(gray, face) for face in faces]

    # Check if the recognized command is "fortune"
    if recognized_command == "fortune":
        frame = show_fortune(frame, landmarks_list[0])  # Apply the fortune to the first person detected
        recognized_command = None  # Reset command after showing fortune
    else:
        # Apply decorations based on number of people
        frame = place_decorations(frame, landmarks_list)
        
        # Add pumpkins in all four corners with bumpy jumping and wiggling
        frame = add_pumpkins_in_corners(frame)

        # Display slogans and jokes if fortune is not active
        frame = display_slogans_and_jokes(frame)

    # Deactivate fortune display after the set duration
    if fortune_active and time.time() - fortune_start_time > fortune_duration:
        fortune_active = False

    # Show the final frame with decorations and fortune
    cv2.imshow('Halloween Magic Mirror', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
