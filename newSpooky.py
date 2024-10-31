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
base_path = os.getcwd() # get current directory path

# Initialize Pygame Mixer for music playback
mixer.init()

# Voice Recognition and Text-to-Speech
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# Load pre-trained facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(f'{base_path}/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat')

# Define the paths to individual image files using the base path
horns_path = os.path.join(base_path, 'horns.png')
witch_hat_path = os.path.join(base_path, 'witch_hat.png')
pumpkin_path = os.path.join(base_path, 'pumpkin.png')
skeleton_path = os.path.join(base_path, 'skeleton.png')
jason_path = os.path.join(base_path, 'jason_vorhees.png')
future_head_path = os.path.join(base_path, 'future_head.gif')
qr_code_path = os.path.join(base_path, 'qr_code.png')

# Load the Halloween-themed decoration images with transparency (RGBA)
horns_img = cv2.imread(horns_path, cv2.IMREAD_UNCHANGED)
witch_hat_img = cv2.imread(witch_hat_path, cv2.IMREAD_UNCHANGED)
skeleton_img = cv2.imread(skeleton_path, cv2.IMREAD_UNCHANGED) 
pumpkin_img = cv2.imread(pumpkin_path, cv2.IMREAD_UNCHANGED) 
jason_img = cv2.imread(jason_path, cv2.IMREAD_UNCHANGED)
qr_code_img = cv2.imread(qr_code_path, cv2.IMREAD_UNCHANGED)

# Load Gothic Font (TrueType Font)
gothic_font_path = f'{base_path}/Blackletter.ttf'  # Replace with the path to your font
font = ImageFont.truetype(gothic_font_path, 70)  # Load the Gothic font with size 70

# Initialize the font size based on frame size (dynamic)
def get_font(frame, font_path=gothic_font_path, scale_factor=25):
    font_size = max(frame.shape[1] // scale_factor, 20)  # Adjust size dynamically
    return ImageFont.truetype(font_path, font_size)

# Add "Dare to scan?" with wiggle and jump effect
def wiggle_text_position(base_pos):
    wiggle_x = base_pos[0] + int(5 * math.sin(time.time() * 2))
    wiggle_y = base_pos[1] + int(5 * abs(math.sin(time.time() * 2)))
    return wiggle_x, wiggle_y

# Global state variables
fortune_active = False
fortune_start_time = 0
fortune_duration = 5  # Show the fortune for 5 seconds

# Function to manage timed display of messages and QR code
def display_spooky_messages(frame, start_time, qr_code_pinned=False):
    elapsed_time = (time.time() - start_time) % 30  # Cycle duration is 30 seconds
    font = get_font(frame)

    # Message 1: "Dare to see your fate?"
    if 5 <= elapsed_time < 15:
        frame = draw_gothic_text(frame, "Dare to see your fate? Say 'show my fortune' to reveal...", 
                                 (50, 100), font, (220, 20, 60))

    # Message 2: "Only the fearless solve the unknown" and centered QR code
    elif 20 <= elapsed_time < 25 and not qr_code_pinned:
        frame = draw_gothic_text(frame, "Only the fearless solve the unknown... \nDare to scan the QR code?", 
                                 (50, 50), font, (220, 20, 60))
        qr_code_size = frame.shape[1] // 10  # Adjust size to 1/10th of frame width
        qr_code_resized = resize_overlay_image(qr_code_img, qr_code_size)
        center_position = ((frame.shape[1] - qr_code_resized.shape[1]) // 2, frame.shape[0] // 3)
        frame = overlay_image_alpha(frame, qr_code_resized, center_position)
    
    # Move QR code to the top-left corner and make it persistent
    if elapsed_time >= 25 or qr_code_pinned:
        qr_code_pinned = True  # Set QR code to stay in top-left
        qr_code_size = frame.shape[1] // 10
        qr_code_resized = resize_overlay_image(qr_code_img, qr_code_size)
        # top_left_position = (10, 10)
        top_right_position = (frame.shape[1] - qr_code_resized.shape[1] - 10, 10)
        frame = overlay_image_alpha(frame, qr_code_resized, top_right_position)
    
    # Add the animated "Dare to scan?" text below the QR code
        wiggle_pos = wiggle_text_position((top_right_position[0], top_right_position[1] + qr_code_resized.shape[0] + 10))
        frame = draw_gothic_text(frame, "Scan!", wiggle_pos, get_font(frame, scale_factor=30), (220, 20, 60))
    
    return frame, qr_code_pinned

# # Function to manage timed display of messages and QR code
# def display_spooky_messages(frame, start_time):
#     elapsed_time = (time.time() - start_time) % 30  # Cycle duration 26 seconds
#     font = get_font(frame)

#     # Message 1: "Dare to see your fate?"
#     if 5 <= elapsed_time < 15:
#         frame = draw_gothic_text(frame, "Dare to see your fate? Say 'show my fortune' to reveal...", 
#                                  (50, 100), font, (220,20,60))

#     # Message 2: "Only the fearless solve the unknown" and QR code
#     elif 20 <= elapsed_time < 25:
#         frame = draw_gothic_text(frame, "Only the fearless solve the unknown... \nDare to scan the QR code?", 
#                                  (50, 50), font)
#         qr_code_size = frame.shape[1] // 10
#         qr_code_resized = resize_overlay_image(qr_code_img, qr_code_size)
#         center_position = ((frame.shape[1] - qr_code_resized.shape[1]) // 2, frame.shape[0] // 3)
#         frame = overlay_image_alpha(frame, qr_code_resized, center_position)
    
#     # Move QR code to the top-left corner after 6 seconds
#     elif 25 <= elapsed_time < 30:
#         qr_code_size = frame.shape[1] // 10
#         qr_code_resized = resize_overlay_image(qr_code_img, qr_code_size)
#         top_left_position = (10, 10)
#         frame = overlay_image_alpha(frame, qr_code_resized, top_left_position)
        
#     return frame

# def display_spooky_message(frame, start_time):
#     elapsed_time = time.time() - start_time
#     font = get_font(frame)
    
#     # Resize the QR code dynamically to 1/10th of the frame width
#     qr_code_size = frame.shape[1] // 10  # Set to 1/10th of the screen width
#     qr_code_resized = resize_overlay_image(qr_code_img, qr_code_size)
    
#     if 15 < elapsed_time <= 20:
#         frame = draw_gothic_text(frame, "Are you ready to solve some spooky riddle?", (50, 100), font, (50, 0, 0))
    
#     elif 20 < elapsed_time <= 25:
#         # Center the QR code with text
#         frame = draw_gothic_text(frame, "Scan the code to solve the riddle.", (50, 100), font)
#         center_position = ((frame.shape[1] - qr_code_resized.shape[1]) // 2, frame.shape[0] // 3)
#         frame = overlay_image_alpha(frame, qr_code_resized, center_position)
    
#     elif elapsed_time > 25:
#         # Move the QR code to the top-left corner
#         top_left_position = (10, 10)
#         frame = overlay_image_alpha(frame, qr_code_resized, top_left_position)
    
#     return frame


# Pumpkin animation function for wiggle and jump
def wiggle_pumpkin_position(base_pos, frame_width, frame_height):
    current_time = time.time()
    wiggle_amount = int(math.sin(current_time * 3) * (frame_width // 60))
    jump_amount = int(abs(math.sin(current_time * 5)) * (frame_height // 60))
    return base_pos[0] + wiggle_amount, base_pos[1] - jump_amount

# Add animated pumpkins in the corners
def add_pumpkins(frame):
    pumpkin_resized = resize_overlay_image(pumpkin_img, frame.shape[1] // 10)
    frame_width, frame_height = frame.shape[1], frame.shape[0]

    # Bottom-left corner with wiggle and jump
    pos_left = wiggle_pumpkin_position((10, frame_height - pumpkin_resized.shape[0] - 10), frame_width, frame_height)
    frame = overlay_image_alpha(frame, pumpkin_resized, pos_left)

    # Bottom-right corner with wiggle and jump
    pos_right = wiggle_pumpkin_position(
        (frame_width - pumpkin_resized.shape[1] - 10, frame_height - pumpkin_resized.shape[0] - 10),
        frame_width, frame_height
    )
    frame = overlay_image_alpha(frame, pumpkin_resized, pos_right)

    return frame

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
    frame = cv2.circle(frame, (center_x, center_y), morph_radius, (220, 20, 60), 3)  # Example: green circle for effect

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
    "A dark figure looms in your path. Tread carefully...",
    "The spirits whisper your name... will you answer?",
    "A shadow follows you... \nbut fear not, it's only looking for a friend!",
    "Beware! Your future holds unexpected twists... \nand maybe a full-grade surprise!",
    "Under the next full moon, \na secret will be revealed to you... in whispers.",
    "A chilling breeze brings whispers of fortune... \nor maybe just the deadline you forgot about!",
    "You may encounter a friendly ghost soon... \nperhaps with a riddle for you to solve.",
    "Your laughter will echo tonight... \nbut will it be yours, or another's?"
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
def draw_gothic_text(image, text, position, font, color=(220, 20, 60)):
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

fortune_display_duration = 3  # in seconds

def show_fortune(frame, landmarks):
    global fortune_active, fortune_start_time, current_fortune_index, recognized_command
    font = get_font(frame)

    # Start the fortune display sequence if not already active
    if not fortune_active:
        fortune_active = True
        fortune_start_time = time.time()
        current_fortune_index = random.randint(0, 8)  # Reset to the first fortune

    # Calculate time since the start of the fortune display
    elapsed_time = time.time() - fortune_start_time

    # Update the fortune text every few seconds based on the display duration
    if elapsed_time > (current_fortune_index + 1) * fortune_display_duration:
        current_fortune_index += 1
        if current_fortune_index >= len(fortunes):
            # End of the fortune sequence, reset states to allow reactivation
            fortune_active = False
            recognized_command = None  # Reset command for reactivation
            return frame  # Exit to avoid displaying additional text after reset

    # Display the current fortune text based on current index
    fortune_text = fortunes[current_fortune_index]
    frame = draw_gothic_text(frame, fortune_text, (50, 200), font, (220, 20, 60))

    # Play music for the fortune if newly activated
    if elapsed_time < fortune_display_duration:
        try:
            mixer.music.load('scary_music.mp3')
            mixer.music.play()
        except Exception as e:
            print(f"Error playing music: {e}")

    # Show GIF and morph face effect for added fortune effect
    frame = show_future_head(frame, landmarks)
    frame = morph_face(frame, landmarks)

    return frame

# # Function to trigger fortunes when voice command is detected
# def show_fortune(frame, landmarks):
#     # # Activate fortune and set the start time
#     # fortune_active = True
#     # fortune_start_time = time.time()
    
#     random_fortune = random.choice(fortunes)
    
#     global fortune_active, fortune_start_time, current_fortune_index
    
#     # Activate fortune and set the start time if not already active
#     if not fortune_active:
#         fortune_active = True
#         fortune_start_time = time.time()
#         current_fortune_index = 0  # Start at the first fortune
    
#     # Calculate elapsed time since fortune started
#     elapsed_time = time.time() - fortune_start_time
    
#     # Update the fortune index every few seconds
#     if elapsed_time > (current_fortune_index + 1) * fortune_display_duration:
#         current_fortune_index += 1  # Move to the next fortune
#         if current_fortune_index >= len(fortunes):
#             current_fortune_index = 0  # Loop back to the beginning if we reach the end
    
#     # Select and display the current fortune
#     fortune_text = fortunes[current_fortune_index]
#     font = get_font(frame)  # Ensure the font size is responsive
#     frame = draw_gothic_text(frame, fortune_text, (50, 200), font, (220, 20, 60))
    
#     # # Deactivate fortune display after the full cycle duration has passed (e.g., 10 seconds total)
#     # if elapsed_time > fortune_display_duration * len(fortunes):
#     #     fortune_active = False

#     # Play music for the fortune
#     try:
#         mixer.music.load('scary_music.mp3')
#         mixer.music.play()
#     except Exception as e:
#         print(f"Error playing music: {e}")
        
#     # Draw fortune text on the frame
#     frame = draw_gothic_text(frame, random_fortune, (50, 200), font)

#     # Show the GIF on the first person's face for 10 seconds
#     frame = show_future_head(frame, landmarks)

#     # Morph the face
#     frame = morph_face(frame, landmarks)

#      # Cycle through fortunes every few seconds
#     elapsed_time = time.time() - fortune_start_time
#     if elapsed_time > (current_fortune_index + 1) * fortune_duration:
#         current_fortune_index += 1
#         if current_fortune_index >= len(fortunes):
#             # End the fortune sequence and reset recognized_command
#             fortune_active = False
#             recognized_command = None  # Reset command for reactivation

#     # Display the current fortune
#     fortune_text = fortunes[current_fortune_index % len(fortunes)]
#     frame = draw_gothic_text(frame, fortune_text, (50, 200), font, (220, 20, 60))

#     return frame

# Main program
cap = cv2.VideoCapture(0)

# Global variable to store the recognized command
recognized_command = None

# Start the voice recognition thread
voice_thread = Thread(target=voice_recognition_thread)
voice_thread.daemon = True
voice_thread.start()

# main program 
start_time = time.time()
qr_code_pinned = False

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    # Display looping messages and QR code based on time
    frame, qr_code_pinned = display_spooky_messages(frame, start_time, qr_code_pinned)
    
    # Collect landmarks for all detected faces
    landmarks_list = [predictor(gray, face) for face in faces]

    # Check if the recognized command is "fortune"
    if recognized_command == "fortune" and landmarks_list:
        frame = show_fortune(frame, landmarks_list[0])  # Apply the fortune to the first person detected
        recognized_command = None  # Reset command after showing fortune
    else:
        # Apply decorations based on number of people
        frame = place_decorations(frame, landmarks_list)
        
        # Add pumpkins in all four corners with bumpy jumping and wiggling
        frame = add_pumpkins(frame)

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
