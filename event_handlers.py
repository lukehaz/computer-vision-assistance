import os
import re
import subprocess
from ultralytics.solutions import speed_estimation
import cv2
from threading import Thread
import queue
from ultralytics import YOLO

# Predefined YOLO model
predefined_model = YOLO("stairsgood.pt")

def execute_speed_detection(video_path):
    # Use the predefined YOLO model
    model = predefined_model

    # Remaining speed detection code...
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    # Video writer
    video_writer = cv2.VideoWriter("speed_estimation.avi",
                                   cv2.VideoWriter_fourcc(*'mp4v'),
                                   fps,
                                   (w, h))

    line_pts = [(0, 360), (1280, 360)]

    # Init speed-estimation obj
    speed_obj = speed_estimation.SpeedEstimator()
    speed_obj.set_args(reg_pts=line_pts,
                       names=model.names,
                       view_img=True)

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print("Video frame is empty or video processing has been successfully completed.")
            break

        tracks = model.track(im0, persist=True, show=False)
        im0 = speed_obj.estimate_speed(im0, tracks)
        video_writer.write(im0)

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()

def handle_first_page_events(event, values):
    is_training = False
    is_detection = False

    if event == 'Next':
        if values['training_radio']:
            is_training = True
        elif values['detection_radio']:
            is_detection = True

    return is_training, is_detection

def handle_training_page_events(event, values, window, yolo_thread, output_queue):
    if event == 'Train':
        yaml_path = values['yaml']

        if not os.path.isfile(yaml_path):
            window['status'].update('Error: Invalid YAML path.')
            sg.popup_error('Invalid YAML path. Please provide a valid YAML file.')
            return

        detection_type = 'segment' if values['segmentation'] else 'detect'
        model_size = 'n' if values['model_n'] else 's' if values['model_s'] else 'm' if values['model_m'] else 'l' if values['model_l'] else 'x'
        model_name = f'yolov8{model_size}-seg.pt' if detection_type == 'segment' else f'yolov8{model_size}.pt'
        epochs = values['epochs']

        command = f'yolo task={detection_type} mode=train model="{model_name}" data="{yaml_path}" epochs={epochs}'

        yolo_thread = Thread(target=execute_yolo_command, args=(command, output_queue), daemon=True)
        yolo_thread.start()

        window['Train'].update(visible=False)

    while not output_queue.empty():
        line = output_queue.get()
        if line is None:
            window['Train'].update(visible=True)
        else:
            print(line)  # This line will print the output to the console
            update_progress(window, line)

def handle_detection_page_events(event, values, window, yolo_thread, output_queue):
    if event == 'Process':
        video_path = values['video']
        enable_speed_detection = values['enable_speed_detection']

        # Existing YOLO detection code...
        # ...

        if enable_speed_detection:
            # Execute speed detection code
            execute_speed_detection(video_path)

        # Existing code for updating progress and handling outputs...
        # ...

def execute_yolo_command(command, output_queue):
    output_queue.put(command)
    print(command)
    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

        for line in iter(process.stdout.readline, b''):
            line = line.decode("utf-8").strip()
            output_queue.put(line)

        process.stdout.close()
        process.wait()
        output_queue.put(None)
    except Exception as e:
        output_queue.put(f'Error: {e}')
        output_queue.put(None)

def update_progress(window, line):
    match = re.search(r'\((\d+)/(\d+)\)', line)
    if match:
        current_frame, total_frames = int(match.group(1)), int(match.group(2))
        progress = int(current_frame / total_frames * 100)
        window['progress'].update(progress)
        return progress  # Return the progress value
