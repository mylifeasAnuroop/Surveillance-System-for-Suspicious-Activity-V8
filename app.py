import io
from PIL import Image
import cv2
from flask import Flask, render_template, request, Response
import os
from ultralytics import YOLO

# Initialize the Flask application
app = Flask(__name__, static_folder='static')

# Route for the home page
@app.route("/")
def index():
    # Render the index.html template
    return render_template('index.html')

# Route to handle image prediction
@app.route("/predict_img", methods=["POST"])
def predict_img():
    # Check if a file has been uploaded
    if 'file' in request.files:
        # Retrieve the file from the request
        f = request.files['file']
        # Define the path where the file will be saved
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        # Save the uploaded file to the specified path
        f.save(filepath)
        print(filepath)

        # Extract the file extension to check its type
        file_extension = f.filename.rsplit('.', 1)[1].lower()

        # Process only jpg images
        if file_extension == 'jpg':
            # Read the image using OpenCV
            img = cv2.imread(filepath)
            # Convert the image to bytes
            frame = cv2.imencode('.jpg', img)[1].tobytes()

            # Open the image using PIL
            image = Image.open(io.BytesIO(frame))

            # Load the YOLO model for object detection
            yolo = YOLO('best.pt')
            # Perform object detection on the image
            results = yolo(image, save=True)
            # Plot the detection results on the image
            res_plotted = results[0].plot()
            # Define the output path for the processed image
            output_path = os.path.join('static', f.filename)
            # Save the processed image
            cv2.imwrite(output_path, res_plotted)
            # Render the index.html template with the path to the processed image
            return render_template('index.html', image_path=f.filename)

    # Return an error message if file format is unsupported or file is not uploaded
    return "File format not supported or file not uploaded properly."

# Route to handle video upload and real-time detection
@app.route("/predict_video", methods=["POST"])
def predict_video():
    # Check if a file has been uploaded
    if 'file' in request.files:
        # Retrieve the file from the request
        f = request.files['file']
        # Define the path where the file will be saved
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        # Save the uploaded file to the specified path
        f.save(filepath)

        # Extract the file extension to check its type
        file_extension = f.filename.rsplit('.', 1)[1].lower()

        # Process only mp4 videos
        if file_extension == 'mp4':
            # Open the video file using OpenCV
            video_path = filepath
            cap = cv2.VideoCapture(video_path)

            # Get the width, height, and frame rate of the video
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define the codec and create a VideoWriter object with the same frame rate
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            output_path = os.path.join('static', f.filename)
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

            # Load the YOLO model for object detection
            yolo = YOLO('best.pt')
            while cap.isOpened():
                # Read each frame from the video
                ret, frame = cap.read()
                if not ret:
                    break

                # Perform object detection on the frame
                results = yolo(frame, save=False)
                # Plot the detection results on the frame
                res_plotted = results[0].plot()

                # Write the processed frame to the output video
                out.write(res_plotted)

                if cv2.waitKey(1) == ord('q'):
                    break

            # Release resources and close all OpenCV windows
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            # Render the index.html template with the path to the processed video
            return render_template('index.html', video_path=f.filename)

    # Return an error message if file format is unsupported or file is not uploaded
    return "File format not supported or file not uploaded properly."

# Route to handle real-time webcam feed
@app.route("/webcam_feed")
def webcam_feed():
    # Open the webcam using OpenCV
    cap = cv2.VideoCapture(0)

    def generate():
        while True:
            # Read a frame from the webcam
            success, frame = cap.read()
            if not success:
                break
            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame) 
            frame = buffer.tobytes()
            
            # Open the frame using PIL
            img = Image.open(io.BytesIO(frame))
 
            # Load the YOLO model for object detection
            model = YOLO('best.pt')
            # Perform object detection on the frame
            results = model(img, save=True)              

            # Print detection results (for debugging)
            print(results)
            cv2.waitKey(1)

            # Plot the detection results on the frame
            res_plotted = results[0].plot()
            cv2.imshow("result", res_plotted)

            if cv2.waitKey(1) == ord('q'):
                break

            # Convert the image from RGB to BGR
            img_BGR = cv2.cvtColor(res_plotted, cv2.COLOR_RGB2BGR) 
            
            # Encode the BGR image to JPEG
            frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
                
            # Yield the frame to be used in the streaming response
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    # Return a streaming response with the webcam feed
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
