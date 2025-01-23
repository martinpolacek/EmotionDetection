from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from flask_socketio import SocketIO
import cv2
import numpy as np
import base64
import io
from PIL import Image
import cv2
from facenet_pytorch import MTCNN
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch


# Načti model a procesor
model_name = "dima806/facial_emotions_image_detection"
processor = AutoImageProcessor.from_pretrained(model_name)
device = torch.device("cpu")
model = AutoModelForImageClassification.from_pretrained(model_name).to(device)

# Detektor obličejů
mtcnn = MTCNN()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
CORS(app)  # Povolení CORS pro celý server
socketio = SocketIO(app, cors_allowed_origins="*")  # Povolení CORS pro socket.io

@app.route('/')
def index():
  return render_template('index.html')

@socketio.on('image')
def handle_image(data):
  # Dekódování base64 obrázku
  sbuf = io.BytesIO()
  sbuf.write(base64.b64decode(data.split(',')[1]))
  pimg = Image.open(sbuf)

  # Konverze na OpenCV formát
  frame = cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)

  boxes, probs = mtcnn.detect(frame)
  print(boxes)

  if boxes is not None:

    #Find largest box
    largest_box = 0
    for i, box in enumerate(boxes):
        x, y, w, h = box
        if ((w-x) * (h-y)) > largest_box:
            largest_box = (w-x) * (h-y)
            largest_box_index = i

    for i, box in enumerate(boxes):
        x, y, w, h = box

        if i == largest_box_index:


          image_from_box = frame[int(y):int(h), int(x):int(w)]
          inputs = processor(images=image_from_box, return_tensors="pt").to(device)

          with torch.no_grad():
              outputs = model(**inputs)
              probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

          # Výstup: kategorie a pravděpodobnosti
          labels = model.config.id2label
          results = {labels[i]: float(probs[0][i]) for i in range(len(labels))}

          max_emotion = max(results, key=results.get)

          cv2.putText(frame, max_emotion, (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        cv2.rectangle(frame, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)


  # Konverze zpět na base64
  retval, buffer = cv2.imencode('.jpg', frame)
  b64_image = base64.b64encode(buffer).decode('utf-8')

  # Odeslání zpracovaného snímku zpět
  emit('processed_image', 'data:image/jpeg;base64,' + b64_image, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, allow_unsafe_werkzeug=True)
