from flask import Flask, request, jsonify
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
import os
# from dotenv import load_dotenv
# load_dotenv()
from flask_cors import CORS
from werkzeug.utils import secure_filename

from semantic_similarity import parseText

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# NOTE: DO NOT COMMIT THIS PART!
subscription_key = "0409316ef1964bf993dccfc81173cfdf"
endpoint = "https://handwriting-htn-2021.cognitiveservices.azure.com/"
## END

vision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# function to call read api from azure computer vision
def extractText(read_image):
  f = open(read_image, 'rb')
  read_response = vision_client.read_in_stream(f, raw=True)
  f.close()
  result = ''

  # operation location
  read_operation_location = read_response.headers["Operation-Location"]

  operation_id = read_operation_location.split("/")[-1]

  # calling the get api to retrieve the results

  while True:
    read_result = vision_client.get_read_result(operation_id)
    if read_result.status not in ['notStarted', 'running']:
      break
    time.sleep(5)

  # adding the detected text to results
  if read_result.status == OperationStatusCodes.succeeded:
    for text_result in read_result.analyze_result.read_results:
      for line in text_result.lines:
        result = result + " " + line.text
  
  return result

# routes
@app.route("/", methods=["GET", "POST"])
def main():
    return "<h1>Hello!</h1>"

@app.route("/find", methods=["POST", "GET"])
def find():
  result = ""
  if request.method == 'POST':
    image = request.files['image']
    # image = request.form.get('image')
    # image.save('im-received.jpg')
    print(image)
    
    # read image via file.stream
    # img = Image.open(image.stream)
    # print(img)
    filename = secure_filename(image.filename)
    image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    #print('upload_image filename: ' + filename)
    # flash('Image successfully uploaded and displayed below')
    image_url = "./uploads/" + filename
    # calling the extractText function
    result = extractText(image_url)
  
  else:
    # Test 
    image = "img1.jpg"
    result = extractText(image)
  
  # Parse text and return status here:
  try:
    markdown = parseText(result)
    print("Success")
    return jsonify({'result': markdown}) # return string.

  except Exception as err:
    print("An error occurred:")
    print(err)
    return jsonify({'error': str(err)}) # return error string


# once again lol

# everytime I forget to add this line of code
# it's necessary for running flask apps on repl.it
if __name__ == "__main__":
  app.run(host="0.0.0.0", port=8000)
  