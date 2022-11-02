import os
import glob
from keras.models import load_model
from flask import Flask, render_template, request, send_from_directory
from process_image import classify_image

app = Flask(__name__)

dir_path = os.path.dirname(os.path.realpath(__file__))
UPLOAD_FOLDER = "uploads/"
STATIC_FOLDER = "static"

# Load model
model = load_model(STATIC_FOLDER + "/eye_closed_detect_model_2.0.0.h5")

IMAGE_SIZE = 24


# home page
@app.route("/", methods=['GET'])
def home():
    filelist = glob.glob("uploads/*.*")
    for filePath in filelist:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file")
    return render_template("home.html")


@app.route("/classify", methods=["POST", "GET"])
def upload_file():
    if request.method == "GET":
        return render_template("home.html")

    else:
        file = request.files["image"]
        upload_image_path = os.path.join(UPLOAD_FOLDER, file.filename.replace("\\\\", ""))
        file.save(upload_image_path)

        l_res, r_res = classify_image(upload_image_path, model, IMAGE_SIZE)

    return render_template(
        "classify.html", image_file_name=file.filename, l_res=l_res, r_res=r_res
    )


@app.route("/classify/<filename>")
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
    app.run()
