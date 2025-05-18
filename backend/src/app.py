# By Tiffany Le and Linus Wong
# Last Updated: May 17, 2025 

from flask import Flask, request, jsonify 
from utils import save_face_db
from main import main

app = Flask(__name__)

main()

# decorator for path of access 
@app.route("/create-profile", methods=["POST"])
# function to create profile with image and name 
def create_profile(): 
    image = request.files.get("image")
    name = request.form.get("name")

    if not name or not image:
        return jsonify({"error": "Missing name or image"}), 400

    save_face_db(image, name) 
    return jsonify({"status": "saved"}), 201

if __name__ == "__main__": 
    app.run(debug = False)
