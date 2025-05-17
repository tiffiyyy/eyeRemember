from flask import Flask, request, jsonify 
from utils import save_face_db

app = Flask(__name__)

# decorator for path of access 
@app.route("/create-profile", methods=["POST"])
# function to create profile with image and name 
def create_profile(): 
    data = request.get_json()
    image = data.get("embeddings")
    name = data.get("name")

    save_face_db(image, name) 
    return jsonify({"status": "saved"}), 201

if __name__ == "__main__": 
    app.run(debug = True)
