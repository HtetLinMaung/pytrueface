import requests
from typing import List, Dict, Any
from typing import List
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import face_recognition
import numpy as np
from PIL import Image
import io

prefix_endpoint = "pytrueface"

app = FastAPI()


@app.post(f"{prefix_endpoint}/encode-face")
async def encode_face(label: str, file: UploadFile = File(...)):
    try:
        # Convert the file data to an image.
        image = Image.open(io.BytesIO(await file.read()))
        image = np.array(image)

        # Detect the face encodings for the uploaded image.
        face_encodings = face_recognition.face_encodings(image)

        # If no faces were found, return an error.
        if len(face_encodings) == 0:
            return JSONResponse(status_code=400, content={
                "code": 400,
                "message": "No faces found in image."
            })

        # If multiple faces were found, return an error.
        elif len(face_encodings) > 1:
            return JSONResponse(status_code=400, content={
                "code": 400,
                "message": "Multiple faces found in image."
            })

        # Return the face encoding data.
        # Convert numpy array to list for JSON serialization
        face_encoding = face_encodings[0].tolist()

        return {
            "code": 200,
            "message": "Calculate face encoding successful",
            "label": label,
            "face_encoding": face_encoding
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "code": 500,
            "message": str(e)
        })


@app.post("/recognize-face")
async def recognize_face(file: UploadFile = File(...), encodings_url: str = ''):
    try:
        # Convert the file data to an image.
        image = Image.open(io.BytesIO(await file.read()))
        image = np.array(image)

        # Detect the face encodings for the uploaded image.
        face_encodings = face_recognition.face_encodings(image)

        if len(face_encodings) == 0:
            return JSONResponse(status_code=400, content={
                "code": 400,
                "message": "No faces found in image."
            })

        # Fetch face encodings from the provided URL.
        response = requests.get(encodings_url)
        response.raise_for_status()
        known_encodings: List[Dict[str, Any]] = response.json()

        for known in known_encodings:
            # Ensure the encoding is a numpy array.
            known_encoding = np.array(known['face_encoding'])
            for face_encoding in face_encodings:
                # Compare face encodings.
                matches = face_recognition.compare_faces(
                    [known_encoding], face_encoding)

                if True in matches:
                    return {
                        "code": 200,
                        "message": "Face recognized",
                        "label": known['label'],
                        "face_data": face_encoding.tolist(),
                    }

        # If no matches were found, return an error.
        return JSONResponse(status_code=400, content={
            "code": 400,
            "message": "No matching face found.",
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={
            "code": 500,
            "message": str(e)
        })
