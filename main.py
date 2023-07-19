import io
from fastapi import FastAPI, File, UploadFile, HTTPException
import face_recognition
import numpy as np
import pickle
import os
import uuid
from asyncpg import create_pool
from typing import Dict
from PIL import Image

app = FastAPI()

# For storing face encodings in-memory
known_face_encodings = {}
# Database connection pool
pool = None

@app.on_event("startup")
async def startup():
    global pool
    pool = await create_pool('your_postgres_connection_string')
    
    # Create face_encodings directory if not exists
    if not os.path.exists("face_encodings"):
        os.makedirs("face_encodings")

    # Load known faces from PostgreSQL to memory
    async with pool.acquire() as connection:
        rows = await connection.fetch('SELECT label, file_name FROM faces')
        for row in rows:
            label = row['label']
            file_name = row['file_name']
            # Load face encoding from file
            with open(os.path.join("face_encodings", f"{file_name}.pkl"), 'rb') as f:
                face_encoding = pickle.load(f)
                known_face_encodings[label] = face_encoding
                

@app.post("/addFace")
async def add_face(label: str, file: UploadFile = File(...)):
    try:
        # Convert the file data to an image.
        image = Image.open(io.BytesIO(await file.read()))
        image = np.array(image)

        # Detect the face encodings for the uploaded image.
        face_encodings = face_recognition.face_encodings(image)

        # If no faces were found, return an error.
        if len(face_encodings) == 0:
            raise HTTPException(status_code=400, detail="No faces found in image.")

        # If multiple faces were found, return an error.
        elif len(face_encodings) > 1:
            raise HTTPException(status_code=400, detail="Multiple faces found in image.")

        # Save face encoding to file
        unique_name = str(uuid.uuid4())
        file_path = os.path.join("face_encodings", f"{unique_name}.pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(face_encodings[0], f)

        # Save label and encoding file name to database
        async with pool.acquire() as connection:
            await connection.execute(
                'INSERT INTO faces (label, file_name) VALUES ($1, $2)',
                label,
                unique_name
            )

        # Also save face encoding in memory
        known_face_encodings[label] = face_encodings[0]

        return {
            "code": 200,
            "message": "Face encoding calculated and stored successfully",
            "label": label,
            "file_name": unique_name
        }

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))
