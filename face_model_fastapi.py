#install packages and pretrained model from Clip and facenet repos
#!pip install git+https://github.com/openai/CLIP.git
#!git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch


from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from PIL import Image
import torch
import open_clip
from sentence_transformers import util
from facenet_pytorch import MTCNN, InceptionResnetV1
import io
import logging
import cv2

# Initialize FastAPI
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the models
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

def imageEncoder(img):
    logging.debug("Encoding image...")
    if img.dtype != 'uint8':
        img = img.astype('uint8')  # Convert to 8-bit if necessary
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1

def crop_comp_img(person, person_id):
    logging.debug("Cropping and comparing images...")
    
    # Generate in-memory file paths
    person_crop_path = "person_crop.jpg"
    person_id_crop_path = "person_id_crop.jpg"
    
    # Create a face detection pipeline using MTCNN:
    mtcnn = MTCNN(image_size=160, margin=0)

    # Get cropped and prewhitened image tensor
    person_crop = mtcnn(person, save_path=person_crop_path)
    person_id_crop = mtcnn(person_id, save_path=person_id_crop_path)

    if person_crop is None or person_id_crop is None:
        raise ValueError("Face not detected in one or both images")
    
    # Calculate embedding (unsqueeze to add batch dimension)
    person_crop_embedding = resnet(person_crop.unsqueeze(0))
    person_id_crop_embedding = resnet(person_id_crop.unsqueeze(0))
    
    # Or, if using for VGGFace2 classification
    resnet.classify = True

    person_crop_probs = resnet(person_crop.unsqueeze(0))
    person_id_crop_probs = resnet(person_id_crop.unsqueeze(0))

    person_compare = cv2.imread(person_crop_path, cv2.IMREAD_UNCHANGED)
    person_id_compare = cv2.imread(person_id_crop_path, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(person_compare)
    img2 = imageEncoder(person_id_compare)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0]) * 100, 2)
    return score, person_compare, person_id_compare

@app.post("/compare/")
async def compare_images(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    try:
        logging.debug("Received images for comparison")
        # Read uploaded files
        image1 = Image.open(io.BytesIO(await file1.read()))
        image2 = Image.open(io.BytesIO(await file2.read()))

        # Generate score
        score, img1, img2 = crop_comp_img(image1, image2)
        result = "Same Person" if score >= 60 else "Not the same Person"  # Adjusted threshold
        logging.debug(f"Comparison result: {result} with score: {score}")
        return JSONResponse(content={"score": score, "result": result})
    except Exception as e:
        logging.error(f"Error in compare_images endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {e}")

if __name__ == "__main__":
    logging.debug("Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
