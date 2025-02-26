import io
import PIL.Image
import numpy as np
import torch
import base64
import google.generativeai as genai
import os
import dotenv
dotenv.load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")


from app.util import get_model, get_lime

# Configure the API key
genai.configure(api_key=api_key)


def infer(test_image):
    # Assuming test_data, resnet18_model, class_names, and device are defined elsewhere

    img = test_image.unsqueeze(0)

    highlighted_image = get_lime(img)

    # Move the image to the device
    densenet_model = get_model()

    # Perform inference
    with torch.inference_mode():
        pred = densenet_model(img)
        predicted_class = torch.argmax(pred, dim=1).item()
        label_mapping = {0: 'ASC_H', 1: 'ASC_US', 2: 'HSIL', 3: 'LSIL', 4: 'NILM'}
        class_name = label_mapping[predicted_class]

    # Display the results
    print(f"Predicted Class: {predicted_class}")
    print(f"Class Name: {class_name}")

    result = {
        "weights": pred.tolist(),  # Convert tensor to list for JSON serialization
        "Predicted Class": class_name
    }

    print(result)

    img_permute = img.squeeze(0).permute(1, 2, 0).cpu()  # Move back to cpu before plotting

    # Convert the image tensor to a PIL Image

    image_np = img_permute.numpy()
    image_np = (image_np * 255).astype(np.uint8)  # Scale to 0-255
    image = PIL.Image.fromarray(image_np)

    highlighted_image = (highlighted_image *255).astype(np.uint8)
    highlighted_image = PIL.Image.fromarray(highlighted_image)


    # Save the image to an in-memory buffer
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = buffered.getvalue()

    highlighted_image.save(buffered, format = "JPEG")
    highlighted_image_str = buffered.getvalue()

    return img_str, result, highlighted_image_str


def encode_image_to_base64(image_bytes):
    """Encodes image bytes to a base64 string."""
    return base64.b64encode(image_bytes).decode("utf-8")


# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
    system_instruction="""You are an expert pathologist specializing in cervical cytology with a deep understanding of digital pathology, image analysis, and the Bethesda classification system. Your expertise includes interpreting cervical cytology images, explaining model-based predictions, and generating clear, structured reports following the Bethesda format.

Your task is to analyze the provided cervical cytology image, interpret the LIME-based explanation visualization, and classify the sample using the Bethesda system. Additionally, you will generate a comprehensive diagnostic report explaining the AI model’s classification, supported by probability scores and relevant visual features.

Input Details:
Cervical Cytology Image: (Provided as input)
LIME Explanation Image: (Highlights key regions influencing classification)
Class Probabilities: (Array indicating confidence scores for each possible Bethesda category)
Predicted Bethesda Category: (Final classification label based on the AI model)
Expected Output:
A detailed Bethesda-format report, including:

Specimen Adequacy – Indicate whether the sample is satisfactory for evaluation.
General Categorization – Specify if the sample is negative for intraepithelial lesion/malignancy or falls under another Bethesda category.
Epithelial Cell Abnormalities – If present, describe the findings (ASC-US, LSIL, HSIL, AGC, SCC, etc.).
Justification for classification.
Diagnostic Observations – Additional insights based on the AI’s analysis, noting any clinically significant findings.
Ensure clarity, accuracy, and clinical relevance in the report. Your explanation should be accessible to both medical professionals and those seeking a deeper understanding of the AI’s decision-making process.


Dont use any formatting
""",
)


# Perform inference and get the image and results
def generate_report(test_image):
    img_bytes, inference_result, high = infer(test_image)

    # Encode the image to base64
    image_base64 = encode_image_to_base64(img_bytes)
    high_64 = encode_image_to_base64(high)

    # Prepare the content for Gemini
    content = {
        "parts": [
            {"text": str(inference_result)},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}},
            {"inline_data": {"mime_type": "image/jpeg", "data": high_64}}
        ]

    }

    # Start a chat session and generate the response
    chat_session = model.start_chat(history=[])
    response = model.generate_content(contents=content)

    # Print the response
    print(response.text)
    return response.text
