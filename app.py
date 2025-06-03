# app.py
# Restored Gemini API usage to original genai.Client pattern.

import streamlit as st
import os
import requests
from PIL import Image, UnidentifiedImageError # Added UnidentifiedImageError for specific exception handling
import wave
import torch
from transformers import pipeline, BitsAndBytesConfig
from dotenv import load_dotenv
from google import genai
from google.genai import types

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="Radiology with MedGemma")

# --- Configuration and API Key Handling ---
# Load environment variables from .env file (primarily for local development)
load_dotenv()

hf_token = os.getenv("HF_TOKEN")
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not hf_token:
    st.warning("HF_TOKEN is not set. MedGemma model loading might fail if the model is private or gated.")
if not gemini_api_key:
    st.warning("GOOGLE_API_KEY is not set. Gemini TTS will not work.")

# Initialize Gemini API Client using genai.Client as per original script
gemini_client = None
if gemini_api_key:
    try:
        gemini_client = genai.Client(api_key=gemini_api_key)
        # st.success("Gemini API client initialized successfully.")
    except Exception as e:
        st.error(f"Failed to initialize Gemini API client: {e}")
        st.info("Ensure 'google-genai' library is up-to-date and your API key is valid.")
else:
    st.error("Gemini API key is missing. Gemini TTS functionality will be disabled.")


# --- Model Loading (MedGemma) ---
@st.cache_resource
def load_medgemma_model():
    """
    Loads the MedGemma model. Uses st.cache_resource to load only once.
    """
    st.write("Loading model... This may take a moment.")
    if not torch.cuda.is_available():
        st.error("CUDA is not available. MedGemma (4b-it) requires a GPU.")
        return None
    try:
        if not hf_token:
            st.info("HF_TOKEN is not provided. Assuming model is public or already cached.")

        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        model_kwargs = dict(
            torch_dtype=torch.bfloat16,
            device_map="cuda:0",
            quantization_config=quantization_config
        )
        hf_pipeline_task = os.getenv("HF_PIPELINE_TASK", "image-text-to-text")
        # st.write(f"Using Hugging Face pipeline task: {hf_pipeline_task}")

        pipe = pipeline(
            hf_pipeline_task,
            model="google/medgemma-4b-it",
            model_kwargs=model_kwargs,
            token=hf_token if hf_token else None
        )
        st.success("MedGemma model loaded successfully!")
        return pipe
    except Exception as e:
        st.error(f"Error loading MedGemma model: {e}")
        st.info("Ensure HF_TOKEN is correct (if model is gated), you have a compatible GPU, sufficient GPU RAM, and correct CUDA/PyTorch/bitsandbytes setup. Also check the HF_PIPELINE_TASK if you've set it.")
        return None

pipe = load_medgemma_model()

# --- Utility Functions ---
def infer(prompt: str, image: Image.Image, system: str = None) -> str:
    """
    Uses MedGemma to generate a plain-language report based on the provided prompt and image.
    The input format (messages vs. direct image/prompt) depends on the pipeline task used.
    This function retains the message-based input from your original script.
    """
    if pipe is None:
        st.error("MedGemma model failed to load. Cannot perform inference.")
        return "Error: Model not loaded."

    # Saving image temporarily to pass its path, as per original script's logic.
    # This is common for some multimodal models expecting file paths in structured input.
    temp_image_path = "temp_medgemma_input_image.png"
    try:
        image.save(temp_image_path)
    except Exception as e:
        st.error(f"Error saving temporary image: {e}")
        return "Error: Could not process image for inference."

    # This format is specific and assumes the MedGemma pipeline (`pipe`) is set up to receive it.
    # If using a standard "image-to-text" pipeline, it might expect `pipe(image_object, prompt=text_prompt)`.
    messages = []
    if system:
        messages.append({
            "role": "system",
            "content": [{"type": "text", "text": system}]
        })
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt},
            {"type": "image", "image": temp_image_path} # Path to the image
        ]
    })

    try:
        # The `pipe(messages, ...)` call assumes the pipeline is configured to handle this message structure.
        # Your original script used `pipe(text=messages, ...)` which is even more specific.
        # Let's use `pipe(messages, ...)` as it's more common for pipelines that take structured input.
        # If `text=messages` is strictly required by your `google/medgemma-4b-it` setup, revert this.
        output = pipe(messages, max_new_tokens=2048)

        # Parsing the output - this can vary greatly between Hugging Face pipelines.
        # The following is an attempt to robustly get the generated text.
        if isinstance(output, list) and len(output) > 0:
            generated_item = output[0]
            if "generated_text" in generated_item:
                response_content = generated_item["generated_text"]
                # Further parsing if response_content itself is structured (e.g., chat format)
                if isinstance(response_content, list) and response_content and isinstance(response_content[-1], dict) and "content" in response_content[-1]:
                    response = response_content[-1]["content"] # Matches original script's deep parse
                elif isinstance(response_content, str):
                    response = response_content
                else:
                    response = str(response_content) # Fallback
            elif "answer" in generated_item: # Common in VQA pipelines
                response = generated_item["answer"]
            else:
                st.warning(f"Unexpected output structure from MedGemma (keys: {generated_item.keys()}): {generated_item}")
                response = "Error: Could not parse MedGemma response."
        else:
            st.warning(f"Unexpected or empty output from MedGemma: {output}")
            response = "Error: No response from MedGemma."

    except Exception as e:
        st.error(f"Error during MedGemma inference: {e}")
        st.exception(e) # Provides more details in logs
        response = f"Error during inference: {str(e)}"
    finally:
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)
    return response

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
    """
    Converts raw PCM audio data into a proper .wav file.
    """
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm)

# --- Streamlit UI ---
st.title("Demo: Radiology with MedGemma")
st.markdown(
    """
Experience how advanced AI can simplify complex medical information with this interactive demo.

## AI for Medical Image and Text Analysis

**MedGemma** is a state-of-the-art AI model from Google DeepMind, purpose-built for healthcare applications. It can analyze a variety of medical images—including chest X-rays, dermatology photos, ophthalmology images, and histopathology slides—and generate clear, accessible reports in plain language. Trained on diverse, de-identified medical datasets, MedGemma highlights key findings and explains them in a way that’s easy to understand.

- Supports image uploads and links for instant analysis
- Provides both written and spoken explanations
- Handles both medical images and text-based queries

MedGemma can be fine-tuned to fit specific medical imaging or text use cases, making it a flexible and privacy-preserving tool for healthcare applications.
    """
)

with st.sidebar:
    st.header("Input Options")
    source_type = st.radio("Select Image Source:", ["Upload File", "Enter URL"])
    image_input = None

    if source_type == "Upload File":
        uploaded_file = st.file_uploader("Upload a medical image (e.g., X-ray, CT scan)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            try:
                image_input = Image.open(uploaded_file)
                st.image(image_input, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error opening uploaded image: {e}")
                image_input = None
    else: # Enter URL
        image_url = st.text_input("Enter Image URL:")
        if image_url:
            try:
                headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
                response = requests.get(image_url, headers=headers, stream=True, timeout=10)
                response.raise_for_status()
                image_input = Image.open(response.raw)
                st.image(image_input, caption="Image from URL", use_container_width=True)
            except requests.exceptions.RequestException as e:
                st.error(f"Error fetching image from URL: {e}. Please check the URL and ensure it's a direct link to an image.")
                image_input = None
            except UnidentifiedImageError:
                 st.error(f"Cannot identify image file. The URL might not point to a valid image or the format is unsupported: {image_url}")
                 image_input = None
            except Exception as e:
                st.error(f"Error loading image from URL: {e}")
                image_input = None

    text_prompt = st.text_area("Instructions (e.g., 'Describe this X-ray in simple terms.')", "Describe this chest X-ray and look for any abnormalities.")

st.header("Generated Report")

if st.button("Generate Report"):
    if image_input is None:
        st.warning("Please upload an image or provide a valid image URL.")
    elif not text_prompt.strip():
        st.warning("Please provide instructions or a question for the model.")
    elif pipe is None:
        st.error("MedGemma model is not loaded. Cannot generate report. Check logs for errors.")
    else:
        with st.spinner("Analyzing image and generating report..."):
            try:
                processed_image = image_input
                if processed_image.mode != 'RGB': # Ensure image is RGB for consistency
                    processed_image = processed_image.convert('RGB')

                report = infer(text_prompt, processed_image)
                st.subheader("Text Report:")
                st.markdown(report)

                # TTS Generation using the restored gemini_client pattern
                if gemini_client: # Check if client was initialized
                    st.subheader("Audio Report:")
                    try:
                        # Using gemini_client.models.generate_content as per original script
                        tts_model_name = os.getenv("GEMINI_TTS_MODEL_NAME", "gemini-2.5-flash-preview-tts") # Allow override via env var
                        tts_voice_name = os.getenv("GEMINI_TTS_VOICE_NAME", "Kore") # Allow override

                        # st.write(f"Generating audio with Gemini TTS (model: {tts_model_name}, voice: {tts_voice_name}) for: '{report[:100]}...'")

                        tts_response = gemini_client.models.generate_content(
                            model=tts_model_name, # e.g., "gemini-2.5-flash-preview-tts"
                            contents=report,
                            config=types.GenerateContentConfig(
                                response_modalities=["AUDIO"],
                                speech_config=types.SpeechConfig(
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=tts_voice_name, # e.g., 'Kore'
                                        )
                                    )
                                ),
                            )
                        )
                        # Parsing response as per original script
                        audio_data_raw_pcm = tts_response.candidates[0].content.parts[0].inline_data.data

                        if not audio_data_raw_pcm:
                            st.warning("Gemini returned empty audio data. This might happen for very short or invalid prompts, API issues, or incorrect model/voice configuration.")
                        else:
                            audio_filename = 'generated_report.wav'
                            # Assuming Gemini TTS output matches these parameters (24kHz, 16-bit PCM mono)
                            # These should be verified from Gemini documentation if issues occur.
                            wave_file(audio_filename, audio_data_raw_pcm, rate=24000, sample_width=2, channels=1)
                            st.audio(audio_filename, format='audio/wav')
                            if os.path.exists(audio_filename):
                                os.remove(audio_filename)
                    except Exception as e:
                        st.error(f"Error generating audio with Gemini TTS: {e}")
                        st.exception(e) # Show full traceback for debugging
                        st.info("Please verify your GOOGLE_API_KEY, ensure 'google-generativeai' library is up-to-date, check the report text, and confirm the TTS model name (GEMINI_TTS_MODEL_NAME) and voice (GEMINI_TTS_VOICE_NAME) are correct for your API version and access.")
                elif not gemini_api_key:
                     st.info("Gemini API key not provided. Skipping audio generation.")
                else: # gemini_api_key was provided, but client initialization failed
                    st.warning("Gemini API client failed to initialize earlier. Skipping audio generation.")

            except Exception as e:
                st.error(f"An error occurred during report generation: {e}")
                st.exception(e)

st.markdown(
    """
    ---
    ### Disclaimer
    **WARNING: This demonstration does NOT provide medical advice, diagnosis, or treatment. Do NOT rely on any information or results from this tool for your health decisions. Misuse of this demo could lead to serious harm. Always consult a qualified healthcare professional for any medical concerns. If you are experiencing a medical emergency, contact your doctor or call emergency services immediately.**
    """
)

