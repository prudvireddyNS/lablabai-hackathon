from langchain.tools import tool
import re
import os
from langchain_groq import ChatGroq

import cv2
from moviepy.editor import ImageClip, AudioFileClip, concatenate_videoclips
from langchain.pydantic_v1 import BaseModel, Field

from diffusers import StableDiffusionXLPipeline, DPMSolverSinglestepScheduler
import bitsandbytes as bnb
import torch.nn as nn
import torch
import pyttsx3
import os
# from langchain_google_genai import ChatGoogleGenerativeAI

# from langchain.chat_models import ChatOpenAI
# # llm2 = ChatOpenAI(model='gpt-3.5-turbo')
# # llm3 = ChatOpenAI(model='gpt-3.5-turbo')
# llm1 = ChatGroq(model='llama3-70b-8192', temperature=0.6, max_tokens=2048)
# # llm2 = ChatGroq(model='mixtral-8x7b-32768', temperature=0.6, max_tokens=2048, api_key='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')
# llm2 = ChatGoogleGenerativeAI(model='gemini-pro', temperature=0.0)
# # llm2 = ChatGroq(model='llama3-70b-8192', temperature=0.6, max_tokens=2048, api_key='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')
# # llm3 = ChatGoogleGenerativeAI(model='gemini-pro')
# llm4 = ChatGroq(model='llama3-70b-8192', temperature=0.6, max_tokens=2048, api_key='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')
llm = ChatGroq(model='llama3-70b-8192', temperature=0.6, max_tokens=2048)

pipe = StableDiffusionXLPipeline.from_pretrained("sd-community/sdxl-flash", torch_dtype=torch.float16).to('cuda')
pipe.scheduler = DPMSolverSinglestepScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")

def quantize_model_to_4bit(model):
    replacements = []

    # Collect layers to be replaced
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            replacements.append((name, module))

    # Replace layers
    for name, module in replacements:
        # Split the name to navigate to the parent module
        *path, last = name.split('.')
        parent = model
        for part in path:
            parent = getattr(parent, part)

        # Create and assign the quantized layer
        quantized_layer = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None)
        quantized_layer.weight.data = module.weight.data
        if module.bias is not None:
            quantized_layer.bias.data = module.bias.data
        setattr(parent, last, quantized_layer)

    return model

pipe.unet = quantize_model_to_4bit(pipe.unet)
pipe.enable_model_cpu_offload()

def generate_speech(text, speech_dir='./outputs/audio', lang='en', speed=170, voice='default', num=0):
    """
    Generates speech for given script.
    """
    engine = pyttsx3.init()
    
    # Set language and voice
    voices = engine.getProperty('voices')
    if voice == 'default':
        voice_id = voices[1].id
    else:
        # Try to find the voice with the given name
        voice_id = None
        for v in voices:
            if voice in v.name:
                voice_id = v.id
                break
        if not voice_id:
            raise ValueError(f"Voice '{voice}' not found.")
    
    engine.setProperty('voice', voice_id)
    engine.setProperty('rate', speed)
    os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)), speech_dir, f'speech_{num}.mp3')) if os.path.exists(os.path.join(speech_dir, f'speech_{num}.mp3')) else None
    engine.save_to_file(text, os.path.join(os.path.dirname(os.path.abspath(__file__)), speech_dir, f'speech_{num}.mp3'))
    engine.runAndWait()

class VideoGeneration(BaseModel):
    images_dir : str = Field(description='Path to images directory, such as "outputs/images"')
    speeches_dir : str = Field(description='Path to speeches directory, such as "outputs/speeches"')

@tool(args_schema=VideoGeneration)
def create_video_from_images_and_audio(images_dir, speeches_dir, zoom_factor=1.2):
    """Creates video using images and audios with zoom-in effect"""
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), images_dir)
    speeches_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), speeches_dir)

    images_paths = os.listdir(images_dir)
    audio_paths = os.listdir(speeches_dir)
    # print(images_paths, audio_paths)
    clips = []
    
    for i in range(min(len(images_paths), len(audio_paths))):
        # Load the image
        img_clip = ImageClip(os.path.join(images_dir, images_paths[i]))
        
        # Load the audio file
        audioclip = AudioFileClip(os.path.join(speeches_dir, audio_paths[i]))
        
        # Set the duration of the video clip to the duration of the audio file
        videoclip = img_clip.set_duration(audioclip.duration)
        
        # Apply zoom-in effect to the video clip
        zoomed_clip = apply_zoom_in_effect(videoclip, zoom_factor)
        
        # Add audio to the zoomed video clip
        zoomed_clip = zoomed_clip.set_audio(audioclip)
        
        clips.append(zoomed_clip)
    
    # Concatenate all video clips
    final_clip = concatenate_videoclips(clips)
    
    # Write the result to a file
    final_clip.write_videofile(os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/final_video/final_video.mp4"), codec='libx264', fps=24)
    
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs/final_video/final_video.mp4")

def apply_zoom_in_effect(clip, zoom_factor=1.2):
    width, height = clip.size
    duration = clip.duration

    def zoom_in_effect(get_frame, t):
        frame = get_frame(t)
        zoom = 1 + (zoom_factor - 1) * (t / duration)
        new_width, new_height = int(width * zoom), int(height * zoom)
        resized_frame = cv2.resize(frame, (new_width, new_height))
        
        # Calculate the position to crop the frame to the original size
        x_start = (new_width - width) // 2
        y_start = (new_height - height) // 2
        cropped_frame = resized_frame[y_start:y_start + height, x_start:x_start + width]
        
        return cropped_frame

    return clip.fl(zoom_in_effect, apply_to=['mask'])

# Example usage
# image_paths = "outputs/images"
# audio_paths = "outputs/audio"

# video_path = create_video_from_images_and_audio(image_paths, audio_paths)
# print(f"Video created at: {video_path}")


# class ImageGeneration(BaseModel):
#     text : str = Field(description='description of sentence used for image generation')
#     num : int = Field(description='sequence of description passed this tool. Used in image saving path. Example 1,2,3,4,5 and so on')

# class SpeechGeneration(BaseModel):
#     text : str = Field(description='description of sentence used for image generation')
#     num : int = Field(description='sequence of description passed this tool. Used in image saving path. Example 1,2,3,4,5 and so on')

# @tool
def process_script(script):
    """Used to process the script into dictionary format"""
    dict = {}
    dict['text_for_image_generation'] = re.findall(r'<image>(.*?)</?image>', script)
    dict['text_for_speech_generation'] = re.findall(r'<narration>.*?</?narration>', script)
    return dict

@tool#(args_schema=ImageGeneration)
def image_generator(script):
    """Generates images for the given script.
    Saves it to images_dir and return path
    Args:
    script: a complete script containing narrations and image descriptions"""
    images_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './outputs/images')
    # if num==1:
    for filename in os.listdir(images_dir):
        file_path = os.path.join(images_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    dict = process_script(script)
    for i, text in enumerate(dict['text_for_image_generation']):
        image = pipe(text, num_inference_steps=12, guidance_scale=2, width=720, height=1280, verbose=0).images[0]
        image.save(os.path.join(images_dir, f'image{i}.jpg'))
    return f'images generated.'#f'image generated for "{text}" and saved to directory {images_dir} as image{num}.jpg'

@tool
def speech_generator(script):
    """Generates speech for given text
    Saves it to speech_dir and return path
    Args:
    script: a complete script containing narrations and image descriptions"""
    speech_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), './outputs/speeches')

    # if num==1:
    for filename in os.listdir(speech_dir):
        file_path = os.path.join(speech_dir, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    dict = process_script(script)
    print(dict)
    for i, text in enumerate(dict['text_for_speech_generation']):
        generate_speech(text, speech_dir, num=i)
    return f'speechs generated.'#f'speech generated for "{text}" and saved to directory {speech_dir} as speech{num}.mp3'