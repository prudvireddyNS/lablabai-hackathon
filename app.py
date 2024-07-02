import gradio as gr
import os

from crewai import Crew, Process
from agents import *


def generate_video(topic):
    crew = Crew(
    agents=[script_agent, image_descriptive_agent, img_speech_generating_agent, editor],
    tasks=[story_writing_task, img_text_task, img_generation_task,speech_generation_task,make_video_task],
    process = Process.sequential,
    # cache = True,
    memory=True,
    verbose=2
    )
    crew.kickoff(inputs={'topic': topic})
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs/final_video/final_video.mp4')

app = gr.Interface(
    fn=generate_video,
    inputs='text',
    # outputs=gr.Video(value=os.path.join('outputs/final_video/video.mp4'),label="Generated Video", width=720/2, height=1280/2),
    outputs = gr.Video(format='mp4',label="Generated Video", width=720/2, height=1280/2),
    title="ShortsIn",
    description="Shorts generator"
)

app.launch(share=True)
#os.path.dirname(os.path.abspath(__file__))