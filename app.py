import gradio as gr
import os

from crewai import Crew, Process
# from agents import *
from agents import get_agents_and_tasks


def generate_video(topic, grow_api_key, stability_ai_api_key,openai_api_key):
    # os.environ['GROQ_API_KEY'] = grow_api_key
    os.environ['STABILITY_AI_API_KEY'] = stability_ai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    agents, tasks = get_agents_and_tasks(grow_api_key)

    crew = Crew(
    # agents=[script_agent, image_descriptive_agent, img_speech_generating_agent, editor],
    # tasks=[content_generation_task, story_writing_task, img_text_task, img_generation_task,speech_generation_task,make_video_task],
    agents = agents, 
    tasks = tasks,
    process = Process.sequential,
    # cache = True,
    memory=True,
    verbose=2
    )
    crew.kickoff(inputs={'topic': topic})
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs/final_video/final_video.mp4')

app = gr.Interface(
    fn=generate_video,
    inputs=['text', 'text', 'text', 'text'],
    # outputs=gr.Video(value=os.path.join('outputs/final_video/video.mp4'),label="Generated Video", width=720/2, height=1280/2),
    outputs = gr.Video(format='mp4',label="Generated Video", width=720/2, height=1280/2),
    title="ShortsIn",
    description="Shorts generator"
)

app.launch(share=True)
#os.path.dirname(os.path.abspath(__file__))