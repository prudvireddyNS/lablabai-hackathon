import streamlit as st
import os
from crewai import Crew, Process
from agents import get_agents_and_tasks

# Define your function to generate the video
def generate_video(topic, grow_api_key, stability_ai_api_key, openai_api_key):
    os.environ['STABILITY_AI_API_KEY'] = stability_ai_api_key
    os.environ['OPENAI_API_KEY'] = openai_api_key
    
    # Retrieve agents and tasks using your function
    agents, tasks = get_agents_and_tasks(grow_api_key)
    
    # Initialize Crew object
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        memory=True,
        verbose=2
    )
    
    # Kick off the Crew with the specified topic
    crew.kickoff(inputs={'topic': topic})
    
    # Return the path to the generated video
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs/final_video/final_video.mp4')

# Streamlit app setup
def main():
    st.title('ShortsIn')
    st.subheader('Shorts generator')
    
    # Input fields
    topic = st.text_input('Topic')
    grow_api_key = st.text_input('Grow API Key')
    stability_ai_api_key = st.text_input('Stability AI API Key')
    openai_api_key = st.text_input('OpenAI API Key')
    
    # Generate video button
    if st.button('Generate Video'):
        if topic and grow_api_key and stability_ai_api_key and openai_api_key:
            # Generate the video
            video_path = generate_video(topic, grow_api_key, stability_ai_api_key, openai_api_key)
            
            # Display the generated video
            st.video(video_path, format='video/mp4', start_time=0)
        else:
            st.warning('Please fill in all the input fields.')

if __name__ == '__main__':
    main()
