from crewai import Agent, Task
from tools import llm, create_video_from_images_and_audio, image_generator, speech_generator

script_agent = Agent(
    role='Senior Content Writer',
    goal='Craft engaging, concise, and informative narrations for YouTube short videos',
    backstory="""As a seasoned content writer, you excel at breaking down complex topics into captivating narratives that educate and entertain audiences. Your expertise lies in writing concise, attention-grabbing scripts for YouTube short videos.""",
    verbose=True,
    llm=llm,
    allow_delegation=False
)

image_descriptive_agent = Agent(
    role='Visual Storyteller',
    goal='Design stunning, contextually relevant visuals for YouTube short videos',
    backstory='With a keen eye for visual storytelling, you create compelling imagery that elevates the narrative and captivates the audience. You will ',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

img_speech_generating_agent = Agent(
    role='Multimedia Content Creator',
    goal='Generate high-quality images and speeches for YouTube short videos one after another based on provided descriptions.',
    backstory='As a multimedia expert, you excel at creating engaging multimedia content that brings stories to life.',
    verbose=True,
    llm=llm,
    allow_delegation=False
)

editor = Agent(
    role = 'Video editor',
    goal = 'To make a video for YouTube shorts.',
    backstory = "You are a video editor working for a YouTube creator",
    verbose=True,
    llm=llm,
    allow_delegation = False,
    tools = [create_video_from_images_and_audio]
)

story_writing_task = Task(
    description='Write an engaging narration for a YouTube short video on the topic: {topic}',
    expected_output="""A short paragraph suitable for narrating in five seconds that provides an immersive experience to the audience. Follow the below example for output length and format.

    **Example input:**
    Ancient Wonders of the World

    **Output format:**
    Embark on a journey through time and marvel at the ancient wonders of the world! 
    From the majestic Great Pyramid of Giza, symbolizing the ingenuity of ancient Egypt, 
    to the Hanging Gardens of Babylon, an oasis of lush beauty amidst ancient Mesopotamia's arid landscape. 
    These remarkable structures continue to intrigue and inspire awe, reminding us of humanity's enduring quest for greatness.
    """,
    agent=script_agent
)


img_text_task = Task(
    description='Given the narration, visually describe each sentence in the narration which will be used as a prompt for image generation.',
    expected_output="""Sentences encoded in <narration> and <image> tags. Follow the example below for the output format.

    **Example input:**
    Embark on a journey through time and marvel at the ancient wonders of the world! From the majestic Great Pyramid of Giza, symbolizing the ingenuity of ancient Egypt, to the Hanging Gardens of Babylon, an oasis of lush beauty amidst ancient Mesopotamia's arid landscape. These remarkable structures continue to intrigue and inspire awe, reminding us of humanity's enduring quest for greatness.

    **Output format:**

    <narration>1. Embark on a journey through time and marvel at the ancient wonders of the world!<narration>
    <narration>2. From the majestic Great Pyramid of Giza, symbolizing the ingenuity of ancient Egypt,<narration>
    <narration>3. to the Hanging Gardens of Babylon, an oasis of lush beauty amidst ancient Mesopotamia's arid landscape,<narration>
    <narration>4. These remarkable structures continue to intrigue and inspire awe, reminding us of humanity's enduring quest for greatness.<narration>

    <image>1. A breathtaking view of various ancient wonders, showcasing their grandeur and mystery.<image>
    <image>2. The majestic Great Pyramid of Giza, standing tall against the desert backdrop, a testament to ancient engineering.<image>
    <image>3. The Hanging Gardens of Babylon, lush greenery cascading from terraced gardens, amidst the arid Mesopotamian landscape.<image>
    <image>4. Visitors captivated by the beauty and historical significance of these ancient marvels, exploring and marveling.<image>
    """,
    agent=image_descriptive_agent,
    context=[story_writing_task]
)

img_generation_task = Task(
    description='Given the input generate images for sequence of sentence enclosed in <image> tag.',
    expected_output="""Acknowledgement of image generation""",
    tools = [image_generator],
    context = [img_text_task],
    # async_execution=True,
    agent=img_speech_generating_agent
)

speech_generation_task = Task(
    description='Given the input generate speech for each sentence enclosed in <narration> tag.',
    expected_output="""Acknowledgement of speech generation""",
    tools = [speech_generator],
    context = [img_text_task],
    # async_execution=True,
    agent=img_speech_generating_agent
)

make_video_task = Task(
    description = 'Create video using images and speeches from the forlders "outpus/images" and "outputs/speeches"',
    expected_output = "output video path",
    agent=editor,
    context = [img_generation_task, speech_generation_task]
)