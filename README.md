# ShortsIn - AI-Powered Short Video Generator

Transform your ideas into captivating short videos with the power of AI


## 📺 Overview

ShortsIn is an innovative AI-powered platform designed to automatically generate engaging short-form videos. By leveraging cutting-edge artificial intelligence and machine learning technologies, ShortsIn transforms raw content, scripts, or concepts into polished, publication-ready short videos optimized for platforms like TikTok, Instagram Reels, YouTube Shorts, and more.

### Key Features

- **🎬 Intelligent Video Generation**: Automatically create videos from text, scripts, or prompts
- **🎨 Smart Visual Design**: AI-powered visual composition and scene generation
- **🔊 Dynamic Audio Processing**: Voice synthesis and music integration
- **⚡ Fast Processing**: Generate videos in minutes, not hours
- **📱 Platform Optimization**: Auto-formatted for all major short-video platforms
- **🎭 Style Customization**: Multiple visual themes and creative styles
- **🌍 Multi-language Support**: Generate videos in various languages
- **📊 Analytics Integration**: Track video performance and engagement

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- Git for version control
- API keys for AI services (as configured)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/prudvireddyNS/lablabai-hackathon.git
   cd lablabai-hackathon
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and configuration
   ```

5. **Run the application**
   ```bash
   python main.py
   ```

## 💻 Usage

### Basic Example

```python
from shortsin import VideoGenerator

# Initialize the generator
generator = VideoGenerator(api_key="your_api_key")

# Generate a video from text
video = generator.create_video(
    prompt="A fun tutorial on making the perfect coffee",
    duration=30,  # 30 seconds
    style="energetic",
    language="en"
)

# Export the video
video.export("output/coffee_tutorial.mp4")
```

### Advanced Usage

```python
from shortsin import VideoGenerator, VideoConfig

config = VideoConfig(
    platform="tiktok",
    aspect_ratio="9:16",
    music_genre="upbeat",
    subtitle_style="modern",
    color_palette="vibrant"
)

generator = VideoGenerator(api_key="your_api_key", config=config)

# Generate with custom parameters
video = generator.create_video(
    script="Welcome to our channel...",
    visual_theme="minimalist",
    transitions="smooth",
    effects_intensity="medium"
)
```

## 📁 Project Structure

```
lablabai-hackathon/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── generators/
│   │   ├── __init__.py
│   │   ├── video_generator.py
│   │   ├── audio_processor.py
│   │   └── visual_renderer.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── ai_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── helpers.py
│   └── api/
│       ├── __init__.py
│       └── endpoints.py
├── tests/
│   ├── __init__.py
│   ├── test_generator.py
│   └── test_utils.py
├── docs/
│   ├── API.md
│   ├── INSTALLATION.md
│   └── EXAMPLES.md
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

## 🛠️ Technologies Used

- **Python 3.8+**: Core programming language
- **OpenAI API**: Advanced AI model integration
- **FFmpeg**: Video processing and encoding
- **MoviePy**: Video composition and editing
- **Pyttsx3/gTTS**: Text-to-speech synthesis
- **PIL/OpenCV**: Image processing
- **FastAPI**: Web API framework (optional)
- **PyTorch**: Deep learning capabilities

## 🔧 Configuration

Create a `.env` file in the project root:

```env
# API Keys
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
VIDEO_OUTPUT_DIR=./output
MAX_VIDEO_DURATION=120
DEFAULT_PLATFORM=tiktok

# Processing Settings
ENABLE_GPU=True
NUM_WORKERS=4
CACHE_ENABLED=True
```

## 📚 API Documentation

For detailed API documentation, see [API.md](docs/API.md)

### Endpoint Example

```
POST /api/v1/generate
Content-Type: application/json

{
  "prompt": "Create a trending video about...",
  "duration": 30,
  "style": "energetic",
  "platform": "tiktok"
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_generator.py -v
```
