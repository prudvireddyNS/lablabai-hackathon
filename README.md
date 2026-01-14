# ShortsIn - AI-Powered Short Video Generator

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-brightgreen)](https://github.com/prudvireddyNS/lablabai-hackathon)

Transform your ideas into captivating short videos with the power of AI

</div>

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

## 🌟 Features in Development

- [ ] Real-time video preview
- [ ] Batch video generation
- [ ] Custom music library integration
- [ ] Advanced subtitle styling
- [ ] Social media auto-posting
- [ ] Analytics dashboard
- [ ] Team collaboration tools
- [ ] Web-based UI

## 📊 Performance

- Average generation time: 2-5 minutes for 30-second videos
- Supported resolutions: Up to 4K
- Platform support: TikTok, Instagram Reels, YouTube Shorts, Snapchat
- Concurrent video generation: Up to 10 simultaneous videos

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and development process.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LabLab.ai](https://lablab.ai) for the hackathon platform
- OpenAI for GPT models and AI capabilities
- The open-source community for incredible tools and libraries
- All contributors and supporters of this project

## 📧 Contact & Support

- **Email**: [your-email@example.com]
- **GitHub Issues**: [Report bugs here](https://github.com/prudvireddyNS/lablabai-hackathon/issues)
- **Discussions**: [Join our community](https://github.com/prudvireddyNS/lablabai-hackathon/discussions)

## 🗺️ Roadmap

See the [open issues](https://github.com/prudvireddyNS/lablabai-hackathon/issues) for a list of proposed features and known issues.

### Q1 2026 Goals
- [ ] v1.0 release with core features
- [ ] Web dashboard launch
- [ ] Multi-language expansion

### Q2 2026 Goals
- [ ] Mobile app development
- [ ] Advanced analytics suite
- [ ] Enterprise features

---

<div align="center">

**Made with ❤️ by the ShortsIn Team**

[Star us on GitHub](https://github.com/prudvireddyNS/lablabai-hackathon) ⭐

</div>
