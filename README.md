# Character and Comic Scene Generation Using Diffusion Models

This project is an AI-powered application for generating character images and multi-panel comic scenes from textual descriptions. Built with cutting-edge generative AI models like Stable Diffusion and CLIP, it bridges the gap between creativity and technology, enabling users to create high-quality artistic content efficiently.

## Features
- **Character Creation**: Generate detailed character images from text prompts.
- **Comic Scene Generation**: Create multi-panel comics with up to four customizable panels.
- **Interactive Interface**: Intuitive web application built with Streamlit for real-time input and output.
- **Customizability**: Diverse artistic styles supported through Hugging Face APIs and latent diffusion techniques.
- **Efficiency**: Combines local model computation with API-based flexibility for scalability.

## Technology Stack
- **Programming Language**: Python
- **Generative Models**: Stable Diffusion v1.5 & v2.0 (diffusers library)
- **Evaluation Model**: CLIP for text-to-image alignment and relevance scoring
- **Web Framework**: Streamlit
- **APIs**: Hugging Face for additional pre-trained models and style diversity
- **Image Processing**: Pillow (PIL) for assembling and formatting outputs

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/comic-scene-generator.git
   cd comic-scene-generator
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

4. Access the app in your browser at `http://localhost:8501
