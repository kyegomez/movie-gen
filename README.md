[![Multi-Modality](agorabanner.png)](https://discord.com/servers/agora-999382051935506503)

# Movie Gen

[![Join Their Discord](https://img.shields.io/badge/Discord-Join%20Their%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


**Movie Gen** is a collection of cutting-edge foundation models developed by the Movie Gen team at **Meta1**. These models are designed to generate high-quality 1080p HD videos with various aspect ratios and synchronized audio. **Movie Gen** excels at a range of tasks, including:

- **Text-to-video synthesis**
- **Video personalization**
- **Precise video editing based on user instructions**
- **Video-to-audio generation**
- **Text-to-audio generation**

Their models set a new state-of-the-art in multiple video and audio generation domains and aim to push the boundaries of what's possible in media creation. The most powerful model in Their collection is a **30 billion parameter transformer**, capable of generating videos up to 16 seconds long at 16 frames per second (fps). The model operates with a maximum context length of **73K video tokens**, allowing for highly detailed and complex media output.

## Key Features

- **HD Video Generation**: Outputs high-quality 1080p videos with various aspect ratios and synchronized audio.
- **Text-to-Video Synthesis**: Generates fully realized videos from natural language descriptions.
- **Personalized Video Creation**: Tailors videos based on user-supplied images or inputs.
- **Instruction-Based Video Editing**: Allows precise control and editing of video content through instructions.
- **Audio Synthesis**: Generates audio based on video content and natural language descriptions.
- **Scaling & Efficiency**: Achieves high scalability through technical innovations in parallelization, architecture simplifications, and efficient data curation.

## Model Overview

| **Model**                    | **Parameters** | **Capabilities**                              | **Max Context Length** | **FPS** |
|------------------------------|----------------|-----------------------------------------------|------------------------|---------|
| **Movie Gen Base**            | 5B             | Text-to-Video, Video-to-Audio                 | 18K video tokens        | 16      |
| **Movie Gen Pro**             | 15B            | Personalized Video, Text-to-Video             | 40K video tokens        | 16      |
| **Movie Gen Max** (State-of-the-art) | 30B        | Full-featured Video & Audio Generation        | 73K video tokens        | 16      |

## Technical Innovations

1. **Architecture Simplifications**: They introduced several architectural simplifications to scale media generation models effectively. These include novel transformer-based structures tailored for handling video data.
   
2. **Latent Spaces & Training Objectives**: By refining latent spaces and optimizing training objectives, Their models can generate realistic, coherent, and high-quality outputs across multiple media modalities.

3. **Data Curation**: They built a highly curated, diverse dataset specifically designed for multi-modal media generation tasks.

4. **Parallelization Techniques**: Their models leverage advanced parallelization techniques, enabling faster training and inference.

5. **Inference Optimizations**: They implemented optimizations that significantly reduce latency during inference, making real-time video generation and editing feasible.

## Installation

To use Movie Gen, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/kyegomez/movie-gen.git
cd movie-gen
pip install -r requirements.txt
```
<!-- 
## Usage

### Text-to-Video Generation

To generate a video from text, use the following script:

```python
from movie_gen import MovieGenModel

# Initialize the model
model = MovieGenModel.load_pretrained('movie-gen-max')

# Provide yTheir text prompt
text_prompt = "A serene beach at sunset with waves crashing gently."

# Generate video
generated_video = model.generate_video(text_prompt)

# Save the video
generated_video.save("beach_sunset.mp4")
```

### Personalized Video Creation

You can create personalized videos by providing an image of the subject:

```python
from movie_gen import MovieGenModel

# Initialize the model
model = MovieGenModel.load_pretrained('movie-gen-pro')

# Provide a user image and text prompt
user_image = "path_to_user_image.jpg"
text_prompt = "A person running in a scenic mountain range."

# Generate personalized video
personalized_video = model.generate_personalized_video(user_image, text_prompt)

# Save the video
personalized_video.save("personalized_video.mp4")
```

### Video Editing with Instructions

To edit an existing video based on instructions, use the script below:

```python
from movie_gen import MovieGenModel

# Initialize the model
model = MovieGenModel.load_pretrained('movie-gen-max')

# Provide a video and edit instructions
video_path = "path_to_existing_video.mp4"
edit_instructions = "Change the sky to a bright pink sunset."

# Perform video editing
edited_video = model.edit_video(video_path, edit_instructions)

# Save the edited video
edited_video.save("edited_video.mp4")
```

## Model Training

To train yTheir own version of the Movie Gen models, follow the steps below:

1. Prepare yTheir dataset following Their data curation guidelines.
2. Run the training script:
   ```bash
   python train.py --config configs/movie_gen_max.yaml
   ``` -->

<!-- ### Training Configurations

The available model configurations are stored in the `configs/` directory. For example, to train the 30B parameter model (`movie-gen-max`), use the following configuration:

```yaml
model:
  name: movie-gen-max
  parameters: 30B
  context_length: 73K
  fps: 16
  tasks:
    - text-to-video
    - video-personalization
    - video-editing
``` -->


## Usage


### `TemporalAutoencoder` or `TAE`

```python
import torch
from loguru import logger
from movie_gen.tae import TemporalAutoencoder

def test_temporal_autoencoder():
    """
    Test the TemporalAutoencoder model with a dummy input tensor.
    This function creates a random input tensor representing a batch of videos,
    passes it through the model, and prints out the input and output shapes.
    """
    # Set the logger to display debug messages
    logger.add(lambda msg: print(msg, end=''))

    # Instantiate the model
    model = TemporalAutoencoder(in_channels=3, latent_channels=16)

    # Create a dummy input tensor representing a batch of videos
    # Batch size B=2, T0=16 frames, 3 channels (RGB), H0=64, W0=64
    B, T0, C_in, H0, W0 = 1, 16, 3, 64, 64
    x = torch.randn(B, T0, C_in, H0, W0)

    # Forward pass through the model
    recon = model(x)

    # Print the shapes
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed output shape: {recon.shape}")

if __name__ == "__main__":
    test_temporal_autoencoder()


```

## Evaluation

Their models have been rigorously evaluated on multiple tasks, including:

- Text-to-video generation benchmarks.
- Video personalization accuracy.
- Instruction-based video editing precision.
- Audio generation quality.

### Reproducing Their Results

To reproduce the evaluation metrics in Their paper, use the following command:

```bash
python evaluate.py --model movie-gen-max --task text-to-video
```

## Contributing

They welcome contributions! Please follow the standard GitHub flow:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature-branch`)
3. Make your changes
4. Submit a pull request

For a list of core contributors, please refer to the appendix of the [Movie Gen Paper](link_to_paper).

## License

Movie Gen is licensed under the MIT License. See `LICENSE` for more information.

## Contact

For any questions or collaboration opportunities, please reach out to the **Movie Gen** team at:

- **Email**: http://agoralab.ai
- **Website**: [http://agoralab.ai](http://agoralab.ai)