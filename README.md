# RawNet2 Deepfake Audio Detection

This script uses the RawNet2 model from the AASIST package to detect whether an audio file is real (bonafide) or fake/AI-generated (spoofed).

## Prerequisites

Make sure you have the following packages installed:
- torch
- torchaudio
- numpy

## Usage

1. Place your audio file in an accessible location
2. Edit the `test_rawnet2.py` script to point to your audio file:
   ```python
   audio_file = 'path/to/your/audio_sample.wav'  # Change this to your audio file path
   ```
3. Run the script:
   ```bash
   python test_rawnet2.py
   ```

## Expected Output

The script will output:
- Whether the audio is predicted to be real or fake
- The probability scores for both real and fake classifications

## Supported Audio Formats

The script supports various audio formats including:
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- OGG (.ogg)

## Notes

- The model expects 16kHz mono audio. The script will automatically convert other formats.
- For optimal results, use high-quality audio recordings.
- The model has been trained on specific types of spoofing attacks, so results may vary with different generation methods.

## Sample Audio Files

You can test the model with:
- Real human speech recordings
- AI-generated speech from tools like ElevenLabs, Google Text-to-Speech, etc.
- Deepfake voice clones
- Voice conversions

## Troubleshooting

If you encounter any issues:
1. Make sure all dependencies are installed
2. Check that the audio file path is correct
3. Ensure the AASIST package is properly installed with all model files
