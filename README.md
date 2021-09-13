## transcribe.py

Transcribe audio files using AWS Transcribe

Example usage:
```
find /path/to/audio/files/ -name "*.mp3" | parallel -j8 python transcribe.py --bucket my-s3-bucket-name {}
```