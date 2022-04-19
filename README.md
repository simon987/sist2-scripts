Create conda env with:

```
conda create -y -n sist2-scripts -c conda-forge python=3.7 cudnn=8.1 cudatoolkit=11.2
conda clean --force-pkgs-dirs -y && conda clean --all -y
conda activate sist2-scripts
pip install -r requirements.txt
```

## transcribe.py

Transcribe audio files using transformers STT

Example usage (Don't use multithreading!!):

```
find /path/to/audio/files/ -name "*.mp3" -exec python transcribe.py {} \;
```

## transcribe_aws.py

Transcribe audio files using AWS Transcribe

Example usage:

```
find /path/to/audio/files/ -name "*.mp3" | parallel -j8 python transcribe_aws.py --bucket my-s3-bucket-name {}
```

## export_meta.py

Save all .s2meta files to a zip archive for easy sharing

Example usage:

```
python export_meta.py [--json] /path/to/dataset/
```
