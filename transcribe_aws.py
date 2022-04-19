import argparse
import json
import os
from time import sleep, time
import boto3

import requests

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Transcribe audio files")
    parser.add_argument("input_file", nargs=1, help="Audio file to transcribe")
    parser.add_argument("--bucket", dest="bucket", action='store', help="S3 bucket name", required=True)
    args = parser.parse_args()

    INPUT_FILE = args.input_file[0]
    BUCKET = args.bucket

    if os.path.exists(INPUT_FILE + ".s2meta"):
        exit(0)

    transcribe = boto3.client("transcribe")
    s3 = boto3.client("s3")

    job_name = "sist2-transcribe-%d" % int(time())

    s3.upload_file(INPUT_FILE, BUCKET, job_name)

    job_uri = "s3://%s/%s" % (BUCKET, job_name)

    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": job_uri},
        MediaFormat=os.path.splitext(INPUT_FILE)[1][1:],
        LanguageCode="en-US",
        Settings={
            "VocabularyFilterMethod": "tag"
        }
    )

    while True:
        status = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        if status["TranscriptionJob"]["TranscriptionJobStatus"] in ["COMPLETED", "FAILED"]:
            break
        sleep(5)

    transcript_url = status["TranscriptionJob"]["Transcript"]["TranscriptFileUri"]
    r = requests.get(transcript_url)
    transcript = r.json()["results"]["transcripts"][0]["transcript"]

    s3.delete_object(Bucket=BUCKET, Key=job_name)
    transcribe.delete_transcription_job(TranscriptionJobName=job_name)

    with open(INPUT_FILE + ".s2meta", "w") as f:
        f.write(json.dumps({
            "content": transcript,
            "_transcribed_by": "AWS/Transcribe"
        }))
