import json
import logging
import os
import sys
import time
import requests
import streamlit as st

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format="[%(asctime)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S %p %Z")
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Text to Avatar Video Synthesis",
    layout="centered",
    initial_sidebar_state="expanded"
)

# CSS for styling
st.markdown(
    """
    <style>
    .st-bf {
        background-color: #f0f0f0; /* Light gray background for main content */
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Soft shadow */
    }
    .st-ei {
        background-color: #ffffff; /* White background for sidebar */
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 0 10px rgba(0, 0, 0, 0.1); /* Soft shadow */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Information Pane
st.sidebar.title("Information")
st.sidebar.markdown(
    """
    <div class="st-ei">
    <p>To use this service, you need an active Azure Service with Text to Speech with Avatar enabled.</p>
    <hr style="border-top: 1px solid #ddd;"> 
    <p>Use regions: West US 2, West Europe, or Southeast Asia.</p>
    <hr style="border-top: 1px solid #ddd;"> 
    <p>Refer Azure Example: <a href="https://github.com/Azure-Samples/cognitive-services-speech-sdk/blob/yulin/batch-avatar/samples/batch-avatar/python/synthesis.py" target="_blank">Azure Avatar Synthesis Example</a></p>
    </div>
    """,
    unsafe_allow_html=True
)

# Input fields
SUBSCRIPTION_KEY = st.text_input("Enter your Speech API subscription key:", type="password")

# Predefined list of regions
regions = ["westus2", "westeurope", "southeastasia"]
SERVICE_REGION = st.selectbox("Select your Speech API service region:", regions)

NAME = "Simple avatar synthesis"
DESCRIPTION = "Simple avatar synthesis description"
SERVICE_HOST = "customvoice.api.speech.microsoft.com"


def submit_synthesis(text):
    """Submit the text synthesis job to Azure API."""
    if not (SERVICE_REGION and SUBSCRIPTION_KEY and text):
        st.error('Please fill in all required fields.')
        return None

    url = f'https://{SERVICE_REGION}.{SERVICE_HOST}/api/texttospeech/3.1-preview1/batchsynthesis/talkingavatar'
    headers = {
        'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY,
        'Content-Type': 'application/json'
    }

    payload = {
        'displayName': NAME,
        'description': DESCRIPTION,
        "textType": "PlainText",
        'synthesisConfig': {
            "voice": "en-US-JennyNeural",
        },
        'inputs': [{"text": text}],
        "properties": {
            "customized": False,
            "talkingAvatarCharacter": "lisa",
            "talkingAvatarStyle": "graceful-sitting",
            "videoFormat": "webm",
            "videoCodec": "vp9",
            "subtitleType": "soft_embedded",
            "backgroundColor": "transparent",
        }
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()

        job_id = response.json().get("id")
        if job_id:
            logger.info(f'Job submitted successfully: {job_id}')
            return job_id
        else:
            st.error('Failed to retrieve Job ID from response.')
            return None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error submitting synthesis: {e}")
        st.error(f"Failed to submit synthesis: {e}")
        return None


def get_synthesis(job_id):
    """Get the status of the synthesis job."""
    url = f'https://{SERVICE_REGION}.{SERVICE_HOST}/api/texttospeech/3.1-preview1/batchsynthesis/talkingavatar/{job_id}'
    headers = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        data = response.json()
        status = data.get('status')

        if status == 'Succeeded':
            outputs = data.get('outputs', {})
            download_url = outputs.get('result')

            if download_url:
                logger.info(f"Download URL: {download_url}")
                return 'Succeeded', download_url

        return status, None

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching job status: {e}")
        st.error(f"Failed to get job status: {e}")
        return 'Failed', None


def list_synthesis_jobs(skip=0, top=100):
    """List all batch synthesis jobs."""
    url = f'https://{SERVICE_REGION}.{SERVICE_HOST}/api/texttospeech/3.1-preview1/batchsynthesis/talkingavatar?skip={skip}&top={top}'
    headers = {'Ocp-Apim-Subscription-Key': SUBSCRIPTION_KEY}

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        jobs = response.json().get('values', [])
        logger.info(f'Fetched {len(jobs)} jobs.')
        return jobs

    except requests.exceptions.RequestException as e:
        logger.error(f"Error listing jobs: {e}")
        st.error(f"Failed to list jobs: {e}")
        return []


def main():
    st.title("Text to Avatar Video Synthesis")

    text = st.text_area("Enter the text to be synthesized into avatar video:", 
                        "Hi, I'm a virtual assistant created by Microsoft.")

    if st.button("Submit"):
        job_id = submit_synthesis(text)
        
        if job_id:
            st.success(f"Job submitted successfully. Job ID: {job_id}")
            st.info("Waiting for job completion...")

            status = None
            while status not in ['Succeeded', 'Failed']:
                status, download_url = get_synthesis(job_id)

                if status == 'Succeeded' and download_url:
                    st.success(f"Job succeeded! Download your video [here]({download_url})")
                    break
                elif status == 'Failed':
                    st.error("Job failed.")
                    break
                else:
                    st.info(f"Job status: {status}. Checking again in 5 seconds...")
                    time.sleep(5)

    st.header("List Batch Synthesis Jobs")
    jobs = list_synthesis_jobs()
    
    if jobs:
        st.write(f"Total jobs: {len(jobs)}")
        for job in jobs:
            st.write(job)


if __name__ == '__main__':
    main()
