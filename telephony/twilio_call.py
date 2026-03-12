"""
twilio_call.py
--------------
Handles outbound Twilio call initiation and TwiML generation.
"""

import os
import logging
from typing import Optional, Dict

from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

logger = logging.getLogger(__name__)


class TwilioCallManager:
    """
    Manages outbound Twilio calls and TwiML for media streams.
    """

    def __init__(self):
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER")
        self.ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
        self.record_calls = os.getenv("RECORD_CALLS", "true").lower() == "true"

        if not all([account_sid, auth_token, self.from_number]):
            raise ValueError(
                "Missing Twilio credentials. Check TWILIO_ACCOUNT_SID, "
                "TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER in your .env"
            )

        self.client = Client(account_sid, auth_token)
        logger.info(f"Twilio initialized. Calling from: {self.from_number}")

    def make_outbound_call(self, to_number: str, session_id: str) -> str:
        """
        Place an outbound call. When answered, Twilio fetches /twiml.

        Returns:
            Twilio Call SID
        """
        if not self.ngrok_url:
            raise ValueError("NGROK_URL must be set in your .env file.")

        twiml_url = f"{self.ngrok_url}/twiml?session_id={session_id}"
        status_url = f"{self.ngrok_url}/call-status?session_id={session_id}"

        logger.info(f"Calling {to_number} | session: {session_id}")
        logger.info(f"TwiML URL: {twiml_url}")

        call_kwargs = dict(
            to=to_number,
            from_=self.from_number,
            url=twiml_url,
            status_callback=status_url,
            status_callback_event=["initiated", "ringing", "answered", "completed"],
            status_callback_method="POST",
            
        )

        # Recording (optional)
        if self.record_calls:
            call_kwargs["record"] = True
            call_kwargs["recording_status_callback"] = f"{self.ngrok_url}/recording-ready"
            call_kwargs["recording_status_callback_method"] = "POST"

        call = self.client.calls.create(**call_kwargs)
        logger.info(f"Call placed: SID={call.sid} | Status={call.status}")
        return call.sid

    def generate_twiml(self, session_id: str) -> str:
        """
        Generate TwiML that connects the call to our WebSocket media stream.

        Returns:
            TwiML XML string
        """
        ngrok_url = os.getenv("NGROK_URL", "").rstrip("/")
        # Convert https:// to wss:// for WebSocket
        ws_base = ngrok_url.replace("https://", "wss://").replace("http://", "ws://")
        websocket_url = f"{ws_base}/ws/{session_id}"

        response = VoiceResponse()
        response.pause(length=1)  # Give WebSocket time to connect

        connect = Connect()
        stream = Stream(url=websocket_url)
        stream.parameter(name="session_id", value=session_id)
        connect.append(stream)
        response.append(connect)

        return str(response)

    def hangup_call(self, call_sid: str):
        """Terminate an active call."""
        try:
            self.client.calls(call_sid).update(status="completed")
            logger.info(f"Call {call_sid} hung up.")
        except Exception as e:
            logger.error(f"Hangup failed for {call_sid}: {e}")

    def get_recording_url(self, call_sid: str) -> Optional[str]:
        """Retrieve recording URL after call completes."""
        try:
            recordings = self.client.recordings.list(call_sid=call_sid, limit=1)
            if recordings:
                rec = recordings[0]
                return f"https://api.twilio.com{rec.uri.replace('.json', '.mp3')}"
            return None
        except Exception as e:
            logger.error(f"Failed to get recording for {call_sid}: {e}")
            return None

    def download_recording(self, call_sid: str, output_path: str) -> bool:
        """Download call recording MP3 to local path."""
        import requests
        from requests.auth import HTTPBasicAuth

        url = self.get_recording_url(call_sid)
        if not url:
            logger.warning(f"No recording for call {call_sid}")
            return False

        try:
            resp = requests.get(
                url,
                auth=HTTPBasicAuth(
                    os.getenv("TWILIO_ACCOUNT_SID"),
                    os.getenv("TWILIO_AUTH_TOKEN"),
                ),
                timeout=60,
            )
            resp.raise_for_status()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(resp.content)
            logger.info(f"Recording saved: {output_path}")
            return True
        except Exception as e:
            logger.error(f"Recording download failed: {e}")
            return False
