import streamlit as st
import easyocr
from PIL import Image
from openai import OpenAI
from datetime import datetime, timedelta
import numpy as np
import json
import re
import gc
import logging
import time
import os
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

IS_DEPLOYED = os.environ.get('IS_DEPLOYED', 'false').lower() == 'true'

def get_api_key():
    return st.secrets["OPENAI_API_KEY"]

def init_openai_client():
    api_key = get_api_key()
    return OpenAI(api_key=api_key)

MODEL_NAME = "gpt-4"

SCOPES = ['https://www.googleapis.com/auth/calendar.events']
CLIENT_CONFIG = {
    "web": {
        "client_id": st.secrets["GOOGLE_CLIENT_ID"],
        "project_id": st.secrets["GOOGLE_PROJECT_ID"],
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_secret": st.secrets["GOOGLE_CLIENT_SECRET"],
        "redirect_uris": [
            "https://calstool.streamlit.app/",
            st.secrets.get("REDIRECT_URI", "https://calstool.streamlit.app/")
        ]
    }
}

def get_redirect_uri():
    if IS_DEPLOYED:
        return st.secrets.get("REDIRECT_URI")
    else:
        return "https://calstool.streamlit.app/"

def get_google_auth_flow():
    redirect_uri = get_redirect_uri()
    flow = Flow.from_client_config(
        CLIENT_CONFIG,
        scopes=SCOPES,
        redirect_uri=redirect_uri
    )
    return flow

def get_google_credentials():
    if 'google_token' not in st.session_state:
        return None
    return Credentials.from_authorized_user_info(st.session_state.google_token, SCOPES)

def credentials_to_dict(credentials):
    return {
        'token': credentials.token,
        'refresh_token': credentials.refresh_token,
        'token_uri': credentials.token_uri,
        'client_id': credentials.client_id,
        'client_secret': credentials.client_secret,
        'scopes': credentials.scopes
    }

@st.cache_resource
def load_ocr():
    try:
        with st.spinner('OCR 모델을 로딩 중입니다. 잠시만 기다려주세요...'):
            reader = easyocr.Reader(['ko', 'en'], gpu=False, model_storage_directory='./model')
            return reader
    except Exception as e:
        st.error(f"OCR 모델 로딩 중 오류 발생: {str(e)}")
        return None

def extract_text_from_image(image):
    reader = load_ocr()
    if reader is None:
        return None
    try:
        image_np = np.array(image)
        result = reader.readtext(image_np)
        text = ' '.join([res[1] for res in result])
        del image_np
        gc.collect()
        return text
    except Exception as e:
        st.error(f"이미지에서 텍스트 추출 중 오류 발생: {str(e)}")
        return None

def clean_json_string(json_string):
    json_string = re.sub(r'```json\s*|\s*```', '', json_string)
    json_string = json_string.strip()
    json_string = json_string[json_string.find('{'):json_string.rfind('}')+1]
    return json_string

def analyze_text_with_ai(client, text):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": """다음 텍스트에서 이벤트 정보를 추출해주세요. JSON 형식으로 다음 정보를 반환해주세요:

1. 주제: 이벤트의 주제 또는 제목
2. 일시: 'YYYY년 MM월 DD일 HH:MM' 형식으로 제공 (여러 개 가능, 배열로 반환)
3. 위치: 이벤트 장소 (구체적인 주소나 장소명 포함)
4. 설명: 이벤트에 대한 간단한 설명
5. 이벤트_유형: '신청', '참여', '참석' 중 하나 (해당되는 경우)
6. 알림_설정: 
   - '신청' 관련 내용이면 "이벤트 2일 전"
   - '참여' 또는 '참석' 관련 내용이면 "당일 오전 8시 45분"
   - 그 외의 경우 "기본 알림"

현재 연도를 기준으로 정보를 추출하세요. JSON만 반환하고 다른 텍스트는 포함하지 마세요."""},
                {"role": "user", "content": text}
            ]
        )
        return clean_json_string(completion.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"AI 분석 중 오류 발생: {str(e)}")
        return None

def create_google_calendar_event(event_info):
    if 'google_token' not in st.session_state:
        st.warning("Google 계정 연동이 필요합니다.")
        return None

    creds = get_google_credentials()
    if not creds:
        st.warning("Google 계정 연동이 필요합니다.")
        return None

    try:
        service = build('calendar', 'v3', credentials=creds)
        event_dict = json.loads(event_info)
        
        text = event_dict.get('주제', '')
        dates = event_dict.get('일시', [])
        location = event_dict.get('위치', '')
        details = event_dict.get('설명', '')
        reminder = event_dict.get('알림_설정', '기본 알림')

        if isinstance(dates, str):
            dates = [dates]

        created_events = []

        for date in dates:
            try:
                dt = datetime.strptime(date, "%Y년 %m월 %d일 %H:%M")
            except ValueError:
                dt = datetime.now() + timedelta(days=7)
            
            end_time = dt + timedelta(hours=1)

            event = {
                'summary': text,
                'location': location,
                'description': details,
                'start': {
                    'dateTime': dt.isoformat(),
                    'timeZone': 'Asia/Seoul',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'Asia/Seoul',
                },
            }

            if reminder == "이벤트 2일 전":
                event['reminders'] = {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'popup', 'minutes': 2 * 24 * 60},
                    ],
                }
            elif reminder == "당일 오전 8시 45분":
                minutes_until_reminder = int((dt.replace(hour=8, minute=45) - dt).total_seconds() / 60)
                if minutes_until_reminder < 0:
                    minutes_until_reminder = 0
                event['reminders'] = {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'popup', 'minutes': minutes_until_reminder},
                    ],
                }
            else:
                event['reminders'] = {'useDefault': True}

            created_event = service.events().insert(calendarId='primary', body=event).execute()
            created_events.append(created_event)

        return created_events
    except HttpError as error:
        if error.resp.status in [401, 403]:
            st.warning("Google 계정 연동이 필요합니다.")
        else:
            st.error(f'Google Calendar API 오류 발생: {error}')
        return None
    except Exception as e:
        st.error(f'예기치 못한 오류 발생: {str(e)}')
        return None

def main():
    st.title("공문 이미지를 Google 캘린더 이벤트로 변환")

    if 'google_token' not in st.session_state:
        auth_code = st.experimental_get_query_params().get("code")
        if auth_code:
            try:
                flow = get_google_auth_flow()
                flow.fetch_token(code=auth_code[0])
                credentials = flow.credentials
                st.session_state.google_token = credentials_to_dict(credentials)
                st.success("Google 계정 인증 성공!")
                time.sleep(2)
                st.rerun()
            except Exception as e:
                st.warning("Google 계정 연동이 필요합니다.")
                st.session_state.pop('google_token', None)
        else:
            st.warning("Google 계정 연동이 필요합니다.")
            if st.button("Google 계정 연동"):
                flow = get_google_auth_flow()
                authorization_url, _ = flow.authorization_url(prompt='consent', access_type='offline', include_granted_scopes='true')
                st.markdown(f"[Google 계정 인증하기]({authorization_url})")
    else:
        st.success("Google 계정이 연동되었습니다.")
        if st.button("Google 계정 연동 해제"):
            st.session_state.pop('google_token', None)
            st.experimental_rerun()

    api_key = get_api_key()
    if not api_key:
        st.error("OpenAI API 키가 설정되지 않았습니다. Streamlit Secrets에서 'OPENAI_API_KEY'를 설정해주세요.")
        return

    client = init_openai_client()

    uploaded_file = st.file_uploader("공문 이미지를 업로드하세요", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='업로드된 이미지', use_column_width=True)

        if st.button("이미지 분석 및 이벤트 생성"):
            with st.spinner('이미지를 분석 중입니다...'):
                try:
                    extracted_text = extract_text_from_image(image)
                    if extracted_text:
                        st.text("추출된 텍스트:")
                        st.text(extracted_text)
                        analyzed_info = analyze_text_with_ai(client, extracted_text)
                        if analyzed_info:
                            st.subheader("분석 결과")
                            st.json(analyzed_info)

                            created_events = create_google_calendar_event(analyzed_info)
                            if created_events:
                                st.subheader("생성된 Google 캘린더 이벤트")
                                for i, event in enumerate(created_events, 1):
                                    st.markdown(f"{i}. [이벤트 {i} 보기]({event.get('htmlLink')})")
                            else:
                                st.warning("Google 계정 연동이 필요합니다.")
                        else:
                            st.error("AI 분석에 실패했습니다.")
                    else:
                        st.error("이미지에서 텍스트를 추출하지 못했습니다.")
                except Exception as e:
                    st.error(f"이미지 처리 중 예기치 못한 오류 발생: {str(e)}")

if __name__ == "__main__":
    main()