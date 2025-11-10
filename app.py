import json
import re
import mimetypes
from uuid import uuid4
from datetime import datetime

import streamlit as st
import boto3
from botocore.config import Config
from streamlit_cognito_auth import CognitoAuthenticator

import time
import io
from streamlit_mic_recorder import mic_recorder


# Ensure Markdown mimetype is registered
mimetypes.add_type("text/markdown", ".md")

st.markdown("""
<style>
.search-row { display:flex; gap:.5rem; align-items:center; }
.search-row input { height:42px; border-radius:10px; }
.mic-btn button { padding:0 .65rem !important; height:42px; border-radius:10px; }
.badge {display:inline-block;padding:4px 10px;border-radius:999px;font-weight:600;}
.badge-green{background:#16a34a;color:white;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
button[kind="primary"] {
    margin-top: .5rem;
}
</style>
""", unsafe_allow_html=True)



# =========================
# Config / Secrets
# =========================
AWS_REGION = st.secrets.get("AWS_REGION", "ap-southeast-2")

# Required secrets (set in .streamlit/secrets.toml locally or Streamlit Cloud Secrets UI)
pool_id            = st.secrets["COGNITO_POOL_ID"]
app_client_id      = st.secrets["COGNITO_APP_CLIENT_ID"]
# app_client_secret  = st.secrets["COGNITO_APP_CLIENT_SECRET"]

kb_id               = st.secrets["KB_ID"]
lambda_function_arn = st.secrets["LAMBDA_FUNCTION_ARN"]
dynamo_table        = st.secrets["DYNAMO_TABLE"]

# Optional: static AWS creds if not using IAM role
aws_access_key_id     = st.secrets.get("AWS_ACCESS_KEY_ID")
aws_secret_access_key = st.secrets.get("AWS_SECRET_ACCESS_KEY")
aws_session_token     = st.secrets.get("AWS_SESSION_TOKEN")

# =========================
# AWS Session & Clients
# =========================
boto_cfg = Config(
    retries={"max_attempts": 10, "mode": "adaptive"},
    read_timeout=60,
    connect_timeout=10,
)

session = boto3.Session(
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    aws_session_token=aws_session_token,  # optional
    region_name=AWS_REGION,
)

s3             = session.client("s3", config=boto_cfg)
bedrock_agent  = session.client("bedrock-agent", config=boto_cfg)
cognito_idp    = session.client("cognito-idp", config=boto_cfg)
lambda_client  = session.client("lambda", config=boto_cfg)
ddb            = session.client("dynamodb", config=boto_cfg)

# =========================
# Auth
# =========================
authenticator = CognitoAuthenticator(
    pool_id=pool_id,
    app_client_id=app_client_id,
    # app_client_secret=app_client_secret,
    use_cookies=False
)

is_logged_in = authenticator.login()
if not is_logged_in:
    st.stop()

def logout():
    authenticator.logout()

# =========================
# Helpers
# =========================
def get_user_sub(user_pool_id: str, username: str):
    try:
        response = cognito_idp.admin_get_user(
            UserPoolId=pool_id,
            Username=authenticator.get_username()
        )
        for attr in response.get("UserAttributes", []):
            if attr.get("Name") == "sub":
                return attr.get("Value")
        return None
    except cognito_idp.exceptions.UserNotFoundException:
        print("User not found.")
        return None

def get_user_sub_from_token() -> str | None:
    """
    Prefer reading 'sub' from the ID token claims (no admin permissions needed).
    Falls back to AdminGetUser only if tokens are unavailable and IAM allows it.
    """
    # 1) Try to read from ID token
    try:
        id_token = getattr(authenticator, "get_id_token", lambda: None)()
        if id_token:
            import jwt  # PyJWT
            claims = jwt.decode(id_token, options={"verify_signature": False})
            return claims.get("sub")
    except Exception:
        pass  # fall through to optional admin call

    # 2) Optional fallback: AdminGetUser (requires IAM perms)
    try:
        username = authenticator.get_username()
        if not username:
            return None
        resp = cognito_idp.admin_get_user(UserPoolId=pool_id, Username=username)
        for attr in resp.get("UserAttributes", []):
            if attr.get("Name") == "sub":
                return attr.get("Value")
        return None
    except Exception:
        return None


def get_patient_ids(doctor_id: str):
    resp = ddb.query(
        TableName=dynamo_table,
        KeyConditionExpression='doctor_id = :doctor_id',
        ExpressionAttributeValues={
            ':doctor_id': {'S': doctor_id}
        }
    )
    items = resp.get("Items", [])
    out = []
    for item in items:
        plist = item.get('patient_id_list', {}).get('L', [])
        out.extend([p['S'] for p in plist])
    return out

def _safe_name(name: str) -> str:
    name = re.sub(r'[^A-Za-z0-9._-]+', '-', name)
    return name[:200]

def _build_unique_key(patient_id: str, filename: str) -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    uid = str(uuid4())[:8]
    return f"{patient_id}__{ts}-{uid}__{_safe_name(filename)}"

def add_patient_to_doctor(table_name: str, doctor_id: str, new_patient_id: str):
    """Append a patient ID to the doctor's patient_id_list (no dupes)."""
    cur = ddb.get_item(TableName=table_name, Key={"doctor_id": {"S": doctor_id}})
    existing = [p["S"] for p in cur.get("Item", {}).get("patient_id_list", {}).get("L", [])]

    if new_patient_id in existing:
        return "exists"

    ddb.update_item(
        TableName=table_name,
        Key={"doctor_id": {"S": doctor_id}},
        UpdateExpression="SET patient_id_list = list_append(if_not_exists(patient_id_list, :empty), :p)",
        ExpressionAttributeValues={
            ":empty": {"L": []},
            ":p": {"L": [{"S": new_patient_id}]}
        }
    )
    return "added"

def search_transcript(doctor_id: str, kb_id: str, text: str, patient_ids: list[str]):
    payload = {
        "doctorId": doctor_id,
        "knowledgeBaseId": kb_id,
        "text": text,
        "patientIds": patient_ids
    }

    try:
        resp = lambda_client.invoke(
            FunctionName=lambda_function_arn,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload).encode('utf-8')
        )

        if resp.get('StatusCode') != 200:
            return {"error": f"Lambda invoke failed with StatusCode={resp.get('StatusCode')}"}

        raw = resp['Payload'].read().decode('utf-8')

        # Lambda can return a JSON string or already-JSON; normalize:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"body": raw}

        if isinstance(data, dict):
            if "body" in data:
                body = data["body"]
                try:
                    body_json = json.loads(body)
                    if isinstance(body_json, dict):
                        if "html" in body_json:
                            return {"body": body_json["html"]}
                        if "message" in body_json:
                            return {"body": body_json["message"]}
                        return {"body": "<pre>" + json.dumps(body_json, indent=2) + "</pre>"}
                    else:
                        return {"body": str(body_json)}
                except Exception:
                    return {"body": body}
            if "html" in data:
                return {"body": data["html"]}
            if "results" in data:
                return {"body": "<pre>" + json.dumps(data["results"], indent=2) + "</pre>"}
            if "error" in data:
                return {"error": data["error"]}

        return {"body": "<pre>" + json.dumps(data, indent=2) + "</pre>"}

    except Exception as e:
        return {"error": str(e)}

def _get_kb_data_source(kb_id: str):
    """Find the first data source and resolve its S3 bucket name."""
    ds = bedrock_agent.list_data_sources(knowledgeBaseId=kb_id).get("dataSourceSummaries", [])
    if not ds:
        raise RuntimeError("No data source found for Knowledge Base.")
    data_source_id = ds[0]["dataSourceId"]
    detail = bedrock_agent.get_data_source(knowledgeBaseId=kb_id, dataSourceId=data_source_id)["dataSource"]
    s3cfg = detail["dataSourceConfiguration"]["s3Configuration"]
    bucket_arn = s3cfg["bucketArn"]  # "arn:aws:s3:::bucket-name"
    bucket_name = bucket_arn.split(":::")[-1]
    return data_source_id, bucket_name

def _guess_content_type(key: str) -> str:
    ctype, _ = mimetypes.guess_type(key)
    return ctype or "application/octet-stream"

def _put_file_and_metadata(bucket: str, filename: str, file_bytes: bytes, patient_id: str):
    key = _build_unique_key(patient_id, filename)

    # 1) Upload the source file
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=file_bytes,
        ContentType=_guess_content_type(filename)
    )

    # 2) Upload metadata JSON next to it
    meta_key = f"{key}.metadata.json"
    metadata_doc = {"metadataAttributes": {"patient_id": patient_id}}
    s3.put_object(
        Bucket=bucket,
        Key=meta_key,
        Body=json.dumps(metadata_doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8"),
        ContentType="application/json"
    )
    return key, meta_key

def _start_ingestion(kb_id: str, data_source_id: str) -> str:
    resp = bedrock_agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=data_source_id,
        clientToken=str(uuid4())
    )
    return resp["ingestionJob"]["ingestionJobId"]

def _latest_ingestion_status(kb_id: str, data_source_id: str):
    jobs = bedrock_agent.list_ingestion_jobs(
        knowledgeBaseId=kb_id,
        dataSourceId=data_source_id
    ).get("ingestionJobSummaries", [])
    if not jobs:
        return "NO_JOBS", None, None
    jobs.sort(key=lambda j: j.get("startedAt", datetime.min), reverse=True)
    j = jobs[0]
    return j["status"], j["ingestionJobId"], j.get("startedAt")


# =========================
# Voice / Transcribe helpers
# =========================
TRANSCRIBE_BUCKET      = st.secrets["TRANSCRIBE_BUCKET"]
TRANSCRIBE_LANGUAGE    = st.secrets.get("TRANSCRIBE_LANGUAGE", "en-AU")
TRANSCRIBE_MEDIA_FORMAT= st.secrets.get("TRANSCRIBE_MEDIA_FORMAT", "wav")

transcribe = session.client("transcribe", config=boto_cfg)

def _build_audio_key(doctor_sub: str, suffix: str = "wav") -> str:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    uid = str(uuid4())[:8]
    return f"inputs/{doctor_sub}/{ts}-{uid}.{suffix}"

def _start_transcription_job(s3_uri: str) -> str:
    job_name = f"sttx-{uuid4().hex[:16]}"
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": s3_uri},
        MediaFormat=TRANSCRIBE_MEDIA_FORMAT,
        LanguageCode=TRANSCRIBE_LANGUAGE,
        OutputBucketName=TRANSCRIBE_BUCKET,
        OutputKey=f"outputs/{job_name}.json"
    )
    return job_name

def _wait_transcription(job_name: str, timeout_s: int = 180) -> dict | None:
    """Poll until COMPLETED/FAILED or timeout. Returns job dict or None."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        resp = transcribe.get_transcription_job(TranscriptionJobName=job_name)
        status = resp["TranscriptionJob"]["TranscriptionJobStatus"]
        if status == "COMPLETED":
            return resp["TranscriptionJob"]
        if status == "FAILED":
            raise RuntimeError(resp["TranscriptionJob"].get("FailureReason", "Transcribe failed"))
        time.sleep(2.0)
    return None

def _read_transcript_text(job_name: str) -> str:
    # Transcribe wrote to our bucket/prefix set above
    key = f"outputs/{job_name}.json"
    blob = s3.get_object(Bucket=TRANSCRIBE_BUCKET, Key=key)["Body"].read()
    data = json.loads(blob.decode("utf-8"))
    # Typical shape: results.transcripts[0].transcript
    return (data.get("results", {})
                .get("transcripts", [{}])[0]
                .get("transcript", "")).strip()



# =========================
# UI
# =========================
if "busy" not in st.session_state:
    st.session_state.busy = False

sub = get_user_sub_from_token()
if not sub:
    st.error("Could not determine your Clinician ID (sub). Please contact admin.")
    st.stop()

patient_ids = get_patient_ids(sub)

with st.sidebar:
    st.header("User Information")

    st.markdown("## Clinician")
    st.text(authenticator.get_username())

    st.markdown("## Clinician ID")
    st.text(sub or "(unknown)")

    # Patient picker
    selected_patient = st.selectbox(
        "Select a patient (or 'All' for all patients)",
        ['All'] + patient_ids,
        key="patient_select"
    )

    st.divider()

    # --- Add Patient (moved into sidebar) ---
    st.subheader("Add patient")
    with st.form("add_patient_form_sidebar", clear_on_submit=True):
        new_pid_input = st.text_input(
            "New Patient ID",
            placeholder="e.g. f9ee14c8-5071-70f7-1ac4-89b2deb95960",
            key="sidebar_new_pid"
        )
        add_submitted = st.form_submit_button("Add")

    if add_submitted:
        new_pid_val = (st.session_state.get("sidebar_new_pid") or "").strip()
        if not new_pid_val:
            st.warning("Please enter a patient ID.")
        else:
            try:
                res = add_patient_to_doctor(dynamo_table, sub, new_pid_val)
                if res == "exists":
                    st.info(f"Patient {new_pid_val} is already assigned to this Clinician.")
                else:
                    st.success(f"Patient {new_pid_val} assigned.")
                    st.rerun()  # refresh sidebar so the new patient appears immediately
            except Exception as e:
                st.error(f"Failed to add patient: {e}")

    st.divider()

    # Logout button stays last
    st.button("Logout", key="logout_btn", on_click=logout)


st.header("Secluded DataDissect environment")

# --- KB status pill (always visible) ---
try:
    _ds_id, _bucket = _get_kb_data_source(kb_id)
    status, last_job_id, started = _latest_ingestion_status(kb_id, _ds_id)
    color = {"COMPLETE": "#16a34a", "IN_PROGRESS": "#f59e0b", "FAILED": "#dc2626", "NO_JOBS": "#6b7280"}.get(status, "#6b7280")
    st.markdown(
        f"<div style='display:inline-block;padding:3px 10px;border-radius:999px;background:{color};color:white;font-weight:600;'>"
        f"KB status: {status.replace('_',' ')}"
        f"{' • job '+last_job_id if last_job_id else ''}</div>",
        unsafe_allow_html=True
    )
    if st.button("Refresh KB status"):
        st.rerun()
except Exception as e:
    st.info(f"KB status unavailable: {e}")

# --- Unified search (text + inline mic) ---
st.subheader("Search")

# init keys once
st.session_state.setdefault("query_text_input", "")
st.session_state.setdefault("last_results", None)
st.session_state.setdefault("run_q", None)
st.session_state.setdefault("prefill_text", None)  # <-- temp buffer for voice text

# If we have a pending prefill from transcription, apply it BEFORE rendering the input
if st.session_state["prefill_text"] is not None:
    st.session_state["query_text_input"] = st.session_state["prefill_text"]
    st.session_state["prefill_text"] = None

# input row: text box + mic only
c1, c2 = st.columns([12, 1])
with c1:
    st.text_input(
        "Enter your search query:",
        key="query_text_input",
        label_visibility="collapsed",
        placeholder="Type your question… or tap the mic"
    )
with c2:
    audio = mic_recorder(
        start_prompt="🎙",
        stop_prompt="⏹",
        format="wav",
        just_once=True,
        use_container_width=True,
        key="mic_inline"
    )

# search button BELOW the input bar
do_search = st.button("Search", type="primary", use_container_width=True)

# if mic captured audio: upload → transcribe → stash to prefill_text → rerun
if audio and audio.get("bytes"):
    try:
        audio_bytes = audio["bytes"]
        key = _build_audio_key(sub, suffix=TRANSCRIBE_MEDIA_FORMAT)
        s3.put_object(
            Bucket=TRANSCRIBE_BUCKET,
            Key=key,
            Body=audio_bytes,
            ContentType="audio/wav"
        )
        s3_uri = f"s3://{TRANSCRIBE_BUCKET}/{key}"
        with st.spinner("Transcribing…"):
            job_name = _start_transcription_job(s3_uri)
            job = _wait_transcription(job_name, timeout_s=240)
        if not job:
            st.warning("Transcription timed out. Please try again.")
        else:
            text = _read_transcript_text(job_name)
            if text:
                # DO NOT touch query_text_input here; stash and rerun
                st.session_state["prefill_text"] = text
                st.toast("Transcript added to search box", icon="📝")
                st.rerun()
            else:
                st.warning("No speech detected. Try again.")
    except Exception as e:
        st.error(f"Voice capture failed: {e}")
st.divider()
# Handle Search click (no auto-clearing per your request)
if do_search:
    q = (st.session_state.get("query_text_input") or "").strip()
    if not q:
        st.warning("Please enter a search query.")
    else:
        st.session_state["run_q"] = q
        st.rerun()

# Execute pending search (after rerun)
if st.session_state.get("run_q"):
    run_q = st.session_state["run_q"]
    st.session_state["run_q"] = None
    patient_ids_filter = [selected_patient] if selected_patient != 'All' else patient_ids
    with st.spinner("Searching KB…"):
        out = search_transcript(sub, kb_id, run_q, patient_ids_filter)
    st.session_state["last_results"] = out
    st.rerun()

# render results (if any)
if st.session_state.get("last_results"):
    res = st.session_state["last_results"]
    if "error" in res:
        st.error(res["error"])
    elif "body" in res:
        st.subheader("Search Results")
        st.markdown(res["body"], unsafe_allow_html=True)
    else:
        st.info("No content returned.")



st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
st.divider()
# --- Add to knowledge base (renamed & simplified) ---
st.subheader("Add to knowledge base")

uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf","txt","md","html","doc","docx","csv","xls","xlsx","png","jpeg","jpg"],
    accept_multiple_files=True,
    help="Each file ≤ 50MB. CSV needs a 'content' column."
)

# Minimal manual entry UI (no filename)
c_opt, _ = st.columns([1, 3])
with c_opt:
    save_and_ingest = st.toggle("Start ingestion after saving", value=True)

typed_content = st.text_area(
    label="",
    height=220,
    placeholder="Write a quick note (Markdown supported)…"
)

col_btn1, col_btn2 = st.columns([1, 1])
with col_btn1:
    save_btn = st.button("Save Note", type="primary")
with col_btn2:
    preview_btn = st.button("Preview Markdown")

if preview_btn and typed_content.strip():
    st.markdown("**Preview:**")
    st.markdown(typed_content)

st.divider()

# Manual note save (auto filename)
if save_btn:
    if selected_patient == "All":
        st.warning("Please select a single patient in the sidebar first.")
    elif not typed_content.strip():
        st.warning("The note is empty.")
    else:
        try:
            data_source_id, bucket = _get_kb_data_source(kb_id)
            file_bytes = typed_content.encode("utf-8")
            auto_name = f"note-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.md"

            stored_key, meta_key = _put_file_and_metadata(
                bucket=bucket,
                filename=auto_name,
                file_bytes=file_bytes,
                patient_id=selected_patient
            )
            st.success(f"Saved `{auto_name}` as `s3://{bucket}/{stored_key}` (metadata: `{meta_key}`).")

            if save_and_ingest:
                job_id = _start_ingestion(kb_id, data_source_id)
                st.info(f"Started ingestion job: {job_id}. Click 'Refresh KB status' to check progress.")
        except Exception as e:
            st.error(f"Failed to save or ingest note: {e}")

# Upload files area
col1, col2 = st.columns([1, 1])
with col1:
    do_ingest = st.button("Upload & Start Ingestion", type="primary")
with col2:
    just_upload = st.button("Upload Only")

if (do_ingest or just_upload):
    if selected_patient == "All":
        st.warning("Please select a single patient in the sidebar before uploading, so we can stamp the correct patient_id in metadata.")
    elif not uploaded_files:
        st.warning("Please pick at least one file.")
    else:
        try:
            data_source_id, bucket = _get_kb_data_source(kb_id)
            uploaded = []
            for f in uploaded_files:
                key = f.name  # keep original filename
                _k, _meta_key = _put_file_and_metadata(bucket, key, f.read(), selected_patient)
                uploaded.append((key, _meta_key))
            st.success(f"Uploaded {len(uploaded)} file(s) to s3://{bucket}/ (with metadata).")

            if do_ingest:
                job_id = _start_ingestion(kb_id, data_source_id)
                st.info(f"Started ingestion job: {job_id}. Click 'Refresh KB status' above to check progress.")
        except Exception as e:
            st.error(f"Upload/ingest failed: {e}")
