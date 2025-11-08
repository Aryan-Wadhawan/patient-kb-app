import json
import re
import mimetypes
from uuid import uuid4
from datetime import datetime

import streamlit as st
import boto3
from botocore.config import Config
from streamlit_cognito_auth import CognitoAuthenticator

# Ensure Markdown mimetype is registered
mimetypes.add_type("text/markdown", ".md")

# =========================
# Config / Secrets
# =========================
AWS_REGION = st.secrets.get("AWS_REGION", "ap-southeast-2")

# Required secrets (set in .streamlit/secrets.toml locally or Streamlit Cloud Secrets UI)
pool_id            = st.secrets["COGNITO_POOL_ID"]
app_client_id      = st.secrets["COGNITO_APP_CLIENT_ID"]
app_client_secret  = st.secrets["COGNITO_APP_CLIENT_SECRET"]

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
    app_client_secret=app_client_secret,
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
# UI
# =========================
if "busy" not in st.session_state:
    st.session_state.busy = False

sub = get_user_sub(pool_id, authenticator.get_username())
patient_ids = get_patient_ids(sub)

with st.sidebar:
    st.header("User Information")
    st.markdown("## Clinician")
    st.text(authenticator.get_username())
    st.markdown("## Clinician ID")
    st.text(sub or "(unknown)")
    selected_patient = st.selectbox("Select a patient (or 'All' for all patients)", ['All'] + patient_ids)
    st.button("Logout", "logout_btn", on_click=logout)

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

# --- Search ---
query = st.text_input("Enter your search query:")

if st.button("Search"):
    if query:
        patient_ids_filter = [selected_patient] if selected_patient != 'All' else patient_ids
        results = search_transcript(sub, kb_id, query, patient_ids_filter)
        if "error" in results:
            st.error(results["error"])
        elif "body" in results:
            st.subheader("Search Results:")
            st.markdown(results["body"], unsafe_allow_html=True)
        else:
            st.info("No content returned.")
    else:
        st.warning("Please enter a search query.")

st.divider()
st.subheader("Upload to Knowledge Base")

uploaded_files = st.file_uploader(
    "Choose files",
    type=["pdf","txt","md","html","doc","docx","csv","xls","xlsx","png","jpeg","jpg"],
    accept_multiple_files=True,
    help="Each file ≤ 50MB. CSV needs a 'content' column."
)

st.subheader("Manual entry to KB")

col_a, col_b = st.columns([2, 1])
with col_a:
    typed_filename = st.text_input(
        "Filename (no path)",
        value="note.md",
        help="Use .md for Markdown or .txt for plain text"
    )
with col_b:
    save_and_ingest = st.toggle("Start ingestion after saving", value=True)

typed_content = st.text_area(
    "Type your note (Markdown supported)",
    height=220,
    placeholder="## Patient update\n- ...\n- ..."
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

if save_btn:
    if selected_patient == "All":
        st.warning("Please select a single patient in the sidebar first.")
    elif not typed_filename.strip():
        st.warning("Please enter a filename (e.g., note.md).")
    elif not (typed_filename.lower().endswith(".md") or typed_filename.lower().endswith(".txt")):
        st.warning("Filename must end with .md or .txt")
    elif not typed_content.strip():
        st.warning("The note is empty.")
    else:
        try:
            data_source_id, bucket = _get_kb_data_source(kb_id)
            file_bytes = typed_content.encode("utf-8")
            stored_key, meta_key = _put_file_and_metadata(
                bucket=bucket,
                filename=typed_filename,
                file_bytes=file_bytes,
                patient_id=selected_patient
            )
            st.success(f"Saved `{typed_filename}` as `s3://{bucket}/{stored_key}` (metadata: `{meta_key}`).")

            if save_and_ingest:
                job_id = _start_ingestion(kb_id, data_source_id)
                st.info(f"Started ingestion job: {job_id}. Click 'Refresh KB status' to check progress.")
        except Exception as e:
            st.error(f"Failed to save or ingest note: {e}")

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

st.divider()
st.subheader("Add Patient to This Clinician")

with st.form("add_patient_form"):
    new_pid = st.text_input("New Patient ID (e.g., f9ee14c8-5071-70f7-1ac4-89b2deb95960)").strip()
    add_submitted = st.form_submit_button("Add Patient")

if add_submitted:
    if not new_pid:
        st.warning("Please enter a patient ID.")
    else:
        try:
            res = add_patient_to_doctor(dynamo_table, sub, new_pid)
            if res == "exists":
                st.info(f"Patient {new_pid} is already assigned to this Clinician.")
            else:
                st.success(f"Patient {new_pid} assigned.")
                st.rerun()
        except Exception as e:
            st.error(f"Failed to add patient: {e}")
