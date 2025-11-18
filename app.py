import json
import re
import mimetypes
import html
from uuid import uuid4
from datetime import datetime
from typing import Optional, Union

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

/* Pill styling */
.pill-container { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; margin: 0.5rem 0; }
.pill { display: inline-flex; align-items: center; padding: 6px 12px; border-radius: 999px; font-size: 0.875rem; font-weight: 500; cursor: pointer; transition: all 0.2s; }
.pill-filter-logic { background: #6366f1; color: white; border: 2px solid #6366f1; }
.pill-filter-logic:hover { background: #4f46e5; border-color: #4f46e5; }
.pill-patient { background: #3b82f6; color: white; border: 2px solid #3b82f6; }
.pill-patient:hover { background: #2563eb; border-color: #2563eb; }
.pill-doc-type { background: #10b981; color: white; border: 2px solid #10b981; }
.pill-doc-type:hover { background: #059669; border-color: #059669; }
.pill-date { background: #f59e0b; color: white; border: 2px solid #f59e0b; }
.pill-date:hover { background: #d97706; border-color: #d97706; }
.pill-misc { background: #8b5cf6; color: white; border: 2px solid #8b5cf6; }
.pill-misc:hover { background: #7c3aed; border-color: #7c3aed; }
.pill-divider { width: 2px; height: 24px; background: #e5e7eb; margin: 0 0.25rem; }
.pill-expanded { margin-top: 0.5rem; padding: 0.75rem; background: #f9fafb; border-radius: 8px; border: 1px solid #e5e7eb; }
.tag-buttons-container { display: flex; flex-wrap: wrap; gap: 0.5rem; align-items: center; }
.tag-row { display: flex !important; flex-direction: row !important; flex-wrap: wrap; gap: 0.25rem; }
.tag-row > * { flex: 0 0 auto !important; }

/* Chat styling */
.chat-container { 
    height: 500px; 
    overflow-y: auto; 
    padding: 1rem; 
    background: #f9fafb; 
    border-radius: 8px; 
    border: 1px solid #e5e7eb;
    margin-bottom: 1rem;
}
.chat-message { 
    margin-bottom: 1rem; 
    display: flex; 
    align-items: flex-start;
}
.chat-message.user { justify-content: flex-end; }
.chat-message.assistant { justify-content: flex-start; }
.chat-bubble { 
    max-width: 70%; 
    padding: 0.75rem 1rem; 
    border-radius: 12px; 
    word-wrap: break-word;
}
.chat-bubble.user { 
    background: #3b82f6; 
    color: white; 
    border-bottom-right-radius: 4px;
}
.chat-bubble.assistant { 
    background: white; 
    color: #1f2937; 
    border: 1px solid #e5e7eb;
    border-bottom-left-radius: 4px;
}
.chat-timestamp {
    font-size: 0.75rem;
    color: #6b7280;
    margin-top: 0.25rem;
    padding: 0 0.5rem;
}
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
bedrock_runtime = session.client("bedrock-runtime", config=boto_cfg)
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

def get_user_sub_from_token() -> Optional[str]:
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

def _construct_metadata_filter(filter_params: dict) -> dict:
    """
    Construct AWS Bedrock metadata filter from filter parameters.
    Note: patient_id filtering is handled separately for access control.
    Returns a filter dict compatible with Bedrock Knowledge Base filtering.
    """
    filters = []
    
    # Document type filter
    if filter_params.get("document_types"):
        doc_types = filter_params["document_types"]
        if len(doc_types) == 1:
            filters.append({
                "equals": {
                    "key": "document_type",
                    "value": doc_types[0]
                }
            })
        else:
            filters.append({
                "in": {
                    "key": "document_type",
                    "value": doc_types
                }
            })
    
    # Miscellaneous tags filter
    if filter_params.get("miscellaneous_tags"):
        misc_tags = filter_params["miscellaneous_tags"]
        if len(misc_tags) == 1:
            filters.append({
                "listContains": {
                    "key": "miscellaneous_tags",
                    "value": misc_tags[0]
                }
            })
        else:
            # For multiple tags, use OR logic within the tag filter
            tag_filters = [{
                "listContains": {
                    "key": "miscellaneous_tags",
                    "value": tag
                }
            } for tag in misc_tags]
            filters.append({
                "orAll": tag_filters
            })
    
    # Construct final filter based on logic type
    if not filters:
        return None
    
    if len(filters) == 1:
        return filters[0]
    
    # Multiple filters - use AND or OR logic
    filter_logic = filter_params.get("filter_logic", "AND")
    if filter_logic == "AND":
        return {"andAll": filters}
    else:
        return {"orAll": filters}

def search_transcript(doctor_id: str, kb_id: str, text: str, patient_ids: list[str], filter_params: dict = None, session_id: str = None):
    """
    Search knowledge base with optional metadata filtering and Bedrock session support.
    
    Args:
        doctor_id: Doctor ID
        kb_id: Knowledge base ID
        text: Search query text
        patient_ids: List of patient IDs (for access control)
        filter_params: Optional dict with filter parameters:
            - filter_logic: "AND" or "OR"
            - patient_ids: List of patient IDs (can override the patient_ids param)
            - document_types: List of document types
            - miscellaneous_tags: List of miscellaneous tags
        session_id: Optional Bedrock session ID for conversation context (from previous response)
    
    Returns:
        dict with "body" (response text), "error" (if any), and "sessionId" (for next request)
    """
    # Use filter_params patient_ids if provided, otherwise use the param
    final_patient_ids = filter_params.get("patient_ids", patient_ids) if filter_params else patient_ids
    
    # Construct metadata filter
    metadata_filter = None
    if filter_params:
        metadata_filter = _construct_metadata_filter(filter_params)
    
    payload = {
        "doctorId": doctor_id,
        "knowledgeBaseId": kb_id,
        "text": text,
        "patientIds": final_patient_ids,
        "metadataFilter": metadata_filter
    }
    
    # Add sessionId if provided (for subsequent requests in the same conversation)
    if session_id:
        payload["sessionId"] = session_id

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
            # If response is not JSON, return as plain text (legacy format)
            return {"body": raw, "sessionId": None}

        result = {"body": None, "sessionId": None}
        
        if isinstance(data, dict):
            if "body" in data:
                body = data["body"]
                try:
                    body_json = json.loads(body)
                    if isinstance(body_json, dict):
                        # New format: {"text": "...", "sessionId": "..."}
                        if "text" in body_json and "sessionId" in body_json:
                            result["body"] = body_json["text"]
                            result["sessionId"] = body_json["sessionId"]
                        # Legacy formats for backward compatibility
                        elif "html" in body_json:
                            result["body"] = body_json["html"]
                            if "sessionId" in body_json:
                                result["sessionId"] = body_json["sessionId"]
                        elif "message" in body_json:
                            result["body"] = body_json["message"]
                            if "sessionId" in body_json:
                                result["sessionId"] = body_json["sessionId"]
                        else:
                            result["body"] = "<pre>" + json.dumps(body_json, indent=2) + "</pre>"
                            if "sessionId" in body_json:
                                result["sessionId"] = body_json["sessionId"]
                    else:
                        result["body"] = str(body_json)
                except Exception:
                    # If body is not JSON, treat as plain text
                    result["body"] = body
            elif "html" in data:
                result["body"] = data["html"]
            elif "results" in data:
                result["body"] = "<pre>" + json.dumps(data["results"], indent=2) + "</pre>"
            elif "error" in data:
                return {"error": data["error"]}
            else:
                result["body"] = "<pre>" + json.dumps(data, indent=2) + "</pre>"
        else:
            result["body"] = "<pre>" + json.dumps(data, indent=2) + "</pre>"
        
        return result

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

def _put_file_and_metadata(bucket: str, filename: str, file_bytes: bytes, patient_id: str, 
                           document_type: str = "other", date: str = None, miscellaneous_tags: list = None):
    """
    Upload file and metadata to S3.
    
    Args:
        bucket: S3 bucket name
        filename: Original filename
        file_bytes: File content as bytes
        patient_id: Patient ID (required)
        document_type: Document type (ultrasound/pathology/other)
        date: Date in ISO format (YYYY-MM-DD), defaults to today if None
        miscellaneous_tags: List of miscellaneous tag strings
    """
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
    
    # Default date to today if not provided
    if date is None:
        date = datetime.utcnow().strftime("%Y-%m-%d")
    
    # Build metadata document
    metadata_attrs = {
        "patient_id": patient_id,
        "document_type": document_type,
        "date": date
    }
    
    # Add miscellaneous_tags if provided
    if miscellaneous_tags and len(miscellaneous_tags) > 0:
        metadata_attrs["miscellaneous_tags"] = miscellaneous_tags
    
    metadata_doc = {"metadataAttributes": metadata_attrs}
    
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

def _wait_transcription(job_name: str, timeout_s: int = 180) -> Optional[dict]:
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

def _render_tagging_pills(patient_ids: list, default_patient: str = None, key_prefix: str = "note"):
    """
    Render pill-based tagging UI for upload/note saving.
    Returns: dict with selected tags
    
    Args:
        patient_ids: List of patient IDs
        default_patient: Default patient ID to select
        key_prefix: Prefix for widget keys to avoid duplicates (use "note" for manual notes, "upload" for file uploads)
    """
    # Initialize session state with prefix-specific keys
    patient_key = f"{key_prefix}_tag_patient_id"
    doc_type_key = f"{key_prefix}_tag_document_type"
    date_key = f"{key_prefix}_tag_date"
    misc_key = f"{key_prefix}_tag_misc"
    
    if patient_key not in st.session_state:
        st.session_state[patient_key] = default_patient if default_patient and default_patient != "All" else None
    if doc_type_key not in st.session_state:
        st.session_state[doc_type_key] = None
    if date_key not in st.session_state:
        st.session_state[date_key] = None
    if misc_key not in st.session_state:
        st.session_state[misc_key] = []
    
    # Render pills container
    st.markdown('<div class="pill-container">', unsafe_allow_html=True)
    
    # Note: AND/OR logic is only used during SEARCH, not during file upload
    # During upload, tags are simply stored as metadata attributes
    col_patient, col_doc, col_date, col_misc = st.columns([2, 2, 2, 2])
    
    with col_patient:
        display_patient = st.session_state[patient_key][:15] + "..." if st.session_state[patient_key] and len(st.session_state[patient_key]) > 15 else (st.session_state[patient_key] or "Select")
        st.markdown(f'<span class="pill pill-patient">👤 {display_patient}</span>', unsafe_allow_html=True)
        with st.expander("👤 Patient", expanded=False):
            # Add "None" option at the beginning
            patient_options = [None] + patient_ids if patient_ids else [None]
            current_index = 0
            if st.session_state[patient_key] and st.session_state[patient_key] in patient_ids:
                current_index = patient_ids.index(st.session_state[patient_key]) + 1
            
            selected_patient = st.selectbox(
                "Select Patient",
                patient_options,
                index=current_index,
                key=f"{key_prefix}_select_tag_patient",
                label_visibility="visible",
                format_func=lambda x: "Select..." if x is None else x
            )
            st.session_state[patient_key] = selected_patient
    
    with col_doc:
        doc_display = st.session_state[doc_type_key] if st.session_state[doc_type_key] else "Select"
        st.markdown(f'<span class="pill pill-doc-type">📄 {doc_display}</span>', unsafe_allow_html=True)
        with st.expander("📄 Doc Type", expanded=False):
            doc_options = [None, "ultrasound", "pathology", "other"]
            current_index = 0
            if st.session_state[doc_type_key] in ["ultrasound", "pathology", "other"]:
                current_index = doc_options.index(st.session_state[doc_type_key])
            
            doc_type = st.selectbox(
                "Document Type",
                doc_options,
                index=current_index,
                key=f"{key_prefix}_select_tag_doc_type",
                label_visibility="visible",
                format_func=lambda x: "Select..." if x is None else x
            )
            st.session_state[doc_type_key] = doc_type
    
    with col_date:
        date_display = st.session_state[date_key].strftime("%Y-%m-%d") if st.session_state[date_key] else "Select"
        st.markdown(f'<span class="pill pill-date">📅 {date_display}</span>', unsafe_allow_html=True)
        with st.expander("📅 Date", expanded=False):
            selected_date = st.date_input(
                "Date",
                value=st.session_state[date_key] if st.session_state[date_key] else datetime.utcnow().date(),
                key=f"{key_prefix}_tag_date_input",
                label_visibility="visible"
            )
            st.session_state[date_key] = selected_date
    
    with col_misc:
        misc_display = ", ".join(st.session_state[misc_key][:1]) if st.session_state[misc_key] else "Add"
        if len(st.session_state[misc_key]) > 1:
            misc_display += f" (+{len(st.session_state[misc_key]) - 1})"
        st.markdown(f'<span class="pill pill-misc">🏷️ {misc_display}</span>', unsafe_allow_html=True)
        with st.expander("🏷️ Misc", expanded=False):
            # Display selected tags with remove buttons
            if st.session_state[misc_key]:
                st.markdown("**Selected tags:**")
                num_tags = len(st.session_state[misc_key])
                
                # Force horizontal layout using columns
                # Display up to 10 tags per row
                max_per_row = 10
                
                for row_start in range(0, num_tags, max_per_row):
                    row_end = min(row_start + max_per_row, num_tags)
                    row_size = row_end - row_start
                    
                    # Create columns for this row - this forces horizontal layout
                    cols = st.columns(row_size)
                    
                    for col_idx in range(row_size):
                        tag_idx = row_start + col_idx
                        tag = st.session_state[misc_key][tag_idx]
                        
                        with cols[col_idx]:
                            safe_tag_key = tag.replace(" ", "_").replace(",", "").replace(":", "").replace("'", "").replace('"', "")[:30]
                            if st.button(f"❌ {tag}", key=f"{key_prefix}_remove_{tag_idx}_{safe_tag_key}", use_container_width=True):
                                st.session_state[misc_key].pop(tag_idx)
                                st.session_state[f"{key_prefix}_input_misc_tags"] = ", ".join(st.session_state[misc_key])
                                st.rerun()
                
                st.markdown("---")
            
            misc_tags_input_key = f"{key_prefix}_input_misc_tags"
            # Sync text input with tags list - update if tags were added via buttons
            current_tags_str = ", ".join(st.session_state[misc_key]) if st.session_state[misc_key] else ""
            if misc_tags_input_key not in st.session_state:
                st.session_state[misc_tags_input_key] = current_tags_str
            else:
                # If tags list changed (e.g., via button clicks), sync the input
                existing_input = st.session_state[misc_tags_input_key]
                existing_tags_from_input = [tag.strip() for tag in existing_input.split(",") if tag.strip()] if existing_input else []
                # Compare sets to see if tags list was updated externally
                if set(st.session_state[misc_key]) != set(existing_tags_from_input):
                    st.session_state[misc_tags_input_key] = current_tags_str
            
            misc_tags = st.text_input(
                "Add/Edit Tags",
                value=st.session_state[misc_tags_input_key],
                key=misc_tags_input_key,
                help="Comma-separated tags (e.g., 'urgent, follow-up, routine'). Use buttons above to remove tags.",
                label_visibility="visible"
            )
            # Update tags list based on the text input value
            # Read from session state to ensure we get the latest value
            current_input_value = st.session_state.get(misc_tags_input_key, "")
            if current_input_value and current_input_value.strip():
                parsed_tags = [tag.strip() for tag in current_input_value.split(",") if tag.strip()]
                st.session_state[misc_key] = parsed_tags
            else:
                st.session_state[misc_key] = []
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        "patient_id": st.session_state[patient_key],
        "document_type": st.session_state[doc_type_key] if st.session_state[doc_type_key] else "other",
        "date": st.session_state[date_key].strftime("%Y-%m-%d") if st.session_state[date_key] else datetime.utcnow().strftime("%Y-%m-%d"),
        "miscellaneous_tags": st.session_state[misc_key]
        # Note: filter_logic is NOT used during upload - it's only used during search
        # The AND/OR toggle you see in the search section controls how multiple filters are combined when querying
    }

def _extract_any_patient_id(content: str) -> Optional[str]:
    """
    Extract any patient ID from content using AI only (no algorithmic pattern matching).
    Returns: patient ID if found, None otherwise
    """
    if not content or not content.strip():
        return None
    
    # Use AI to extract patient identifier
    content_preview = content[:3000] if len(content) > 3000 else content
    
    prompt = f"""Analyze the following medical document and extract the patient identifier.
Look for:
- Patient ID fields (e.g., "Patient ID: 123", "Patient: 5")
- Patient identifiers (UUIDs or text like "Patient 1", "Patient 5")
- UUIDs in the format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
- Any explicit patient ID mentions

Document content:
{content_preview}

CRITICAL: Respond with ONLY the patient identifier itself, nothing else. Do not include any explanation, sentence, or additional text.
Examples:
- If you see "Patient ID: 4567", respond with: 4567
- If you see "Patient identifier is ABC123", respond with: ABC123
- If you see a UUID "f9ee14c8-5071-70f7-1ac4-89b2deb95960", respond with: f9ee14c8-5071-70f7-1ac4-89b2deb95960

If no patient ID is found, respond with exactly: NOT_FOUND"""

    try:
        model_id = f"anthropic.claude-3-haiku-20240307-v1:0"
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read())
        result = response_body.get('content', [{}])[0].get('text', '').strip()
        
        # Post-process to extract just the ID if AI returned a sentence
        if result and result != "NOT_FOUND":
            # Remove common prefixes/suffixes that AI might add
            result = result.strip()
            
            # Try to extract UUID pattern
            uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
            uuid_match = re.search(uuid_pattern, result)
            if uuid_match:
                return uuid_match.group(0)
            
            # Try to extract after common phrases (handles sentences like "The patient identifier in the document is 4567")
            patterns = [
                r'(?:patient\s*(?:id|identifier|ID|Identifier)?\s*(?:in\s+the\s+document\s+)?(?:is|:|=)\s*)([A-Za-z0-9\-_]+)',
                r'(?:identifier\s*(?:in\s+the\s+document\s+)?(?:is|:|=)\s*)([A-Za-z0-9\-_]+)',
                r'(?:ID\s*(?:in\s+the\s+document\s+)?(?:is|:|=)\s*)([A-Za-z0-9\-_]+)',
                r'(?:is\s+)([A-Za-z0-9\-_]+)(?:\s*\.?\s*$)',  # "is 4567." at end
                r'(?:document\s+is\s+)([A-Za-z0-9\-_]+)',  # "document is 4567"
                r'(?:is\s+)([A-Za-z0-9\-_]+)(?:\s*\.)',  # "is 4567." anywhere
            ]
            for pattern in patterns:
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    # Remove trailing punctuation
                    extracted = re.sub(r'[.,;:!?]+$', '', extracted)
                    if extracted and len(extracted) > 0:
                        return extracted
            
            # If result looks like it might be just the ID (alphanumeric, no spaces, reasonable length)
            if re.match(r'^[A-Za-z0-9\-_]+$', result) and len(result) <= 100:
                return result
        
        return None
    except Exception as e:
        print(f"Patient ID extraction failed: {e}")
        return None

def _extract_patient_id(content: str, available_patient_ids: list[str]) -> Optional[str]:
    """
    Extract patient ID from content using AI only (no algorithmic pattern matching).
    Returns: patient ID if found in available list, None otherwise
    """
    if not content or not content.strip() or not available_patient_ids:
        return None
    
    # Use AI to extract patient identifier
    content_preview = content[:3000] if len(content) > 3000 else content
    
    prompt = f"""Analyze the following medical document and extract the patient identifier.
Available patient IDs: {', '.join(available_patient_ids[:10])}

Look for:
- Patient ID fields
- Patient identifiers
- UUIDs that match the format of available patient IDs
- Any explicit patient ID mentions

Document content:
{content_preview}

CRITICAL: Respond with ONLY the patient ID exactly as it appears in the available list above. Do not include any explanation, sentence, or additional text.
- Match the ID exactly (case-sensitive) from the available list
- If you see "Patient ID: 4567" and 4567 is in the available list, respond with: 4567
- If you see a UUID that matches one in the available list, respond with that exact UUID

If no matching patient ID is found, respond with exactly: NOT_FOUND"""

    try:
        model_id = f"anthropic.claude-3-haiku-20240307-v1:0"
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 50,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read())
        result = response_body.get('content', [{}])[0].get('text', '').strip()
        
        # Post-process to extract just the ID if AI returned a sentence
        if result and result != "NOT_FOUND":
            # Try to extract UUID pattern first
            uuid_pattern = r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'
            uuid_match = re.search(uuid_pattern, result)
            if uuid_match:
                extracted_id = uuid_match.group(0)
                # Check if it matches any available patient ID
                for pid in available_patient_ids:
                    if pid == extracted_id:
                        return pid
            
            # Try to extract after common phrases (handles sentences like "The patient identifier in the document is 4567")
            patterns = [
                r'(?:patient\s*(?:id|identifier|ID|Identifier)?\s*(?:in\s+the\s+document\s+)?(?:is|:|=)\s*)([A-Za-z0-9\-_]+)',
                r'(?:identifier\s*(?:in\s+the\s+document\s+)?(?:is|:|=)\s*)([A-Za-z0-9\-_]+)',
                r'(?:ID\s*(?:in\s+the\s+document\s+)?(?:is|:|=)\s*)([A-Za-z0-9\-_]+)',
                r'(?:is\s+)([A-Za-z0-9\-_]+)(?:\s*\.?\s*$)',  # "is 4567." at end
                r'(?:document\s+is\s+)([A-Za-z0-9\-_]+)',  # "document is 4567"
                r'(?:is\s+)([A-Za-z0-9\-_]+)(?:\s*\.)',  # "is 4567." anywhere
            ]
            for pattern in patterns:
                match = re.search(pattern, result, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    # Remove trailing punctuation
                    extracted = re.sub(r'[.,;:!?]+$', '', extracted)
                    # Check if extracted ID matches any available patient ID (case-insensitive)
                    if extracted:
                        extracted_lower = extracted.lower()
                        for pid in available_patient_ids:
                            if pid.lower() == extracted_lower:
                                return pid
            
            # Check if result itself matches any available patient ID (case-insensitive)
            result_lower = result.lower()
            for pid in available_patient_ids:
                if pid.lower() == result_lower:
                    return pid
        
        return None
    except Exception as e:
        print(f"Patient ID extraction failed: {e}")
        return None

def _extract_date(content: str) -> Optional[str]:
    """
    Extract date from content using pattern matching and AI.
    Returns: date in YYYY-MM-DD format if found, None otherwise
    """
    if not content or not content.strip():
        return None
    
    # Try common date patterns first
    date_patterns = [
        (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', '%Y-%m-%d'),  # YYYY-MM-DD
        (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', '%m/%d/%Y'),  # MM/DD/YYYY
        (r'\b(\d{1,2})-(\d{1,2})-(\d{4})\b', '%m-%d-%Y'),  # MM-DD-YYYY
        (r'\b(\d{1,2})/(\d{1,2})/(\d{2})\b', '%m/%d/%y'),  # MM/DD/YY
    ]
    
    for pattern, date_format in date_patterns:
        matches = re.findall(pattern, content)
        if matches:
            try:
                from datetime import datetime
                match = matches[0]
                if date_format == '%Y-%m-%d':
                    date_str = '-'.join(match)
                    parsed_date = datetime.strptime(date_str, date_format)
                elif date_format == '%m/%d/%Y':
                    date_str = '/'.join(match)
                    parsed_date = datetime.strptime(date_str, date_format)
                elif date_format == '%m-%d-%Y':
                    date_str = '-'.join(match)
                    parsed_date = datetime.strptime(date_str, date_format)
                elif date_format == '%m/%d/%y':
                    date_str = '/'.join(match)
                    # Convert 2-digit year to 4-digit
                    year = int(match[2])
                    if year < 50:
                        year += 2000
                    else:
                        year += 1900
                    date_str = f"{match[0]}/{match[1]}/{year}"
                    parsed_date = datetime.strptime(date_str, '%m/%d/%Y')
                else:
                    continue
                return parsed_date.strftime('%Y-%m-%d')
            except Exception as e:
                continue
    
    # If pattern matching fails, use AI
    content_preview = content[:3000] if len(content) > 3000 else content
    
    prompt = f"""Extract the document date from the following medical document.
Look for dates in formats like YYYY-MM-DD, MM/DD/YYYY, or written dates.

Document content:
{content_preview}

Respond with ONLY the date in YYYY-MM-DD format, or "NOT_FOUND" if no date is found.
If multiple dates are found, return the most relevant document date (usually near the top or in a header)."""

    try:
        model_id = f"anthropic.claude-3-haiku-20240307-v1:0"
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 20,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read())
        result = response_body.get('content', [{}])[0].get('text', '').strip()
        
        if result and result != "NOT_FOUND":
            # Validate date format
            try:
                from datetime import datetime
                parsed = datetime.strptime(result, '%Y-%m-%d')
                return result
            except:
                # Try to extract date from response
                date_match = re.search(r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', result)
                if date_match:
                    return result
        return None
    except Exception as e:
        print(f"Date extraction failed: {e}")
        return None

def _suggest_misc_tags(content: str, num_suggestions: int = 5) -> list[str]:
    """
    Generate AI suggestions for miscellaneous tags based on document content.
    Returns: list of suggested tag strings
    """
    if not content or not content.strip():
        return []
    
    content_preview = content[:4000] if len(content) > 4000 else content
    
    prompt = f"""Analyze the following medical document and suggest {num_suggestions} relevant tags for categorization.
Tags should be:
- Short (1-3 words)
- Descriptive of key findings, conditions, or document characteristics
- Useful for filtering and searching
- Examples: "urgent", "follow-up", "abnormal findings", "routine", "surgical", "biopsy", "pregnancy", "cardiac", etc.

Document content:
{content_preview}

Respond with ONLY a comma-separated list of {num_suggestions} tags, nothing else.
Example format: urgent, abnormal findings, cardiac, follow-up, routine"""

    try:
        model_id = f"anthropic.claude-3-haiku-20240307-v1:0"
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read())
        result = response_body.get('content', [{}])[0].get('text', '').strip()
        
        if result:
            # Parse comma-separated tags
            tags = [tag.strip() for tag in result.split(',') if tag.strip()]
            return tags[:num_suggestions]  # Limit to requested number
        
        return []
    except Exception as e:
        print(f"Tag suggestion failed: {e}")
        return []

def _extract_document_type(content: str) -> str:
    """
    Extract document type from content using Amazon Bedrock (prompt-based).
    Returns: 'ultrasound', 'pathology', or 'other'
    """
    if not content or not content.strip():
        return "other"
    
    # Truncate content if too long (keep first 5000 chars for prompt)
    content_preview = content[:5000] if len(content) > 5000 else content
    
    prompt = f"""Analyze the following medical document content and classify it as one of these types:
- ultrasound: If the document contains ultrasound imaging reports, sonography findings, or related terminology
- pathology: If the document contains pathology reports, biopsy results, histology findings, or laboratory analysis
- other: If the document does not clearly fit into ultrasound or pathology categories

Document content:
{content_preview}

Respond with ONLY one word: 'ultrasound', 'pathology', or 'other'."""

    try:
        # Use Claude model for classification
        model_id = f"anthropic.claude-3-haiku-20240307-v1:0"
        
        response = bedrock_runtime.invoke_model(
            modelId=model_id,
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 10,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
        )
        
        response_body = json.loads(response['body'].read())
        result = response_body.get('content', [{}])[0].get('text', 'other').strip().lower()
        
        # Validate result
        if result in ['ultrasound', 'pathology', 'other']:
            return result
        else:
            return "other"
    except Exception as e:
        print(f"Document type extraction failed: {e}. Defaulting to 'other'.")
        return "other"



# =========================
# UI Callbacks
# =========================
def auto_detect_note_callback():
    """Callback for auto-detecting tags from manual note content"""
    # Get content from text area widget using its key
    typed_content = st.session_state.get("note_text_area", "")
    if not typed_content or not typed_content.strip():
        return
    
    patient_ids = st.session_state.get("available_patient_ids", [])
    
    # Store content hash to detect changes
    import hashlib
    content_hash = hashlib.md5(typed_content.encode()).hexdigest()
    st.session_state.note_last_detected_content_hash = content_hash
    
    # First try to find patient ID in available list
    detected_patient = _extract_patient_id(typed_content, patient_ids)
    if detected_patient:
        st.session_state.note_tag_patient_id = detected_patient
    else:
        # Try to extract any patient ID (even if not in list)
        any_patient_id = _extract_any_patient_id(typed_content)
        if any_patient_id:
            # Check if it's a new patient - only exact match (case-insensitive)
            matched = False
            for pid in patient_ids:
                if any_patient_id.lower() == pid.lower():
                    st.session_state.note_tag_patient_id = pid
                    matched = True
                    break
            
            if not matched:
                # It's a new patient
                st.session_state.note_detected_new_patient_id = any_patient_id
                st.session_state.note_show_new_patient_confirmation = True
    
    # Detect date
    detected_date = _extract_date(typed_content)
    if detected_date:
        from datetime import datetime
        date_obj = datetime.strptime(detected_date, '%Y-%m-%d').date()
        st.session_state.note_tag_date = date_obj
        st.session_state.note_tag_date_input = date_obj
    
    # Detect document type
    detected_type = _extract_document_type(typed_content)
    st.session_state.note_tag_document_type = detected_type
    
    # Suggest misc tags
    suggested_tags = _suggest_misc_tags(typed_content, num_suggestions=5)
    if suggested_tags:
        st.session_state.note_suggested_tags = suggested_tags
        st.session_state.note_show_suggestions = True
    
    # Mark that auto-detect was just run
    st.session_state.note_auto_detect_just_ran = True

def auto_detect_upload_callback():
    """Callback for auto-detecting tags from uploaded file content"""
    uploaded_files = st.session_state.get("uploaded_files_for_detection", [])
    if not uploaded_files:
        return
    
    first_file = uploaded_files[0]
    patient_ids = st.session_state.get("available_patient_ids", [])
    
    try:
        file_bytes = first_file.read()
        content = file_bytes.decode('utf-8', errors='ignore')
        first_file.seek(0)  # Reset file pointer
        
        # Store content hash to detect changes
        import hashlib
        content_hash = hashlib.md5(content.encode()).hexdigest()
        st.session_state.upload_last_detected_content_hash = content_hash
        
        # First try to find patient ID in available list
        detected_patient = _extract_patient_id(content, patient_ids)
        if detected_patient:
            st.session_state.upload_tag_patient_id = detected_patient
        else:
            # Try to extract any patient ID (even if not in list)
            any_patient_id = _extract_any_patient_id(content)
            if any_patient_id:
                # Check if it's a new patient - only exact match (case-insensitive)
                matched = False
                for pid in patient_ids:
                    if any_patient_id.lower() == pid.lower():
                        st.session_state.upload_tag_patient_id = pid
                        matched = True
                        break
                
                if not matched:
                    # It's a new patient
                    st.session_state.upload_detected_new_patient_id = any_patient_id
                    st.session_state.upload_show_new_patient_confirmation = True
        
        # Detect date
        detected_date = _extract_date(content)
        if detected_date:
            from datetime import datetime
            date_obj = datetime.strptime(detected_date, '%Y-%m-%d').date()
            st.session_state.upload_tag_date = date_obj
            st.session_state.upload_tag_date_input = date_obj
        
        # Detect document type
        detected_type = _extract_document_type(content)
        st.session_state.upload_tag_document_type = detected_type
        
        # Suggest misc tags
        suggested_tags = _suggest_misc_tags(content, num_suggestions=5)
        if suggested_tags:
            st.session_state.upload_suggested_tags = suggested_tags
            st.session_state.upload_show_suggestions = True
        
        # Mark that auto-detect was just run
        st.session_state.upload_auto_detect_just_ran = True
    except Exception as e:
        st.session_state.upload_detection_error = str(e)

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

# --- Unified search (text + inline mic) ---
st.subheader("Search")

# Initialize search filter session state
if "search_filter_logic" not in st.session_state:
    st.session_state.search_filter_logic = True  # AND by default
if "search_patient_ids" not in st.session_state:
    st.session_state.search_patient_ids = []
if "search_doc_types" not in st.session_state:
    st.session_state.search_doc_types = []
if "search_misc_tags" not in st.session_state:
    st.session_state.search_misc_tags = []

# Render search filter pills
st.markdown("**Filter search results:**")
st.markdown('<div class="pill-container">', unsafe_allow_html=True)

col_slogic, col_sdiv1, col_spatient, col_sdoctype, col_smisc = st.columns([1, 0.1, 2, 2, 2])

with col_slogic:
    # Use selectbox to match structure of other filters and allow toggle
    options = ["AND", "OR"]
    
    # Initialize selectbox value in session state if needed
    if "search_filter_logic_select" not in st.session_state:
        st.session_state.search_filter_logic_select = "AND" if st.session_state.search_filter_logic else "OR"
    
    # Let Streamlit manage the widget state via key
    selected_logic = st.selectbox(
        "Logic",
        options,
        key="search_filter_logic_select",
        label_visibility="visible"
    )
    # Sync the boolean logic state with the selectbox value
    st.session_state.search_filter_logic = (selected_logic == "AND")

with col_sdiv1:
    st.markdown('<span class="pill-divider"></span>', unsafe_allow_html=True)

with col_spatient:
    selected_search_patients = st.multiselect(
        "Patient IDs",
        patient_ids,
        default=st.session_state.search_patient_ids,
        key="search_patient_multiselect",
        label_visibility="visible"
    )
    st.session_state.search_patient_ids = selected_search_patients
    if selected_search_patients:
        display_text = f"👤 {len(selected_search_patients)} patient(s)"
        st.markdown(f'<span class="pill pill-patient">{display_text}</span>', unsafe_allow_html=True)

with col_sdoctype:
    selected_doc_types = st.multiselect(
        "Document Types",
        ["ultrasound", "pathology", "other"],
        default=st.session_state.search_doc_types,
        key="search_doc_type_multiselect",
        label_visibility="visible"
    )
    st.session_state.search_doc_types = selected_doc_types
    if selected_doc_types:
        display_text = f"📄 {', '.join(selected_doc_types)}"
        st.markdown(f'<span class="pill pill-doc-type">{display_text}</span>', unsafe_allow_html=True)

with col_smisc:
    misc_input = st.text_input(
        "Misc Tags",
        value=", ".join(st.session_state.search_misc_tags),
        key="search_misc_input",
        help="Comma-separated tags",
        label_visibility="visible"
    )
    if misc_input:
        st.session_state.search_misc_tags = [tag.strip() for tag in misc_input.split(",") if tag.strip()]
    else:
        st.session_state.search_misc_tags = []
    if st.session_state.search_misc_tags:
        display_text = f"🏷️ {', '.join(st.session_state.search_misc_tags[:2])}"
        if len(st.session_state.search_misc_tags) > 2:
            display_text += f" (+{len(st.session_state.search_misc_tags) - 2})"
        st.markdown(f'<span class="pill pill-misc">{display_text}</span>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Initialize chat session state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "bedrock_session_id" not in st.session_state:
    st.session_state.bedrock_session_id = None  # Bedrock will generate this on first request
if "input_key_counter" not in st.session_state:
    st.session_state.input_key_counter = 0

# Chat container with scrollable messages
if st.session_state.chat_messages:
    # Chat header with New Chat button
    chat_header_col1, chat_header_col2 = st.columns([1, 0.1])
    with chat_header_col1:
        st.markdown("**Chat History**")
    with chat_header_col2:
        if st.button("🔄", key="new_chat_btn", help="Start a new chat session", use_container_width=True):
            st.session_state.chat_messages = []
            st.session_state.bedrock_session_id = None  # Clear Bedrock session to start fresh
            st.session_state.input_key_counter += 1  # Reset input widget by changing key
            st.rerun()
    
    # Scrollable chat container
    chat_html = '<div class="chat-container">'
    for msg in st.session_state.chat_messages:
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        timestamp_str = timestamp.strftime("%H:%M") if timestamp else ""
        
        # Escape HTML for user messages, but allow HTML for assistant responses (which come as HTML from Lambda)
        if role == "user":
            content_escaped = html.escape(content)
        else:
            content_escaped = content  # Assistant content is already HTML from Lambda
        
        chat_html += f'<div class="chat-message {role}">'
        chat_html += f'<div class="chat-bubble {role}">'
        if role == "assistant":
            # For assistant, render HTML directly
            chat_html += f'<div>{content_escaped}</div>'
        else:
            # For user, use escaped text
            chat_html += f'<div>{content_escaped}</div>'
        if timestamp_str:
            chat_html += f'<div class="chat-timestamp">{timestamp_str}</div>'
        chat_html += '</div></div>'
    chat_html += '</div>'
    st.markdown(chat_html, unsafe_allow_html=True)
else:
    # Empty state - show New Chat button if needed
    col_empty1, col_empty2 = st.columns([1, 0.1])
    with col_empty1:
        st.info("💬 Start a conversation by asking a question below")
    with col_empty2:
        if st.button("🔄", key="new_chat_btn_empty", help="Start a new chat session", use_container_width=True, disabled=True):
            pass

# init keys once
st.session_state.setdefault("run_q", None)
st.session_state.setdefault("prefill_text", None)  # <-- temp buffer for voice text

# Get prefill text if available (before widget creation)
prefill_value = ""
if st.session_state["prefill_text"] is not None:
    prefill_value = st.session_state["prefill_text"]
    st.session_state["prefill_text"] = None

# input row: text box + mic only
c1, c2 = st.columns([12, 1])
with c1:
    st.text_input(
        "Enter your message:",
        key=f"query_text_input_{st.session_state.input_key_counter}",
        value=prefill_value,
        label_visibility="collapsed",
        placeholder="Type your message… or tap the mic"
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
do_search = st.button("Send", type="primary", use_container_width=True)

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
# Handle Send click
if do_search:
    current_input_key = f"query_text_input_{st.session_state.input_key_counter}"
    q = (st.session_state.get(current_input_key) or "").strip()
    if not q:
        st.warning("Please enter a message.")
    else:
        st.session_state["run_q"] = q
        st.rerun()

# Execute pending search (after rerun)
if st.session_state.get("run_q"):
    run_q = st.session_state["run_q"]
    st.session_state["run_q"] = None
    
    # Add user message to chat history
    st.session_state.chat_messages.append({
        "role": "user",
        "content": run_q,
        "timestamp": datetime.now()
    })
    
    # Build filter parameters from search pills
    # Use search pills if any are selected, otherwise use all patients
    if st.session_state.search_patient_ids:
        patient_ids_filter = st.session_state.search_patient_ids
    else:
        patient_ids_filter = patient_ids
    
    # Only create filter_params if there are any filters beyond patient_id
    filter_params = None
    if (st.session_state.search_doc_types or 
        st.session_state.search_misc_tags or 
        st.session_state.search_patient_ids):  # If using search pills for patients
        filter_params = {
            "filter_logic": "AND" if st.session_state.search_filter_logic else "OR",
            "patient_ids": patient_ids_filter,
            "document_types": st.session_state.search_doc_types,
            "miscellaneous_tags": st.session_state.search_misc_tags
        }
    
    with st.spinner("Thinking…"):
        # Pass Bedrock sessionId for conversation context (None for first request)
        out = search_transcript(sub, kb_id, run_q, patient_ids_filter, filter_params, st.session_state.bedrock_session_id)
    
    # Store the sessionId from response for next request
    if "sessionId" in out and out["sessionId"]:
        st.session_state.bedrock_session_id = out["sessionId"]
    
    # Add assistant response to chat history
    if "error" in out:
        assistant_content = f"Error: {out['error']}"
    elif "body" in out:
        assistant_content = out["body"]
    else:
        assistant_content = "No content returned."
    
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": assistant_content,
        "timestamp": datetime.now()
    })
    
    # Clear the input after sending by incrementing the key counter
    st.session_state.input_key_counter += 1
    st.rerun()



st.markdown("<div style='height:100px;'></div>", unsafe_allow_html=True)
st.divider()
# --- Add to knowledge base (renamed & simplified) ---
# Get KB status for the dot indicator
kb_status_color = "#6b7280"  # default gray
kb_status_text = "Unknown"
try:
    _ds_id, _bucket = _get_kb_data_source(kb_id)
    status, last_job_id, started = _latest_ingestion_status(kb_id, _ds_id)
    kb_status_color = {"COMPLETE": "#16a34a", "IN_PROGRESS": "#f59e0b", "FAILED": "#dc2626", "NO_JOBS": "#6b7280"}.get(status, "#6b7280")
    kb_status_text = f"{status.replace('_',' ')}{' • job '+last_job_id if last_job_id else ''}"
except Exception as e:
    kb_status_text = f"Status unavailable: {e}"

# Create header with status-colored dot and redo button
col_header1, col_header2, col_header3 = st.columns([0.05, 0.9, 0.05])
with col_header1:
    st.markdown(f'<div style="width:8px;height:8px;border-radius:50%;background:{kb_status_color};border:1px solid rgba(0,0,0,0.1);margin-top:8px;" title="{kb_status_text}"></div>', unsafe_allow_html=True)
with col_header2:
    st.subheader("Add to knowledge base")
with col_header3:
    if st.button("↻", help=f"KB: {kb_status_text}", key="kb_refresh_btn", use_container_width=True):
        st.toast(f"KB status: {kb_status_text}", icon="ℹ️")
        st.rerun()

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
    placeholder="Write a quick note (Markdown supported)…",
    key="note_text_area"
)

# Render tagging pills (using "note" prefix for manual notes)
tags = _render_tagging_pills(patient_ids, None, key_prefix="note")

# Auto-detect all tags if content is provided
if typed_content.strip():
    # Store patient_ids in session state for callback (content is already in session state via widget key)
    st.session_state.available_patient_ids = patient_ids
    
    # Check if content has changed since last detection
    import hashlib
    current_content_hash = hashlib.md5(typed_content.encode()).hexdigest()
    last_detected_hash = st.session_state.get("note_last_detected_content_hash")
    
    # Clear detection results if content has changed
    if last_detected_hash and current_content_hash != last_detected_hash:
        st.session_state.note_auto_detect_just_ran = False
        st.session_state.note_show_suggestions = False
        st.session_state.note_show_new_patient_confirmation = False
    
    col_auto1, col_auto2 = st.columns([2, 1])
    with col_auto1:
        st.markdown("**AI Auto-Detection:**")
    with col_auto2:
        st.button(
            "🔍 Auto-detect All", 
            key="btn_auto_detect_all_note", 
            help="Auto-detect patient ID, date, document type, and suggest tags", 
            use_container_width=True, 
            type="secondary",
            on_click=auto_detect_note_callback
        )
    
    # Show new patient confirmation if detected
    if st.session_state.get("note_show_new_patient_confirmation"):
        new_patient_id = st.session_state.get("note_detected_new_patient_id")
        if new_patient_id:
            st.warning(f"⚠️ **New Patient Detected:** `{new_patient_id}`")
            st.info("This patient ID was found in the document but is not in your patient list. Registering this patient will add them to your list and allow you to upload the document.")
            col_confirm1, col_confirm2 = st.columns([1, 1])
            with col_confirm1:
                if st.button("Register & Continue", key="note_confirm_new_patient", type="primary", use_container_width=True):
                    try:
                        res = add_patient_to_doctor(dynamo_table, sub, new_patient_id)
                        if res == "exists":
                            st.info(f"Patient {new_patient_id} is already assigned to this Clinician.")
                        else:
                            st.success(f"Patient {new_patient_id} registered successfully.")
                        # Update session state
                        st.session_state.note_tag_patient_id = new_patient_id
                        st.session_state.note_show_new_patient_confirmation = False
                        st.session_state.note_detected_new_patient_id = None
                        # Refresh patient_ids list
                        st.session_state.available_patient_ids = get_patient_ids(sub)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to register patient: {e}")
            with col_confirm2:
                if st.button("Cancel", key="note_cancel_new_patient", use_container_width=True):
                    st.session_state.note_show_new_patient_confirmation = False
                    st.session_state.note_detected_new_patient_id = None
                    st.rerun()
            st.divider()
    
    # Show detection results ONLY if auto-detect was just run and content matches
    if st.session_state.get("note_auto_detect_just_ran") and current_content_hash == st.session_state.get("note_last_detected_content_hash"):
        results = []
        if st.session_state.get("note_tag_patient_id"):
            pid = st.session_state.note_tag_patient_id
            results.append(f"Patient ID: {pid[:20]}...")
        if st.session_state.get("note_tag_date"):
            date = st.session_state.note_tag_date
            results.append(f"Date: {date}")
        if st.session_state.get("note_tag_document_type") != "other":
            results.append(f"Document Type: {st.session_state.note_tag_document_type}")
        if st.session_state.get("note_suggested_tags"):
            results.append(f"Suggested {len(st.session_state.note_suggested_tags)} tags")
    
    # Show suggested tags if available
    if st.session_state.get("note_show_suggestions") and st.session_state.get("note_suggested_tags"):
        st.markdown("**💡 Suggested Tags (click to add):**")
        suggested = st.session_state.note_suggested_tags
        current_tags = set(st.session_state.note_tag_misc)
        
        # Create columns for tag buttons
        cols = st.columns(min(5, len(suggested)))
        for idx, tag in enumerate(suggested):
            with cols[idx % len(cols)]:
                if tag not in current_tags:
                    if st.button(f"➕ {tag}", key=f"add_tag_note_{idx}", use_container_width=True):
                        st.session_state.note_tag_misc.append(tag)
                        st.rerun()
                else:
                    st.button(f"✓ {tag}", key=f"added_tag_note_{idx}", use_container_width=True, disabled=True)
        
        # Add "Select All" and "Clear suggestions" buttons
        col_select_all, col_clear = st.columns([1, 1])
        with col_select_all:
            tags_to_add = [tag for tag in suggested if tag not in current_tags]
            if tags_to_add:
                if st.button("Select All", key="select_all_note", use_container_width=True):
                    st.session_state.note_tag_misc.extend(tags_to_add)
                    st.rerun()
        with col_clear:
            if st.button("Clear suggestions", key="clear_suggestions_note", use_container_width=True):
                st.session_state.note_show_suggestions = False
                st.session_state.note_suggested_tags = []
                st.rerun()

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
    if not tags["patient_id"]:
        st.warning("Please select a patient ID in the tagging pills above.")
    elif not typed_content.strip():
        st.warning("The note is empty.")
    else:
        # Check if patient ID is in the list, if not, try to add it
        selected_patient_id = tags["patient_id"]
        current_patient_ids = get_patient_ids(sub)
        if selected_patient_id.lower() not in [pid.lower() for pid in current_patient_ids]:
            # Patient not in list - try to add it
            try:
                res = add_patient_to_doctor(dynamo_table, sub, selected_patient_id)
                if res == "exists":
                    st.info(f"Patient {selected_patient_id} is already assigned to this Clinician.")
                else:
                    st.success(f"Patient {selected_patient_id} registered successfully.")
                # Refresh patient_ids list
                patient_ids = get_patient_ids(sub)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to register patient: {e}")
                st.stop()
        
        try:
            data_source_id, bucket = _get_kb_data_source(kb_id)
            file_bytes = typed_content.encode("utf-8")
            auto_name = f"note-{datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.md"
            
            # Auto-detect document type if still "other" or None and content exists
            doc_type = tags["document_type"] if tags["document_type"] else "other"
            if (doc_type == "other" or doc_type is None) and typed_content.strip():
                with st.spinner("Auto-detecting document type..."):
                    doc_type = _extract_document_type(typed_content)
                    # Update session state so UI reflects the detected type
                    st.session_state.note_tag_document_type = doc_type

            stored_key, meta_key = _put_file_and_metadata(
                bucket=bucket,
                filename=auto_name,
                file_bytes=file_bytes,
                patient_id=tags["patient_id"],
                document_type=doc_type,
                date=tags["date"],
                miscellaneous_tags=tags["miscellaneous_tags"] if tags["miscellaneous_tags"] else None
            )
            st.success(f"Saved `{auto_name}` as `s3://{bucket}/{stored_key}` (metadata: `{meta_key}`).")

            if save_and_ingest:
                job_id = _start_ingestion(kb_id, data_source_id)
                st.info(f"Started ingestion job: {job_id}. Click 'Refresh KB status' to check progress.")
            
            # Reset note tags after successful save
            st.session_state.note_tag_patient_id = None
            st.session_state.note_tag_document_type = None
            st.session_state.note_tag_date = None
            st.session_state.note_tag_misc = []
            # Note: Cannot reset note_text_area directly as it's a widget value
            # The text area will remain, but tags are reset for next entry
        except Exception as e:
            st.error(f"Failed to save or ingest note: {e}")

# Upload files area - render tagging pills for file uploads
upload_tags = None
if uploaded_files:
    st.markdown("**Tag files before uploading:**")
    upload_tags = _render_tagging_pills(patient_ids, None, key_prefix="upload")
    
    # Auto-detect all tags for uploaded files
    # Show button if we have text-based files
    first_file = uploaded_files[0]
    can_auto_detect = first_file.name.lower().endswith(('.txt', '.md'))
    
    if can_auto_detect:
        # Store file and patient_ids in session state for callback
        st.session_state.uploaded_files_for_detection = uploaded_files
        st.session_state.available_patient_ids = patient_ids
        
        # Check if file content has changed since last detection
        try:
            first_file.seek(0)
            file_bytes = first_file.read()
            content = file_bytes.decode('utf-8', errors='ignore')
            first_file.seek(0)  # Reset for later use
            
            import hashlib
            current_content_hash = hashlib.md5(content.encode()).hexdigest()
            last_detected_hash = st.session_state.get("upload_last_detected_content_hash")
            
            # Clear detection results if content has changed
            if last_detected_hash and current_content_hash != last_detected_hash:
                st.session_state.upload_auto_detect_just_ran = False
                st.session_state.upload_show_suggestions = False
                st.session_state.upload_show_new_patient_confirmation = False
        except:
            current_content_hash = None
            last_detected_hash = None
        
        st.markdown("---")
        col_auto1, col_auto2 = st.columns([2, 1])
        with col_auto1:
            st.markdown("**AI Auto-Detection:**")
        with col_auto2:
            st.button(
                "🔍 Auto-detect All", 
                key="btn_auto_detect_all_upload", 
                help="Auto-detect patient ID, date, document type, and suggest tags", 
                use_container_width=True, 
                type="secondary",
                on_click=auto_detect_upload_callback
            )
        
        # Show detection error if any
        if st.session_state.get("upload_detection_error"):
            st.error(f"Failed to read file for auto-detection: {st.session_state.upload_detection_error}")
            st.session_state.upload_detection_error = None
        
        # Show new patient confirmation if detected
        if st.session_state.get("upload_show_new_patient_confirmation"):
            new_patient_id = st.session_state.get("upload_detected_new_patient_id")
            if new_patient_id:
                st.warning(f"⚠️ **New Patient Detected:** `{new_patient_id}`")
                st.info("This patient ID was found in the document but is not in your patient list. Registering this patient will add them to your list and allow you to upload the document.")
                col_confirm1, col_confirm2 = st.columns([1, 1])
                with col_confirm1:
                    if st.button("Register & Continue", key="upload_confirm_new_patient", type="primary", use_container_width=True):
                        try:
                            res = add_patient_to_doctor(dynamo_table, sub, new_patient_id)
                            if res == "exists":
                                st.info(f"Patient {new_patient_id} is already assigned to this Clinician.")
                            else:
                                st.success(f"Patient {new_patient_id} registered successfully.")
                            # Update session state
                            st.session_state.upload_tag_patient_id = new_patient_id
                            st.session_state.upload_show_new_patient_confirmation = False
                            st.session_state.upload_detected_new_patient_id = None
                            # Refresh patient_ids list
                            st.session_state.available_patient_ids = get_patient_ids(sub)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to register patient: {e}")
                with col_confirm2:
                    if st.button("Cancel", key="upload_cancel_new_patient", use_container_width=True):
                        st.session_state.upload_show_new_patient_confirmation = False
                        st.session_state.upload_detected_new_patient_id = None
                        st.rerun()
                st.divider()
        
        # Show detection results ONLY if auto-detect was just run and content matches
        if (st.session_state.get("upload_auto_detect_just_ran") and 
            current_content_hash and 
            current_content_hash == st.session_state.get("upload_last_detected_content_hash")):
            results = []
            if st.session_state.get("upload_tag_patient_id"):
                pid = st.session_state.upload_tag_patient_id
                results.append(f"Patient ID: {pid[:20]}...")
            if st.session_state.get("upload_tag_date"):
                date = st.session_state.upload_tag_date
                results.append(f"Date: {date}")
            if st.session_state.get("upload_tag_document_type") != "other":
                results.append(f"Document Type: {st.session_state.upload_tag_document_type}")
            if st.session_state.get("upload_suggested_tags"):
                results.append(f"Suggested {len(st.session_state.upload_suggested_tags)} tags")
        
        # Show suggested tags if available
        if st.session_state.get("upload_show_suggestions") and st.session_state.get("upload_suggested_tags"):
            st.markdown("**💡 Suggested Tags (click to add):**")
            suggested = st.session_state.upload_suggested_tags
            current_tags = set(st.session_state.upload_tag_misc)
            
            # Create columns for tag buttons
            cols = st.columns(min(5, len(suggested)))
            for idx, tag in enumerate(suggested):
                with cols[idx % len(cols)]:
                    if tag not in current_tags:
                        if st.button(f"➕ {tag}", key=f"add_tag_upload_{idx}", use_container_width=True):
                            st.session_state.upload_tag_misc.append(tag)
                            st.rerun()
                    else:
                        st.button(f"✓ {tag}", key=f"added_tag_upload_{idx}", use_container_width=True, disabled=True)
            
            # Add "Select All" and "Clear suggestions" buttons
            col_select_all, col_clear = st.columns([1, 1])
            with col_select_all:
                tags_to_add = [tag for tag in suggested if tag not in current_tags]
                if tags_to_add:
                    if st.button("✅ Select All", key="select_all_upload", use_container_width=True):
                        st.session_state.upload_tag_misc.extend(tags_to_add)
                        st.rerun()
            with col_clear:
                if st.button("Clear suggestions", key="clear_suggestions_upload", use_container_width=True):
                    st.session_state.upload_show_suggestions = False
                    st.session_state.upload_suggested_tags = []
                    st.rerun()
        
        st.markdown("---")

col1, col2 = st.columns([1, 1])
with col1:
    do_ingest = st.button("Upload & Start Ingestion", type="primary")
with col2:
    just_upload = st.button("Upload Only")

if (do_ingest or just_upload):
    if not upload_tags or not upload_tags.get("patient_id"):
        st.warning("Please select a patient ID in the tagging pills above.")
    elif not uploaded_files:
        st.warning("Please pick at least one file.")
    else:
        # Check if patient ID is in the list, if not, try to add it
        selected_patient_id = upload_tags.get("patient_id")
        current_patient_ids = get_patient_ids(sub)
        if selected_patient_id and selected_patient_id.lower() not in [pid.lower() for pid in current_patient_ids]:
            # Patient not in list - try to add it
            try:
                res = add_patient_to_doctor(dynamo_table, sub, selected_patient_id)
                if res == "exists":
                    st.info(f"Patient {selected_patient_id} is already assigned to this Clinician.")
                else:
                    st.success(f"Patient {selected_patient_id} registered successfully.")
                # Refresh patient_ids list
                patient_ids = get_patient_ids(sub)
                st.rerun()
            except Exception as e:
                st.error(f"Failed to register patient: {e}")
                st.stop()
        
        try:
            data_source_id, bucket = _get_kb_data_source(kb_id)
            uploaded = []
            for f in uploaded_files:
                key = f.name  # keep original filename
                file_bytes = f.read()
                
                # Use the document type from tags (which may have been auto-detected)
                doc_type = upload_tags["document_type"] if upload_tags["document_type"] else "other"
                
                # If still "other" or None, try to auto-detect during upload (but don't update UI at this point)
                if doc_type == "other" or doc_type is None:
                    # Try to read text from file for document type detection
                    try:
                        if key.lower().endswith('.txt') or key.lower().endswith('.md'):
                            content = file_bytes.decode('utf-8', errors='ignore')
                            with st.spinner(f"Auto-detecting type for {key}..."):
                                doc_type = _extract_document_type(content)
                                # Update session state so UI reflects the detected type
                                st.session_state.upload_tag_document_type = doc_type
                        elif key.lower().endswith('.pdf'):
                            # For PDF, we'd need pdf parsing, but for now default to other
                            # In production, you might want to use PyPDF2 or similar
                            pass
                    except Exception:
                        pass  # Keep as "other" if extraction fails
                
                # Reset file pointer for upload
                f.seek(0)
                file_bytes = f.read()
                
                _k, _meta_key = _put_file_and_metadata(
                    bucket=bucket,
                    filename=key,
                    file_bytes=file_bytes,
                    patient_id=upload_tags["patient_id"],
                    document_type=doc_type,
                    date=upload_tags["date"],
                    miscellaneous_tags=upload_tags["miscellaneous_tags"] if upload_tags["miscellaneous_tags"] else None
                )
                uploaded.append((key, _meta_key))
            st.success(f"Uploaded {len(uploaded)} file(s) to s3://{bucket}/ (with metadata).")

            if do_ingest:
                job_id = _start_ingestion(kb_id, data_source_id)
                st.info(f"Started ingestion job: {job_id}. Click 'Refresh KB status' above to check progress.")
            
            # Reset upload tags after successful upload
            st.session_state.upload_tag_patient_id = None
            st.session_state.upload_tag_document_type = None
            st.session_state.upload_tag_date = None
            st.session_state.upload_tag_misc = []
        except Exception as e:
            st.error(f"Upload/ingest failed: {e}")
