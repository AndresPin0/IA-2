import json
import requests
import streamlit as st
from datetime import datetime

OLLAMA_BASE = "http://localhost:11434"
CHAT_API = f"{OLLAMA_BASE}/api/chat"
TAGS_API = f"{OLLAMA_BASE}/api/tags"

st.set_page_config(page_title="Ollama Chat Pro", page_icon="🤖", layout="wide")
st.title("🤖 Ollama Chat Pro")

# -------- Helpers --------
@st.cache_data(ttl=10)
def list_local_models():
    try:
        r = requests.get(TAGS_API, timeout=5)
        r.raise_for_status()
        data = r.json()
        # Return display name like "llama3.1:8b" if available, else "name:tag"
        models = []
        for m in data.get("models", []):
            name = m.get("name")  # e.g., "llama3.1:8b"
            if name:
                models.append(name)
            else:
                # fallback
                models.append(f'{m.get("model","unknown")}:{m.get("tag","latest")}')
        # Deduplicate & sort
        return sorted(list(set(models)))
    except Exception:
        # fallback defaults
        return ["mistral", "llama3.1:8b", "gemma:7b", "phi3"]

def stream_chat(model, messages, options):
    """
    Generator that yields streamed chunks from Ollama /api/chat.
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": options,
    }
    with requests.post(CHAT_API, json=payload, stream=True) as r:
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("done"):
                yield {"done": True, "final": obj}
                break
            msg = obj.get("message", {})
            chunk = msg.get("content", "")
            if chunk:
                yield {"done": False, "content": chunk}

def call_chat_once(model, messages, options):
    """
    Non-stream call (fallback).
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    r = requests.post(CHAT_API, json=payload)
    r.raise_for_status()
    return r.json()["message"]["content"]

def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "model" not in st.session_state:
        st.session_state.model = "llama3.1:8b"
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = "Eres un asistente útil y conciso. Responde en español."
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "top_p" not in st.session_state:
        st.session_state.top_p = 0.9
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 512
    if "seed" not in st.session_state:
        st.session_state.seed = None
    if "streaming" not in st.session_state:
        st.session_state.streaming = True

init_state()

# -------- Sidebar (settings) --------
with st.sidebar:
    st.header("⚙️ Configuración")
    models = list_local_models()
    st.session_state.model = st.selectbox("Modelo", models, index=models.index(st.session_state.model) if st.session_state.model in models else 0)
    st.session_state.system_prompt = st.text_area("System prompt", st.session_state.system_prompt, height=100, help="Contexto/rol del asistente.")
    cols = st.columns(2)
    with cols[0]:
        st.session_state.temperature = st.slider("Temperature", 0.0, 1.5, float(st.session_state.temperature), 0.05)
        st.session_state.top_p = st.slider("Top-p", 0.0, 1.0, float(st.session_state.top_p), 0.05)
    with cols[1]:
        st.session_state.max_tokens = st.number_input("Máx. tokens respuesta", min_value=16, max_value=8192, value=int(st.session_state.max_tokens), step=16)
        seed_in = st.text_input("Seed (opcional)", value=str(st.session_state.seed) if st.session_state.seed is not None else "")
        st.session_state.seed = int(seed_in) if seed_in.strip().isdigit() else None

    st.session_state.streaming = st.toggle("Streaming en tiempo real", value=st.session_state.streaming)

    # Botones utilitarios
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🧹 Limpiar chat"):
            st.session_state.messages = []
            st.experimental_rerun()
    with c2:
        # Descargar historial
        export = {
            "model": st.session_state.model,
            "system_prompt": st.session_state.system_prompt,
            "messages": st.session_state.messages,
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        st.download_button("💾 Descargar historial", data=json.dumps(export, ensure_ascii=False, indent=2), file_name="ollama_chat_history.json", mime="application/json")

# -------- Inicializar historial con system prompt si no está --------
def ensure_system_message():
    if not st.session_state.messages or st.session_state.messages[0].get("role") != "system":
        st.session_state.messages.insert(0, {"role": "system", "content": st.session_state.system_prompt})
    else:
        # Mantenerlo actualizado si cambió en el sidebar
        st.session_state.messages[0]["content"] = st.session_state.system_prompt

ensure_system_message()

# -------- Mostrar historial --------
for msg in st.session_state.messages:
    role = msg["role"]
    if role == "system":
        continue
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(msg["content"])

# -------- Entrada de usuario --------
user_text = st.chat_input("Escribe tu mensaje…")
if user_text:
    # Append user msg
    st.session_state.messages.append({"role": "user", "content": user_text})
    with st.chat_message("user"):
        st.markdown(user_text)

    # Prepare assistant container
    with st.chat_message("assistant"):
        placeholder = st.empty()
        accumulated = ""

        # Build options
        options = {
            "temperature": float(st.session_state.temperature),
            "top_p": float(st.session_state.top_p),
            "num_predict": int(st.session_state.max_tokens),
        }
        if st.session_state.seed is not None:
            options["seed"] = int(st.session_state.seed)

        try:
            if st.session_state.streaming:
                for chunk in stream_chat(
                    model=st.session_state.model,
                    messages=st.session_state.messages,
                    options=options,
                ):
                    if chunk.get("done"):
                        # done event; nothing else to display here
                        continue
                    piece = chunk.get("content", "")
                    if piece:
                        accumulated += piece
                        placeholder.markdown(accumulated)
            else:
                reply = call_chat_once(
                    model=st.session_state.model,
                    messages=st.session_state.messages,
                    options=options,
                )
                accumulated = reply
                placeholder.markdown(accumulated)

        except requests.HTTPError as e:
            accumulated = f"❌ Error HTTP: {e.response.status_code} - {e.response.text}"
            placeholder.error(accumulated)
        except requests.ConnectionError:
            accumulated = "❌ No se puede conectar con Ollama. ¿Está ejecutándose en http://localhost:11434?"
            placeholder.error(accumulated)
        except Exception as e:
            accumulated = f"❌ Error: {e}"
            placeholder.error(accumulated)

        # Guardar respuesta en el historial
        st.session_state.messages.append({"role": "assistant", "content": accumulated})