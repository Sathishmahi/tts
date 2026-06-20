"""
Qwen3-TTS — Streamlit Frontend (0.6B-only build)
Interactive UI for all 3 generation modes: Custom Voice, Voice Design, Voice Clone.
Trimmed for low-VRAM laptops (e.g. RTX 3050 6GB / 16GB RAM).

Custom Voice & Voice Clone always use the 0.6B checkpoints.
Voice Design always uses 1.7B — Qwen doesn't publish a 0.6B VoiceDesign model,
so that tab will use more VRAM than the other two.

Run with:  streamlit run frontend.py
"""

from typing import Optional

import requests
import streamlit as st

BACKEND_URL = "http://localhost:8000"

# ─── Static reference data (kept in sync with backend.py) ─────────────────────
SPEAKERS = {
    "Vivian":   {"gender": "Female", "native_language": "Chinese",                   "description": "Bright, slightly edgy young female voice.",            "emoji": "🇨🇳"},
    "Serena":   {"gender": "Female", "native_language": "Chinese",                   "description": "Warm, gentle young female voice.",                      "emoji": "🇨🇳"},
    "Uncle_fu": {"gender": "Male",   "native_language": "Chinese",                   "description": "Seasoned male voice, low mellow timbre.",                "emoji": "🇨🇳"},
    "Dylan":    {"gender": "Male",   "native_language": "Chinese (Beijing dialect)", "description": "Youthful Beijing male voice, clear and natural.",       "emoji": "🇨🇳"},
    "Eric":     {"gender": "Male",   "native_language": "Chinese (Sichuan dialect)", "description": "Lively Chengdu male voice, slightly husky brightness.", "emoji": "🇨🇳"},
    "Ryan":     {"gender": "Male",   "native_language": "English",                   "description": "Dynamic male voice with strong rhythmic drive.",        "emoji": "🇺🇸"},
    "Aiden":    {"gender": "Male",   "native_language": "English",                   "description": "Sunny American male voice, clear midrange.",            "emoji": "🇺🇸"},
    "Ono_anna": {"gender": "Female", "native_language": "Japanese",                  "description": "Playful Japanese female voice, light and nimble.",      "emoji": "🇯🇵"},
    "Sohee":    {"gender": "Female", "native_language": "Korean",                    "description": "Warm Korean female voice with rich emotion.",           "emoji": "🇰🇷"},
}

LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "German", "French", "Russian", "Portuguese", "Spanish", "Italian",
]

VOICE_DESIGN_PRESETS = [
    "A cheerful young female voice with high pitch and energetic tone.",
    "A deep, resonant male voice, narrator style, calm and professional.",
    "Warm, confident narrator with slight British accent.",
    "An elderly grandmother's voice, soft and slow, full of warmth.",
    "A robotic, monotone synthetic voice with no emotion.",
    "Young male fitness coach voice, confident, motivational, energetic but not aggressive.",
]

# ─── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Qwen3-TTS Demo (0.6B)", page_icon="🗣️", layout="wide")

st.markdown(
    """
    <style>
        .speaker-card { border: 1px solid #333; border-radius: 10px; padding: 10px 14px; margin-bottom: 6px; }
        .param-caption { font-size: 0.78rem; color: #888; margin-top: -8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🗣️ Qwen3-TTS Demo — 0.6B build")
st.caption(
    "Custom Voice (0.6B) · Voice Design (1.7B — no 0.6B checkpoint exists) · Voice Clone (0.6B) · 10 languages · "
    "[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)"
)
st.caption("⚙️ Tuned for low-VRAM laptops (e.g. RTX 3050 6GB / 16GB RAM).")

# Backend status
status = st.empty()
try:
    r = requests.get(f"{BACKEND_URL}/health", timeout=3)
    h = r.json()
    loaded = h.get("loaded_models", [])
    status.success(f"✅ Backend connected · Loaded checkpoints: {', '.join(loaded) if loaded else 'none yet (loads on first request)'}")
except Exception:
    status.error("❌ Cannot connect to backend. Start it with: `python backend.py`")

st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# MODE TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_custom, tab_design, tab_clone = st.tabs(["🎙️ Custom Voice", "🎨 Voice Design", "🧬 Voice Clone"])

SAMPLE_RATE = 24000


def advanced_sampling_controls(key_prefix: str):
    """Shared sampling/generation-kwarg controls used by every mode."""
    with st.expander("⚙️ Advanced generation parameters", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            max_new_tokens = st.slider(
                "max_new_tokens", 256, 4096, 2048, step=128,
                key=f"{key_prefix}_mnt",
                help="Upper bound on generated audio tokens. Lower = shorter max output / faster cutoff.",
            )
            top_p = st.slider(
                "top_p", 0.1, 1.0, 0.9, step=0.05,
                key=f"{key_prefix}_topp",
                help="Nucleus sampling threshold. Lower = more deterministic/focused.",
            )
            top_k = st.slider(
                "top_k", 0, 100, 50, step=5,
                key=f"{key_prefix}_topk",
                help="Restrict sampling to the k most likely tokens. 0 disables top-k filtering.",
            )
        with c2:
            temperature = st.slider(
                "temperature", 0.1, 2.0, 0.9, step=0.05,
                key=f"{key_prefix}_temp",
                help="Sampling randomness. Lower = more stable/monotone, higher = more expressive/varied.",
            )
            repetition_penalty = st.slider(
                "repetition_penalty", 1.0, 2.0, 1.05, step=0.05,
                key=f"{key_prefix}_reppen",
                help="Penalizes repeated tokens. Helps avoid stutter/looping artifacts.",
            )
        st.caption("These map directly to HuggingFace `model.generate(...)` kwargs supported by Qwen3-TTS.")
        return dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
        )


def render_result(resp: requests.Response, filename: str):
    if resp.status_code == 200:
        audio_bytes = resp.content
        st.success("✅ Audio generated!")
        st.audio(audio_bytes, format="audio/wav")
        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            st.download_button("⬇️ Download WAV", data=audio_bytes, file_name=filename,
                                mime="audio/wav", use_container_width=True)
        with col_info:
            size_kb = len(audio_bytes) / 1024
            approx_sec = len(audio_bytes) / (SAMPLE_RATE * 2)
            st.markdown(f"📊 `{size_kb:.1f} KB` · ~`{approx_sec:.1f}s` · `24 kHz`")
    else:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        st.error(f"❌ Generation failed (HTTP {resp.status_code}): {detail}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CUSTOM VOICE (0.6B only)
# ══════════════════════════════════════════════════════════════════════════════
with tab_custom:
    st.subheader("🎙️ Custom Voice — 9 premium preset speakers")
    st.caption("🟢 Running on the 0.6B checkpoint — lightest mode, best fit for 6GB VRAM.")

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown("**Choose a speaker**")
        speaker = st.selectbox(
            "speaker",
            options=list(SPEAKERS.keys()),
            format_func=lambda s: f"{SPEAKERS[s]['emoji']} {s} — {SPEAKERS[s]['native_language']}",
            label_visibility="collapsed",
            key="cv_speaker",
        )
        info = SPEAKERS[speaker]
        st.markdown(
            f"""<div class="speaker-card">
            <b>{info['emoji']} {speaker}</b> · {info['gender']} · native: {info['native_language']}<br>
            <span style="color:#aaa;">{info['description']}</span><br>
            <span style="color:#888; font-size:0.8rem;">💡 Best quality in native language, but can speak any of the 10 supported languages.</span>
            </div>""",
            unsafe_allow_html=True,
        )

        language_cv = st.selectbox("Language", LANGUAGES, key="cv_lang",
                                    help="'Auto' lets the model adapt to the text automatically.")

        st.caption(
            "ℹ️ Natural-language style instructions (e.g. 'speak happily') are a "
            "1.7B-only feature and aren't available on the 0.6B checkpoint."
        )

        gen_kwargs_cv = advanced_sampling_controls("cv")

    with col_right:
        text_cv = st.text_area(
            "Text to synthesize", height=200, key="cv_text",
            placeholder="Type the text you want this speaker to say…",
        )
        st.caption(f"{len(text_cv)} characters")

        if st.button("🎙️ Generate (Custom Voice)", type="primary", use_container_width=True,
                      disabled=not text_cv.strip(), key="cv_btn"):
            with st.spinner("Generating…"):
                try:
                    data = {
                        "text": text_cv, "speaker": speaker, "language": language_cv,
                        **{k: str(v) for k, v in gen_kwargs_cv.items()},
                    }
                    resp = requests.post(f"{BACKEND_URL}/generate/custom-voice", data=data, timeout=1600)
                    render_result(resp, "custom_voice.wav")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend.")
                except Exception as exc:
                    st.error(f"❌ {exc}")

    with st.expander("🗂️ All 9 speakers at a glance"):
        for name, d in SPEAKERS.items():
            st.markdown(f"**{d['emoji']} {name}** ({d['gender']}, {d['native_language']}) — {d['description']}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — VOICE DESIGN (1.7B — no 0.6B checkpoint exists)
# ══════════════════════════════════════════════════════════════════════════════
with tab_design:
    st.subheader("🎨 Voice Design — invent a voice purely from text")
    st.warning(
        "⚠️ This mode always runs on **Qwen3-TTS-12Hz-1.7B-VoiceDesign** — Qwen does not "
        "publish a 0.6B VoiceDesign checkpoint, so this is the only size available for it. "
        "It needs more VRAM than the other two tabs (~3.5GB+). On a 6GB card this should "
        "still fit, but close other GPU-heavy apps first if you hit an out-of-memory error.",
        icon="⚠️",
    )

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        language_vd = st.selectbox("Language", LANGUAGES, key="vd_lang")

        instruct_vd = st.text_area(
            "Voice description *", height=140, key="vd_instruct",
            placeholder="e.g. Warm, confident narrator with slight British accent.",
            help="Describe gender, age, pitch, accent, tone, pacing, emotion — anything.",
        )
        st.caption("**Quick presets:**")
        for p in VOICE_DESIGN_PRESETS:
            if st.button(f"📋 {p[:48]}{'…' if len(p) > 48 else ''}", key=f"vd_preset_{hash(p)}", use_container_width=True):
                st.session_state["vd_instruct"] = p
                st.rerun()

        gen_kwargs_vd = advanced_sampling_controls("vd")

    with col_right:
        text_vd = st.text_area(
            "Text to synthesize", height=200, key="vd_text",
            placeholder="Type the text this designed voice should say…",
        )
        st.caption(f"{len(text_vd)} characters")

        ready_vd = bool(text_vd.strip()) and bool(instruct_vd.strip())
        if st.button("🎨 Generate (Voice Design)", type="primary", use_container_width=True,
                      disabled=not ready_vd, key="vd_btn"):
            with st.spinner("Designing voice & generating…"):
                try:
                    data = {
                        "text": text_vd, "instruct": instruct_vd, "language": language_vd,
                        **{k: str(v) for k, v in gen_kwargs_vd.items()},
                    }
                    resp = requests.post(f"{BACKEND_URL}/generate/voice-design", data=data, timeout=180)
                    render_result(resp, "voice_design.wav")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend.")
                except Exception as exc:
                    st.error(f"❌ {exc}")
        if not instruct_vd.strip() and text_vd.strip():
            st.warning("⚠️ A voice description is required.", icon="⚠️")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — VOICE CLONE (0.6B only)
# ══════════════════════════════════════════════════════════════════════════════
with tab_clone:
    st.subheader("🧬 Voice Clone — clone any voice from a short reference clip")
    st.caption("🟢 Running on the 0.6B Base checkpoint (3+ seconds of reference audio is enough).")

    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        ref_source = st.radio("Reference audio source", ["📁 Upload file", "🔗 URL"], key="vc_source", horizontal=True)
        ref_audio_file, ref_audio_url = None, ""
        if ref_source == "📁 Upload file":
            ref_audio_file = st.file_uploader(
                "Reference audio *", type=["wav", "mp3", "flac", "ogg", "m4a"], key="vc_upload",
            )
            if ref_audio_file:
                st.audio(ref_audio_file)
        else:
            ref_audio_url = st.text_input("Reference audio URL *", key="vc_url",
                                           placeholder="https://example.com/clone.wav")

        x_vector_only = st.toggle(
            "⚡ X-Vector-only mode", value=False, key="vc_xvec",
            help=(
                "ON: skip the transcript — only the speaker embedding is used "
                "(faster, reusable, slightly lower fidelity).\n\n"
                "OFF (default): full ICL mode — requires an accurate transcript "
                "of the reference audio for best timbre matching."
            ),
        )

        ref_text = st.text_area(
            "Reference transcript" + ("" if x_vector_only else " *"),
            height=100, key="vc_reftext", disabled=x_vector_only,
            placeholder="Exact transcript of what's said in the reference audio…",
            help="Must match the reference audio exactly. Skip this only if X-Vector-only mode is ON.",
        )

        language_vc = st.selectbox(
            "Output language", LANGUAGES, key="vc_lang",
            help="Can differ from the reference audio's language — cross-lingual cloning is supported.",
        )

        gen_kwargs_vc = advanced_sampling_controls("vc")

    with col_right:
        text_vc = st.text_area(
            "Text to synthesize", height=200, key="vc_text",
            placeholder="Type the new content the cloned voice should say…",
        )
        st.caption(f"{len(text_vc)} characters")

        has_ref = bool(ref_audio_file) or bool(ref_audio_url.strip())
        has_transcript = x_vector_only or bool(ref_text.strip())
        ready_vc = bool(text_vc.strip()) and has_ref and has_transcript

        if st.button("🧬 Generate (Voice Clone)", type="primary", use_container_width=True,
                      disabled=not ready_vc, key="vc_btn"):
            with st.spinner("Extracting voice & generating…"):
                try:
                    data = {
                        "text": text_vc,
                        "language": language_vc,
                        "x_vector_only_mode": str(x_vector_only).lower(),
                        **{k: str(v) for k, v in gen_kwargs_vc.items()},
                    }
                    if not x_vector_only:
                        data["ref_text"] = ref_text.strip()
                    if ref_audio_url.strip():
                        data["ref_audio_url"] = ref_audio_url.strip()

                    files = None
                    if ref_audio_file is not None:
                        ref_audio_file.seek(0)
                        files = {"ref_audio": (ref_audio_file.name, ref_audio_file.read(), "audio/wav")}

                    resp = requests.post(
                        f"{BACKEND_URL}/generate/voice-clone", data=data, files=files, timeout=180,
                    )
                    render_result(resp, "voice_clone.wav")
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to backend.")
                except Exception as exc:
                    st.error(f"❌ {exc}")

        if text_vc.strip() and not has_ref:
            st.warning("⚠️ Provide a reference audio file or URL.", icon="⚠️")
        elif text_vc.strip() and not has_transcript:
            st.warning("⚠️ Provide a reference transcript, or enable X-Vector-only mode.", icon="⚠️")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) · "
    "FastAPI backend · Streamlit frontend · 24 kHz output · 0.6B build · "
    "10 languages: " + ", ".join(LANGUAGES[1:])
)