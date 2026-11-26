from __future__ import annotations

import json
import re
import uuid
import zipfile
from datetime import date, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterator, Tuple

import pandas as pd
import streamlit as st

from backend_visuals_v2 import app 

APP_DIR = Path(__file__).resolve().parent
HISTORY_DIR = APP_DIR / "blog_history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def safe_slug(title: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9_-]+", "_", (title or "blog").strip().lower())
    return value.strip("_")[:70] or "blog"


def json_safe(value: Any) -> Any:
    """Convert LangGraph/Pydantic values into JSON-persistable data."""
    if hasattr(value, "model_dump"):
        return json_safe(value.model_dump())
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def try_stream(graph_app, inputs: Dict[str, Any]) -> Iterator[Tuple[str, Any]]:
    latest_state: Optional[Dict[str, Any]] = None
    for state in graph_app.stream(inputs, stream_mode="values"):
        if isinstance(state, dict):
            latest_state = state
            yield "values", state
    if latest_state is None:
        raise RuntimeError("LangGraph finished without returning a state.")
    yield "final", latest_state


def extract_latest_state(current: Dict[str, Any], payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        if len(payload) == 1 and isinstance(next(iter(payload.values())), dict):
            current.update(next(iter(payload.values())))
        else:
            current.update(payload)
    return current


def plan_to_dict(plan: Any) -> Optional[dict]:
    if plan is None:
        return None
    return json_safe(plan)


def get_blog_title(out: dict) -> str:
    plan = plan_to_dict(out.get("plan")) or {}
    if plan.get("blog_title"):
        return str(plan["blog_title"])
    for line in str(out.get("final") or "").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "Untitled blog"


def save_history(out: dict, logs: List[str]) -> Path:
    blog_id = str(out.get("blog_id") or uuid.uuid4().hex[:12])
    blog_dir = HISTORY_DIR / blog_id
    images_dir = blog_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    final_md = str(out.get("final") or "")
    (blog_dir / "blog.md").write_text(final_md, encoding="utf-8")
    record = json_safe(out)
    record.update({
        "blog_id": blog_id,
        "blog_title": get_blog_title(out),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "logs": list(logs),
        "final": final_md,
        "blog_dir": str(blog_dir),
    })
    (blog_dir / "record.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return blog_dir


def load_history_records() -> List[dict]:
    records = []
    for record_path in HISTORY_DIR.glob("*/record.json"):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            if record.get("final") is None:
                md_path = record_path.parent / "blog.md"
                record["final"] = md_path.read_text(encoding="utf-8") if md_path.exists() else ""
            record["_record_path"] = str(record_path)
            records.append(record)
        except Exception as exc:
            records.append({
                "blog_id": record_path.parent.name,
                "blog_title": f"Unreadable history item ({record_path.parent.name})",
                "final": "",
                "_error": str(exc),
                "_record_path": str(record_path),
            })
    # Keep older root-level Markdown blogs visible in the history selector.
    for legacy_md in APP_DIR.glob("*.md"):
        try:
            content = legacy_md.read_text(encoding="utf-8", errors="replace")
            title = next((line[2:].strip() for line in content.splitlines() if line.startswith("# ")), legacy_md.stem)
            records.append({
                "blog_id": f"legacy-{legacy_md.stem}",
                "blog_title": title,
                "created_at": datetime.fromtimestamp(legacy_md.stat().st_mtime).astimezone().isoformat(timespec="seconds"),
                "final": content,
                "plan": None,
                "evidence": [],
                "image_specs": [],
                "image_manifest": [],
                "logs": ["Loaded from a legacy Markdown file; historical plan/evidence/logs were not stored."],
                "_legacy_path": str(legacy_md),
            })
        except Exception:
            continue
    records.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return records


def display_local_image(image_path: Path, caption: Optional[str] = None) -> None:
    if image_path.suffix.lower() == ".svg":
        svg_markup = image_path.read_text(encoding="utf-8", errors="replace")
        st.markdown(svg_markup, unsafe_allow_html=True)
        if caption:
            st.caption(caption)
    else:
        st.image(str(image_path), caption=caption, use_container_width=True)


def render_markdown_with_local_images(md: str) -> None:
    image_re = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
    caption_re = re.compile(r"^\*(?P<caption>.+)\*$")
    matches = list(image_re.finditer(md))
    if not matches:
        st.markdown(md, unsafe_allow_html=False)
        return
    last = 0
    parts: List[tuple[str, str, str]] = []
    for match in matches:
        if match.start() > last:
            parts.append(("md", md[last:match.start()], ""))
        parts.append(("img", match.group("alt").strip(), match.group("src").strip()))
        last = match.end()
    if last < len(md):
        parts.append(("md", md[last:], ""))

    index = 0
    while index < len(parts):
        kind, first, second = parts[index]
        if kind == "md":
            if first.strip():
                st.markdown(first, unsafe_allow_html=False)
            index += 1
            continue
        caption = first or None
        if index + 1 < len(parts) and parts[index + 1][0] == "md":
            next_md = parts[index + 1][1]
            lines = next_md.lstrip().splitlines()
            if lines:
                cap_match = caption_re.match(lines[0].strip())
                if cap_match:
                    caption = cap_match.group("caption").strip()
                    consumed = next_md.lstrip().find(lines[0]) + len(lines[0])
                    parts[index + 1] = ("md", next_md[:len(next_md)-len(next_md.lstrip())] + next_md.lstrip()[consumed:].lstrip("\n"), "")
        if second.startswith(("http://", "https://")):
            st.image(second, caption=caption, use_container_width=True)
        else:
            image_path = (APP_DIR / second).resolve()
            if image_path.exists() and image_path.is_file():
                display_local_image(image_path, caption)
            else:
                st.warning(f"Image file is missing: {second}")
        index += 1


def make_bundle(blog_dir: Path) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file_path in blog_dir.rglob("*"):
            if file_path.is_file():
                archive_path = Path("blog_history") / blog_dir.name / file_path.relative_to(blog_dir)
                archive.write(file_path, arcname=archive_path.as_posix())
    return buffer.getvalue()


def show_plan(plan_value: Any) -> None:
    plan = plan_to_dict(plan_value) or {}
    if not plan:
        st.info("No plan was returned for this blog.")
        return
    st.write("**Title:**", plan.get("blog_title", ""))
    cols = st.columns(3)
    cols[0].write("**Audience:** " + str(plan.get("audience", "")))
    cols[1].write("**Tone:** " + str(plan.get("tone", "")))
    cols[2].write("**Blog kind:** " + str(plan.get("blog_kind", "")))
    if plan.get("constraints"):
        st.write("**Constraints:**", plan["constraints"])
    tasks = plan.get("tasks", [])
    if tasks:
        frame = pd.DataFrame([{
            "id": task.get("id"),
            "title": task.get("title"),
            "target_words": task.get("target_words"),
            "requires_research": task.get("requires_research"),
            "requires_citations": task.get("requires_citations"),
            "requires_code": task.get("requires_code"),
            "tags": ", ".join(task.get("tags") or []),
        } for task in tasks])
        st.dataframe(frame, use_container_width=True, hide_index=True)
        with st.expander("Full task details"):
            st.json(tasks)


st.set_page_config(page_title="Blog Writing Agent", layout="wide")
st.title("Blog Writing Agent")
st.caption("Generate, review, and keep every blog with its plan, evidence, images, and logs.")

if "last_out" not in st.session_state:
    st.session_state.last_out = None
if "last_logs" not in st.session_state:
    st.session_state.last_logs = []

with st.sidebar:
    st.header("Generate New Blog")
    topic = st.text_area("Topic", height=120, key="topic_input", placeholder="Enter the topic for your blog...")
    as_of = st.date_input("As-of date", value=date.today())
    run_btn = st.button("🚀 Generate Blog", type="primary", use_container_width=True)
    st.divider()
    st.subheader("Blog History")
    history_records = load_history_records()
    if history_records:
        labels = [f"{r.get('blog_title', 'Untitled')} · {r.get('created_at', '')[:16]} · {r.get('blog_id', '')}" for r in history_records]
        selected_index = st.selectbox("Saved blogs", range(len(labels)), format_func=lambda i: labels[i], key="history_select")
        if st.button("📂 Load selected blog", use_container_width=True):
            chosen = history_records[selected_index]
            st.session_state.last_out = chosen
            st.session_state.last_logs = chosen.get("logs", [])
            st.rerun()
    else:
        st.caption("Your generated blogs will appear here and remain available after restarting the app.")

if run_btn:
    if not topic.strip():
        st.warning("Please enter a topic before generating.")
        st.stop()
    blog_id = datetime.now().strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
    blog_dir = HISTORY_DIR / blog_id
    inputs: Dict[str, Any] = {
        "topic": topic.strip(),
        "mode": "",
        "needs_research": False,
        "queries": [],
        "evidence": [],
        "plan": None,
        "as_of": as_of.isoformat(),
        "recency_days": 7,
        "sections": [],
        "merged_md": "",
        "md_with_placeholders": "",
        "image_specs": [],
        "final": "",
        "blog_id": blog_id,
        "blog_dir": str(blog_dir),
        "image_manifest": [],
    }
    run_logs: List[str] = []
    status = st.status("Generating your blog...", expanded=True)
    progress_area = st.empty()
    current_state: Dict[str, Any] = {}
    try:
        for kind, payload in try_stream(app, inputs):
            if kind == "values":
                current_state = extract_latest_state(current_state, payload)
                plan_data = plan_to_dict(current_state.get("plan")) or {}
                summary = {
                    "mode": current_state.get("mode"),
                    "needs_research": current_state.get("needs_research"),
                    "evidence_count": len(current_state.get("evidence") or []),
                    "planned_sections": len(plan_data.get("tasks") or []),
                    "completed_sections": len(current_state.get("sections") or []),
                    "planned_images": len(current_state.get("image_specs") or []),
                }
                progress_area.json(summary)
                run_logs.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Graph state: "
                    f"{json.dumps(json_safe(payload), ensure_ascii=False, default=str)[:1200]}"
                )
            else:
                out = json_safe(payload)
                out["blog_id"] = blog_id
                out["blog_dir"] = str(blog_dir)
                run_logs.append(
                    f"[{datetime.now().strftime('%H:%M:%S')}] Finalized blog with "
                    f"{len(out.get('image_manifest') or [])} image asset(s)."
                )
                out["logs"] = list(run_logs)
                save_history(out, run_logs)
                st.session_state.last_out = out
                st.session_state.last_logs = run_logs
                status.update(label="✅ Blog generated and saved to history", state="complete", expanded=False)
                st.success(f"Saved permanently in: {blog_dir.relative_to(APP_DIR)}")
    except Exception as exc:
        run_logs.append(f"[ERROR] {type(exc).__name__}: {exc}")
        st.session_state.last_logs = run_logs
        status.update(label="❌ Blog generation failed", state="error", expanded=True)
        st.error(f"{type(exc).__name__}: {exc}")
        st.exception(exc)

out = st.session_state.get("last_out")
if out:
    tab_plan, tab_evidence, tab_preview, tab_images, tab_logs = st.tabs(
        ["🧩 Plan", "🔎 Evidence", "📝 Markdown Preview", "🖼️ Images", "🧾 Logs"]
    )
    with tab_plan:
        st.subheader("Generation Plan")
        show_plan(out.get("plan"))
        with st.expander("Generation settings and research routing"):
            st.write("**Topic:**", out.get("topic", ""))
            st.write("**Mode:**", out.get("mode", ""))
            st.write("**Research required:**", out.get("needs_research", False))
            st.write("**Queries:**")
            st.write(out.get("queries") or [])
    with tab_evidence:
        st.subheader("Research Evidence")
        evidence = json_safe(out.get("evidence") or [])
        if not evidence:
            mode = out.get("mode", "unknown")
            if mode == "closed_book":
                st.info("This topic was routed to closed-book mode, so web research was not requested. Try a topic requiring current information or adjust the router rules.")
            else:
                st.warning("No Tavily evidence was returned. Check the Tavily key, network access, generated queries, and the Logs tab.")
        else:
            for index, item in enumerate(evidence, start=1):
                if not isinstance(item, dict):
                    st.write(item)
                    continue
                st.markdown(f"**{index}. {item.get('title') or f'Source {index}'}**")
                if item.get("url"):
                    st.markdown(f"[Open source]({item['url']})")
                st.caption(
                    f"Published: {item.get('published_at') or 'Date not supplied'}"
                    f" · Source: {item.get('source') or 'Not supplied'}"
                )
                if item.get("snippet"):
                    st.write(item["snippet"])
                st.divider()
    with tab_preview:
        st.subheader("Markdown Preview")
        final_md = str(out.get("final") or "")
        if final_md:
            render_markdown_with_local_images(final_md)
            title = get_blog_title(out)
            st.download_button("⬇️ Download Markdown", final_md.encode("utf-8"), file_name=f"{safe_slug(title)}.md", mime="text/markdown")
            blog_dir = HISTORY_DIR / str(out.get("blog_id", ""))
            if blog_dir.exists():
                st.download_button("📦 Download complete blog bundle (Markdown + images + history data)", make_bundle(blog_dir), file_name=f"{safe_slug(title)}_bundle.zip", mime="application/zip")
        else:
            st.warning("No final Markdown content was saved.")
    with tab_images:
        st.subheader("Blog Images")
        manifest = out.get("image_manifest") or []
        if manifest:
            for image in manifest:
                st.markdown(f"**{image.get('placement', 'section').title()} image — {image.get('alt', '')}**")
                image_path = (APP_DIR / str(image.get("path", ""))).resolve()
                if image_path.exists():
                    display_local_image(image_path, image.get("caption", ""))
                st.caption(f"Provider: {image.get('provider', 'Unknown')} · File: {image.get('filename', '')}")
                if image.get("source_url"):
                    st.markdown(f"[View image source]({image['source_url']})")
                if image.get("image_errors"):
                    st.warning(f"Image provider fallback details: {image['image_errors']}")
        else:
            st.info("No image manifest was saved for this blog.")
        specs = out.get("image_specs") or []
        if specs:
            with st.expander("Image prompts and specifications"):
                st.json(specs)
    with tab_logs:
        st.subheader("Generation Logs")
        logs = out.get("logs") or st.session_state.get("last_logs") or []
        if logs:
            st.text_area("Event log", value="\n\n".join(str(line) for line in logs), height=480)
        else:
            st.info("No logs were captured for this blog.")
else:
    st.info("Enter a topic in the sidebar and click **Generate Blog**. Previous blogs are saved in the history panel.")
