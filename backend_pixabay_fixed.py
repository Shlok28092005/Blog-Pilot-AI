from __future__ import annotations

import operator
import os
import re
import json
import html
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Blog Writer (Router → (Research?) → Orchestrator → Workers → ReducerWithImages)
# Model routing: Gemini 2.5 Flash-Lite for planning/research decisions, Gemini 3.5 Flash-Lite
# for section writing, and Gemini 3.1 Flash Image for generated images.
# Image capability uses your 3-node reducer flow:
#   merge_content -> decide_images -> generate_and_place_images
# ============================================================


# -----------------------------
# 1) Schemas
# -----------------------------
class Task(BaseModel):
    id: int
    title: str
    goal: str = Field(..., description="One sentence describing what the reader should do/understand.")
    bullets: List[str] = Field(..., min_length=3, max_length=6)
    target_words: int = Field(..., description="Target words (120–550).")

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str
    blog_kind: Literal["explainer", "tutorial", "news_roundup", "comparison", "system_design"] = "explainer"
    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None  # ISO "YYYY-MM-DD" preferred
    snippet: Optional[str] = None
    source: Optional[str] = None


class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = Field(5)


class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# ---- Image planning schema (ported from your image flow) ----
class ImageSpec(BaseModel):
    placeholder: str = Field(..., description="e.g. [[IMAGE_1]]")
    filename: str = Field(..., description="Save under images/, e.g. qkv_flow.png")
    alt: str
    caption: str
    prompt: str = Field(..., description="Prompt to send to the image model.")
    size: Literal["1024x1024", "1024x1536", "1536x1024"] = "1024x1024"
    quality: Literal["low", "medium", "high"] = "medium"
    visual_type: Literal["photo", "diagram"] = "diagram"


class GlobalImagePlan(BaseModel):
    md_with_placeholders: str
    images: List[ImageSpec] = Field(default_factory=list)

class State(TypedDict):
    topic: str

    # routing / research
    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    # recency
    as_of: str
    recency_days: int

    # workers
    sections: Annotated[List[tuple[int, str]], operator.add]  # (task_id, section_md)

    # reducer/image
    merged_md: str
    md_with_placeholders: str
    image_specs: List[dict]

    final: str
    blog_id: str
    blog_dir: str
    image_manifest: List[dict]


# -----------------------------
# 2) Model configuration
# -----------------------------
# Only the section-writing workers use the requested Gemini 3.5 Flash-Lite model.
# The other text/planning stages use Gemini 2.5 Flash-Lite to avoid assigning another
# Gemini 3.x model to those stages.
# Image generation uses the requested Gemini 3.1 Flash Image model.
TEXT_WRITER_MODEL = os.getenv("BLOG_WRITER_MODEL", "gemini-3.5-flash-lite")
GENERAL_TEXT_MODEL = os.getenv("BLOG_GENERAL_MODEL", "gemini-3.5-flash-lite")
IMAGE_MODEL = os.getenv("BLOG_IMAGE_MODEL", "gemini-3.1-flash-image")

router_llm = ChatGoogleGenerativeAI(model=GENERAL_TEXT_MODEL, temperature=0)
research_llm = ChatGoogleGenerativeAI(model=GENERAL_TEXT_MODEL, temperature=0)
planner_llm = ChatGoogleGenerativeAI(model=GENERAL_TEXT_MODEL, temperature=0.2)
image_planner_llm = ChatGoogleGenerativeAI(model=GENERAL_TEXT_MODEL, temperature=0.2)
writer_llm = ChatGoogleGenerativeAI(model=TEXT_WRITER_MODEL, temperature=0.4)

# -----------------------------
# 3) Router
# -----------------------------
ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false): evergreen concepts.
- hybrid (needs_research=true): evergreen + needs up-to-date examples/tools/models.
- open_book (needs_research=true): volatile weekly/news/"latest"/pricing/policy.

If needs_research=true:
- Output 3–10 high-signal, scoped queries.
- For open_book weekly roundup, include queries reflecting last 7 days.
"""

def router_node(state: State) -> dict:
    decider = router_llm.with_structured_output(RouterDecision)
    decision = decider.invoke(
        [
            SystemMessage(content=ROUTER_SYSTEM),
            HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}"),
        ]
    )

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"

# -----------------------------
# 4) Research (Tavily)
# -----------------------------
def _tavily_search(query: str, max_results: int = 5) -> List[dict]:
    if not os.getenv("TAVILY_API_KEY"):
        return []
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults  # type: ignore
        tool = TavilySearchResults(max_results=max_results)
        results = tool.invoke({"query": query})
        out: List[dict] = []
        for r in results or []:
            out.append(
                {
                    "title": r.get("title") or "",
                    "url": r.get("url") or "",
                    "snippet": r.get("content") or r.get("snippet") or "",
                    "published_at": r.get("published_date") or r.get("published_at"),
                    "source": r.get("source"),
                }
            )
        return out
    except Exception as exc:
        raise RuntimeError(
            f"Tavily search failed for query {query!r}: {type(exc).__name__}: {exc}"
        ) from exc

def _iso_to_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return date.fromisoformat(s[:10])
    except Exception:
        return None

RESEARCH_SYSTEM = """You are a research synthesizer.

Given raw web search results, produce EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Normalize published_at to ISO YYYY-MM-DD if reliably inferable; else null (do NOT guess).
- Keep snippets short.
- Deduplicate by URL.
"""

def research_node(state: State) -> dict:
    """
    Preserve Tavily's actual results directly so evidence remains traceable
    and is not silently lost during a second LLM extraction step.
    """
    queries = (state.get("queries") or [])[:10]
    raw: List[dict] = []
    for query in queries:
        raw.extend(_tavily_search(query, max_results=6))

    dedup: dict[str, EvidenceItem] = {}
    for item in raw:
        url = str(item.get("url") or "").strip()
        if not url:
            continue
        published = item.get("published_at")
        if published:
            published = str(published)[:10]
        dedup[url] = EvidenceItem(
            title=str(item.get("title") or "Untitled source"),
            url=url,
            snippet=str(item.get("snippet") or "")[:1200],
            published_at=published,
            source=str(item.get("source") or "") or None,
        )

    evidence = list(dedup.values())
    # Keep undated results visible; only discard results with a known old date.
    if state.get("mode") == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=int(state["recency_days"]))
        evidence = [
            item for item in evidence
            if not item.published_at
            or (parsed := _iso_to_date(item.published_at)) is None
            or parsed >= cutoff
        ]
    return {"evidence": evidence}


# -----------------------------
# 5) Orchestrator (Plan)
# -----------------------------
ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Produce a highly actionable outline for a technical blog post.

Requirements:
- 5–9 tasks, each with goal + 3–6 bullets + target_words.
- Tags are flexible; do not force a fixed taxonomy.

Grounding:
- closed_book: evergreen, no evidence dependence.
- hybrid: use evidence for up-to-date examples; mark those tasks requires_research=True and requires_citations=True.
- open_book: weekly/news roundup:
  - Set blog_kind="news_roundup"
  - No tutorial content unless requested
  - If evidence is weak, plan should explicitly reflect that (don’t invent events).

Output must match Plan schema.
"""

def orchestrator_node(state: State) -> dict:
    planner = planner_llm.with_structured_output(Plan)
    mode = state.get("mode", "closed_book")
    evidence = state.get("evidence", [])

    forced_kind = "news_roundup" if mode == "open_book" else None

    plan = planner.invoke(
        [
            SystemMessage(content=ORCH_SYSTEM),
            HumanMessage(
                content=(
                    f"Topic: {state['topic']}\n"
                    f"Mode: {mode}\n"
                    f"As-of: {state['as_of']} (recency_days={state['recency_days']})\n"
                    f"{'Force blog_kind=news_roundup' if forced_kind else ''}\n\n"
                    f"Evidence:\n{[e.model_dump() for e in evidence][:16]}"
                )
            ),
        ]
    )
    if forced_kind:
        plan.blog_kind = "news_roundup"

    return {"plan": plan}


# -----------------------------
# 6) Fanout
# -----------------------------
def fanout(state: State):
    assert state["plan"] is not None
    return [
        Send(
            "worker",
            {
                "task": task.model_dump(),
                "topic": state["topic"],
                "mode": state["mode"],
                "as_of": state["as_of"],
                "recency_days": state["recency_days"],
                "plan": state["plan"].model_dump(),
                "evidence": [e.model_dump() for e in state.get("evidence", [])],
            },
        )
        for task in state["plan"].tasks
    ]

# -----------------------------
# 7) Worker
# -----------------------------
WORKER_SYSTEM = """You are a senior technical writer and developer advocate.
Write ONE section of a technical blog post in Markdown.

Constraints:
- Cover ALL bullets in order.
- Target words ±15%.
- Output only section markdown starting with "## <Section Title>".

Scope guard:
- If blog_kind=="news_roundup", do NOT drift into tutorials (scraping/RSS/how to fetch).
  Focus on events + implications.

Grounding:
- If mode=="open_book": do not introduce any specific event/company/model/funding/policy claim unless supported by provided Evidence URLs.
  For each supported claim, attach a Markdown link ([Source](URL)).
  If unsupported, write "Not found in provided sources."
- If requires_citations==true (hybrid tasks): cite Evidence URLs for external claims.

Code:
- If requires_code==true, include at least one minimal snippet.
"""

def _content_to_text(content: object) -> str:
    """Normalize common LangChain/Gemini response content formats to plain text."""
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, dict):
        # Gemini/LangChain may return text blocks as dictionaries.
        text_value = content.get("text")
        if isinstance(text_value, str):
            return text_value
        return str(content)

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                # Ignore non-text multimodal blocks instead of serializing them.
            else:
                text_value = getattr(item, "text", None)
                if isinstance(text_value, str):
                    parts.append(text_value)
        return "\n".join(part for part in parts if part)

    return str(content)


def worker_node(state: dict) -> dict:
    """Generate one planned section and return it in the reducer's expected format."""
    task_data = state.get("task")
    if not task_data:
        raise ValueError("Worker received no 'task'. Check the fanout() Send payload.")

    task = Task.model_validate(task_data)
    plan_data = state.get("plan")
    plan = plan_data if isinstance(plan_data, Plan) else Plan.model_validate(plan_data)

    topic = str(state.get("topic", "")).strip()
    mode = str(state.get("mode", "closed_book"))
    evidence = state.get("evidence") or []
    evidence_text = "\n".join(
        f"- {item.get('title', 'Source')}: {item.get('url', '')} — {item.get('snippet', '')}"
        for item in evidence
        if isinstance(item, dict) and item.get("url")
    ) or "No research evidence was provided."

    task_brief = (
        f"Goal: {task.goal}\n"
        f"Cover these bullets in order:\n"
        + "\n".join(f"- {bullet}" for bullet in task.bullets)
        + f"\nTarget length: approximately {task.target_words} words.\n"
        f"Requires research: {task.requires_research}.\n"
        f"Requires citations: {task.requires_citations}.\n"
        f"Requires code: {task.requires_code}.\n"
        f"Tags: {', '.join(task.tags) if task.tags else 'None'}."
    )

    response = writer_llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=(
            f"Blog Title: {plan.blog_title}\n"
            f"Main Topic: {topic}\n"
            f"Blog kind: {plan.blog_kind}\n"
            f"Mode: {mode}\n"
            f"As-of date: {state.get('as_of', '')}\n\n"
            f"Section Title: {task.title}\n"
            f"Section Brief:\n{task_brief}\n\n"
            f"Available evidence and source URLs:\n{evidence_text}\n\n"
            "Return only the section content in Markdown. Start with a level-two heading "
            f"such as '## {task.title}'."
        ))
    ])

    section_md = _content_to_text(getattr(response, "content", response)).strip()
    if not section_md:
        raise ValueError(f"The model returned an empty section for task {task.id}: {task.title}")

    # State.sections is a list of (task_id, markdown) tuples combined with operator.add.
    return {"sections": [(task.id, section_md)]}

    
# ============================================================
# 8) ReducerWithImages (subgraph)
#    merge_content -> decide_images -> generate_and_place_images
# ============================================================
def merge_content(state: State) -> dict:
    plan = state["plan"]
    if plan is None:
        raise ValueError("merge_content called without plan.")
    ordered_sections = [md for _, md in sorted(state["sections"], key=lambda x: x[0])]
    body = "\n\n".join(ordered_sections).strip()
    merged_md = f"# {plan.blog_title}\n\n{body}\n"
    return {"merged_md": merged_md}


DECIDE_IMAGES_SYSTEM = """You are an expert technical editor.
Decide if images/diagrams are needed for THIS blog.

Rules:
- Create up to 3 useful in-article images (not including the separate cover image).
- Each image must materially improve understanding (diagram, architecture, workflow, or explanatory visual).
- Set visual_type="diagram" for architecture/workflow/process visuals and visual_type="photo" for real-world editorial photography.
- Insert placeholders exactly: [[IMAGE_1]], [[IMAGE_2]], [[IMAGE_3]].
- If no in-article images are useful: md_with_placeholders must equal input and images=[].
- Avoid decorative images; prefer technical diagrams with short labels.
Return strictly GlobalImagePlan.
"""

def decide_images(state: State) -> dict:
    planner = image_planner_llm.with_structured_output(GlobalImagePlan)
    merged_md = state["merged_md"]
    plan = state["plan"]
    assert plan is not None

    image_plan = planner.invoke(
        [
            SystemMessage(content=DECIDE_IMAGES_SYSTEM),
            HumanMessage(
                content=(
                    f"Blog kind: {plan.blog_kind}\n"
                    f"Topic: {state['topic']}\n\n"
                    "Insert placeholders + propose image prompts.\n\n"
                    f"{merged_md}"
                )
            ),
        ]
    )

    return {
        "md_with_placeholders": image_plan.md_with_placeholders,
        "image_specs": [img.model_dump() for img in image_plan.images],
    }


def _gemini_generate_image_bytes(prompt: str) -> bytes:
    """Generate an image with the configured Gemini image model."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set.")

    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(
        model=IMAGE_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(response_modalities=["IMAGE"]),
    )
    parts = getattr(resp, "parts", None)
    if not parts and getattr(resp, "candidates", None):
        try:
            parts = resp.candidates[0].content.parts
        except Exception:
            parts = None
    for part in parts or []:
        inline = getattr(part, "inline_data", None)
        if inline and getattr(inline, "data", None):
            data = inline.data
            return data if isinstance(data, bytes) else bytes(data)
    raise RuntimeError("Gemini returned no inline image data.")


def _safe_slug(title: str) -> str:
    s = title.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]+", "", s)
    s = re.sub(r"\s+", "_", s).strip("_")
    return s[:70] or "blog"


def _pixabay_image(prompt: str, output_path: Path) -> dict:
    """Search Pixabay and download a relevant stock photo."""
    import requests

    api_key = os.getenv("PIXABAY_API_KEY")
    if not api_key:
        raise RuntimeError("PIXABAY_API_KEY is not configured in .env.")

    query = re.sub(r"[^a-zA-Z0-9 ]", " ", prompt)
    query = " ".join(query.split()[:8]) or "technology"
    response = requests.get(
        "https://pixabay.com/api/",
        params={
            "key": api_key,
            "q": query,
            "image_type": "photo",
            "orientation": "horizontal",
            "safesearch": "true",
            "per_page": 5,
        },
        timeout=25,
    )
    response.raise_for_status()
    hits = response.json().get("hits", [])
    if not hits:
        raise RuntimeError(f"Pixabay returned no photos for search: {query!r}")

    photo = hits[0]
    image_url = photo.get("largeImageURL") or photo.get("webformatURL")
    if not image_url:
        raise RuntimeError("Pixabay result did not contain an image URL.")

    image_response = requests.get(image_url, timeout=30)
    image_response.raise_for_status()
    content_type = image_response.headers.get("Content-Type", "").lower()
    extension = ".jpg" if "jpeg" in content_type or "jpg" in content_type else ".png"
    output_path = output_path.with_suffix(extension)
    output_path.write_bytes(image_response.content)
    return {
        "provider": "Pixabay",
        "filename": output_path.name,
        "source_url": photo.get("pageURL"),
        "image_url": image_url,
        "photographer": photo.get("user"),
    }


def _make_fallback_svg(title: str, prompt: str, output_path: Path) -> None:
    """Create a lightweight, local SVG concept diagram when both image APIs fail."""
    clean = re.sub(r"\s+", " ", prompt).strip()
    words = re.findall(r"[A-Za-z][A-Za-z0-9+-]*", clean)
    labels = []
    for i in range(3):
        chunk = words[i * 5:(i + 1) * 5]
        labels.append(" ".join(chunk[:4]) or ["Input", "Process", "Output"][i])
    title = html.escape((title or "Concept overview")[:72])
    boxes = []
    for i, label in enumerate(labels):
        x = 55 + i * 280
        boxes.append(f'<rect x="{x}" y="170" width="220" height="105" rx="16" fill="#eef4ff" stroke="#4263a5" stroke-width="2"/>')
        boxes.append(f'<text x="{x+110}" y="{215}" text-anchor="middle" font-size="17" font-family="Arial" fill="#172b4d">{html.escape(label[:34])}</text>')
        if i < 2:
            boxes.append(f'<path d="M{x+225} 222 H{x+270} l-10 -8 m10 8 l-10 8" fill="none" stroke="#4263a5" stroke-width="3"/>')
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="900" height="360" viewBox="0 0 900 360">
<rect width="900" height="360" fill="#ffffff"/><text x="450" y="75" text-anchor="middle" font-size="26" font-family="Arial" font-weight="bold" fill="#172b4d">{title}</text>
{''.join(boxes)}<text x="450" y="325" text-anchor="middle" font-size="13" font-family="Arial" fill="#52627a">Concept diagram generated as an image fallback</text></svg>"""
    output_path.write_text(svg, encoding="utf-8")


def generate_and_place_images(state: State) -> dict:
    plan = state["plan"]
    assert plan is not None
    blog_id = str(state.get("blog_id") or uuid.uuid4().hex[:12])
    blog_dir = Path(state.get("blog_dir") or Path("blog_history") / blog_id)
    images_dir = blog_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    md = state.get("md_with_placeholders") or state["merged_md"]
    image_specs = [dict(x) for x in (state.get("image_specs", []) or [])]

    # Always plan a distinct cover image as well as any useful section visuals.
    cover_spec = {
        "placeholder": "[[BLOG_COVER_IMAGE]]",
        "filename": "cover.png",
        "alt": f"Cover image illustrating {plan.blog_title}",
        "caption": f"Overview of {plan.blog_title}",
        "prompt": f"Professional editorial hero photograph illustrating {plan.blog_title}. Topic: {state['topic']}. Wide landscape composition, no text, clean composition suitable for a technical blog cover.",
        "size": "1536x1024",
        "quality": "medium",
        "visual_type": "photo",
        "placement": "cover",
    }
    # Put cover placeholder directly below the H1, preserving any opening prose.
    if "[[BLOG_COVER_IMAGE]]" not in md:
        first_newline = md.find("\n")
        if first_newline >= 0:
            md = md[:first_newline + 1] + "\n[[BLOG_COVER_IMAGE]]\n\n" + md[first_newline + 1:]
        else:
            md = md + "\n\n[[BLOG_COVER_IMAGE]]\n"

    all_specs = [cover_spec] + image_specs
    manifest = []
    for index, spec in enumerate(all_specs, start=1):
        placeholder = str(spec.get("placeholder") or f"[[IMAGE_{index}]]")
        filename = Path(str(spec.get("filename") or f"image_{index}.jpg")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".svg"}:
            filename = f"image_{index}.jpg"
        out_path = images_dir / filename
        # Avoid overwriting files if model supplied duplicate names.
        if any(item["filename"] == filename for item in manifest):
            out_path = images_dir / f"{index}_{filename}"
            filename = out_path.name

        metadata = {"provider": None}
        if not out_path.exists():
            visual_type = str(spec.get("visual_type", "diagram")).lower()
            image_errors = []

            # Pixabay supplies stock photos; Gemini generates diagrams and can
            # serve as a fallback if Pixabay has no suitable photo.
            if visual_type == "photo":
                try:
                    metadata = _pixabay_image(str(spec.get("prompt", "")), out_path)
                    out_path = images_dir / metadata["filename"]
                    filename = out_path.name
                except Exception as exc:
                    image_errors.append(f"Pixabay: {type(exc).__name__}: {exc}")

            if metadata.get("provider") is None:
                try:
                    image_bytes = _gemini_generate_image_bytes(str(spec.get("prompt", "")))
                    out_path.write_bytes(image_bytes)
                    metadata = {"provider": "Gemini"}
                except Exception as exc:
                    image_errors.append(f"Gemini: {type(exc).__name__}: {exc}")
                    diagram_title = str(spec.get("caption") or spec.get("alt") or plan.blog_title)
                    svg_name = str(Path(filename).with_suffix(".svg"))
                    svg_path = images_dir / svg_name
                    _make_fallback_svg(diagram_title, str(spec.get("prompt", "")), svg_path)
                    out_path = svg_path
                    filename = svg_name
                    metadata = {
                        "provider": "SVG diagram fallback",
                        "image_errors": " | ".join(image_errors)[:1000],
                    }

        try:
            rel_path = out_path.resolve().relative_to(Path(__file__).resolve().parent).as_posix()
        except ValueError:
            rel_path = out_path.as_posix()
        alt = str(spec.get("alt") or "Blog illustration")
        caption = str(spec.get("caption") or "")
        attribution = ""
        if metadata.get("provider") == "Pexels":
            photographer = metadata.get("photographer") or "Pexels contributor"
            source_url = metadata.get("source_url")
            attribution = f" Photo by [{photographer}]({source_url}) on Pexels." if source_url else f" Photo by {photographer} on Pexels."
        image_md = f"![{alt}]({rel_path})\n*{caption}{attribution}*"
        md = md.replace(placeholder, image_md)
        manifest.append({
            "placeholder": placeholder,
            "filename": filename,
            "path": rel_path,
            "alt": alt,
            "caption": caption,
            "placement": spec.get("placement", "cover" if index == 1 else "section"),
            **metadata,
        })

    # Replace any unresolved model placeholders with a clean, non-blocking note.
    md = re.sub(r"\[\[IMAGE_[0-9]+\]\]", "", md)
    md = md.replace("[[BLOG_COVER_IMAGE]]", "")
    md_path = blog_dir / "blog.md"
    md_path.write_text(md, encoding="utf-8")
    return {
        "final": md,
        "blog_id": blog_id,
        "blog_dir": str(blog_dir),
        "image_specs": all_specs,
        "image_manifest": manifest,
    }

# build reducer subgraph
reducer_graph = StateGraph(State)
reducer_graph.add_node("merge_content", merge_content)
reducer_graph.add_node("decide_images", decide_images)
reducer_graph.add_node("generate_and_place_images", generate_and_place_images)
reducer_graph.add_edge(START, "merge_content")
reducer_graph.add_edge("merge_content", "decide_images")
reducer_graph.add_edge("decide_images", "generate_and_place_images")
reducer_graph.add_edge("generate_and_place_images", END)
reducer_subgraph = reducer_graph.compile()

# -----------------------------
# 9) Build main graph
# -----------------------------
g = StateGraph(State)
g.add_node("router", router_node)
g.add_node("research", research_node)
g.add_node("orchestrator", orchestrator_node)
g.add_node("worker", worker_node)
g.add_node("reducer", reducer_subgraph)

g.add_edge(START, "router")
g.add_conditional_edges("router", route_next, {"research": "research", "orchestrator": "orchestrator"})
g.add_edge("research", "orchestrator")

g.add_conditional_edges("orchestrator", fanout, ["worker"])
g.add_edge("worker", "reducer")
g.add_edge("reducer", END)

app = g.compile()
app