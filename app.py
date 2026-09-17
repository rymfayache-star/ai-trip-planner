import asyncio
import html
import json
import os
import sys
from pathlib import Path

import chromadb
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from openai import OpenAI

load_dotenv()  # loads .env for local development (file is gitignored)

BASE_DIR = Path(__file__).resolve().parent
TIPS_FILE = BASE_DIR / "TIPS.txt"
CHROMA_PATH = BASE_DIR / "chroma"
WEATHER_SERVER = BASE_DIR / "weather_server.py"
COLLECTION_NAME = "travel_tips"
EMBEDDING_MODEL = "text-embedding-3-small"

st.set_page_config(
    page_title="AI Trip Planner",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&family=Outfit:wght@400;500;600&display=swap');

    :root {
        --ocean: #0b4f6c;
        --teal: #148a8a;
        --foam: #e8f4f6;
        --sand: #f3efe6;
        --ink: #1a2b33;
        --muted: #5a6d76;
        --card: rgba(255, 255, 255, 0.88);
        --line: rgba(11, 79, 108, 0.12);
    }

    .stApp {
        background:
            radial-gradient(1200px 500px at 10% -10%, #cfe9ef 0%, transparent 55%),
            radial-gradient(900px 420px at 95% 0%, #d9efe8 0%, transparent 50%),
            linear-gradient(165deg, #eaf6f8 0%, #f5f1e8 48%, #eef7f4 100%);
        color: var(--ink);
        font-family: "Outfit", sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #eaf6f8 0%, #f3efe6 55%, #eef7f4 100%);
        border-right: 1px solid rgba(11, 79, 108, 0.12);
    }

    [data-testid="stSidebar"],
    [data-testid="stSidebar"] *,
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] [data-testid="stCaption"],
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] p {
        color: #12262f !important;
        -webkit-text-fill-color: #12262f !important;
    }

    [data-testid="stSidebar"] label {
        font-weight: 500 !important;
        letter-spacing: 0.02em;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(11, 79, 108, 0.18) !important;
    }

    /* Inputs + select: dark text on light background */
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stNumberInput input,
    [data-testid="stSidebar"] .stNumberInput input[type="number"],
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] span,
    [data-testid="stSidebar"] [data-baseweb="select"] div,
    [data-testid="stSidebar"] [data-baseweb="select"] div[aria-selected],
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-baseweb="select"] *,
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] textarea {
        background: #ffffff !important;
        border: 1px solid rgba(11, 79, 108, 0.22) !important;
        color: #12262f !important;
        -webkit-text-fill-color: #12262f !important;
        border-radius: 10px !important;
        caret-color: #12262f !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] svg {
        color: #12262f !important;
        fill: #12262f !important;
    }

    div[data-baseweb="popover"] li,
    div[data-baseweb="popover"] li *,
    ul[role="listbox"] li,
    ul[role="listbox"] li *,
    div[role="listbox"] *,
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] li * {
        color: #12262f !important;
        -webkit-text-fill-color: #12262f !important;
    }

    [data-testid="stSidebar"] .stTextInput input::placeholder,
    [data-testid="stSidebar"] input::placeholder {
        color: rgba(18, 38, 47, 0.55) !important;
        -webkit-text-fill-color: rgba(18, 38, 47, 0.55) !important;
        opacity: 1 !important;
    }

    /* Buttons: dark text */
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stButton > button:hover,
    [data-testid="stSidebar"] .stButton > button:focus,
    [data-testid="stSidebar"] .stButton > button:active,
    [data-testid="stSidebar"] .stButton > button[kind="primary"],
    [data-testid="stSidebar"] .stButton > button[kind="secondary"],
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div,
    [data-testid="stSidebar"] .stButton > button * {
        width: 100%;
        background: #ffffff !important;
        color: #12262f !important;
        -webkit-text-fill-color: #12262f !important;
        border: 1px solid rgba(11, 79, 108, 0.2) !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        padding: 0.65rem 1rem !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 20px rgba(26, 43, 51, 0.12);
    }

    /* Main-area download / action buttons: dark text too */
    .stDownloadButton > button,
    .stDownloadButton > button *,
    div[data-testid="stMain"] .stButton > button,
    div[data-testid="stMain"] .stButton > button * {
        color: #12262f !important;
        -webkit-text-fill-color: #12262f !important;
        background: #ffffff !important;
        border: 1px solid rgba(11, 79, 108, 0.18) !important;
    }

    .hero-banner {
        background:
            linear-gradient(120deg, rgba(11, 79, 108, 0.92), rgba(20, 138, 138, 0.78)),
            url("data:image/svg+xml,%3Csvg width='160' height='160' viewBox='0 0 160 160' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-opacity='0.12' stroke='%23ffffff' stroke-width='1'%3E%3Cpath d='M0 80c20-20 40-20 60 0s40 20 60 0 40-20 60 0'/%3E%3Cpath d='M0 110c20-20 40-20 60 0s40 20 60 0 40-20 60 0'/%3E%3C/g%3E%3C/svg%3E");
        background-size: cover, 160px 160px;
        border-radius: 22px;
        padding: 2.4rem 2.2rem;
        margin-bottom: 1.6rem;
        color: #fff;
        box-shadow: 0 18px 40px rgba(11, 79, 108, 0.18);
        animation: rise-in 0.55s ease-out;
    }

    .hero-banner h1 {
        font-family: "Fraunces", serif;
        font-size: clamp(2.1rem, 3.4vw, 3rem);
        font-weight: 700;
        margin: 0 0 0.45rem 0;
        letter-spacing: -0.02em;
        line-height: 1.1;
    }

    .hero-banner p {
        margin: 0;
        max-width: 38rem;
        font-size: 1.05rem;
        opacity: 0.92;
        line-height: 1.5;
    }

    .itinerary-card {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 2.4rem 2rem;
        min-height: 280px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        backdrop-filter: blur(8px);
        box-shadow: 0 10px 28px rgba(26, 43, 51, 0.06);
        animation: rise-in 0.7s ease-out;
    }

    .itinerary-card h2 {
        font-family: "Fraunces", serif;
        color: var(--ocean);
        font-size: 1.55rem;
        margin: 0 0 0.5rem 0;
    }

    .itinerary-card p {
        margin: 0;
        color: var(--muted);
        font-size: 1rem;
    }

    .meta-chip {
        display: inline-block;
        margin: 0 0.35rem 1rem 0;
        padding: 0.35rem 0.75rem;
        border-radius: 999px;
        background: rgba(20, 138, 138, 0.12);
        color: var(--ocean);
        font-size: 0.88rem;
        font-weight: 500;
    }

    .weather-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        margin: 0 0 1rem 0;
        padding: 0.45rem 0.9rem;
        border-radius: 999px;
        background: linear-gradient(120deg, rgba(11, 79, 108, 0.12), rgba(20, 138, 138, 0.16));
        border: 1px solid var(--line);
        color: var(--ocean);
        font-size: 0.92rem;
        font-weight: 500;
        max-width: 100%;
    }

    .weather-badge .label {
        font-weight: 600;
        letter-spacing: 0.02em;
        text-transform: uppercase;
        font-size: 0.72rem;
        opacity: 0.75;
    }

    div[data-testid="stExpander"] {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 16px;
        margin-bottom: 0.85rem;
        box-shadow: 0 8px 22px rgba(26, 43, 51, 0.05);
    }

    .day-slot {
        margin: 0.65rem 0;
        padding: 0.75rem 0.9rem;
        border-radius: 12px;
        background: rgba(11, 79, 108, 0.05);
        border: 1px solid var(--line);
    }

    .day-slot strong {
        color: var(--ocean);
        display: block;
        margin-bottom: 0.25rem;
    }

    .day-slot p {
        margin: 0;
        color: var(--ink);
        line-height: 1.45;
    }

    .tips-box {
        background: var(--card);
        border: 1px solid var(--line);
        border-left: 4px solid var(--teal);
        border-radius: 16px;
        padding: 1.35rem 1.5rem;
        margin-top: 1.25rem;
        box-shadow: 0 8px 22px rgba(26, 43, 51, 0.05);
        animation: rise-in 0.7s ease-out;
    }

    .tips-box h3 {
        font-family: "Fraunces", serif;
        color: var(--ocean);
        font-size: 1.25rem;
        margin: 0 0 0.75rem 0;
    }

    .tips-box ul {
        margin: 0;
        padding-left: 1.2rem;
        color: var(--ink);
    }

    .tips-box li {
        margin: 0.35rem 0;
        line-height: 1.45;
    }

    .summary-bar {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.75rem;
        margin: 0 0 1rem 0;
        padding: 0.9rem 1.1rem;
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 8px 22px rgba(26, 43, 51, 0.05);
    }

    .summary-bar .summary-text {
        color: var(--ocean);
        font-weight: 600;
        font-size: 1rem;
        margin: 0;
    }

    .summary-bar .summary-sub {
        display: block;
        color: var(--muted);
        font-weight: 400;
        font-size: 0.88rem;
        margin-top: 0.15rem;
    }

    .empty-state {
        background: var(--card);
        border: 1px dashed rgba(11, 79, 108, 0.28);
        border-radius: 18px;
        padding: 2.6rem 2rem;
        min-height: 280px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        animation: rise-in 0.7s ease-out;
    }

    .empty-state h2 {
        font-family: "Fraunces", serif;
        color: var(--ocean);
        font-size: 1.55rem;
        margin: 0 0 0.55rem 0;
    }

    .empty-state p {
        margin: 0 auto;
        max-width: 28rem;
        color: var(--muted);
        font-size: 1rem;
        line-height: 1.5;
    }

    [data-testid="stSidebar"] .past-trip-label {
        font-size: 0.78rem;
        opacity: 0.85;
        margin: 0.35rem 0 0.15rem 0;
    }

    div[data-testid="stToolbar"],
    #MainMenu,
    footer {
        visibility: hidden;
    }

    @keyframes rise-in {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


class MissingAPIKeyError(Exception):
    """Raised when OPENAI_API_KEY is not configured."""


def get_openai_api_key() -> str | None:
    """Prefer env / .env locally; fall back to Streamlit secrets (Cloud)."""
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        return None


def require_openai_api_key() -> str:
    api_key = get_openai_api_key()
    if not api_key:
        raise MissingAPIKeyError(
            "OpenAI API key is missing. Add OPENAI_API_KEY to your .env file "
            "(or Streamlit secrets), then restart the app."
        )
    return api_key


def get_database_url() -> str | None:
    """Read the Postgres connection string from the environment."""
    return os.getenv("DATABASE_URL")


def run_query(
    sql: str,
    params: tuple | list | None = None,
    *,
    fetch: bool = False,
    fetch_all: bool = False,
):
    """Open a PostgreSQL connection, run a parameterised query, then commit."""
    database_url = get_database_url()
    if not database_url:
        raise ValueError(
            "DATABASE_URL is not set. Configure it in your environment and restart the app."
        )

    conn = psycopg2.connect(database_url)
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            if fetch_all:
                result = cur.fetchall()
            elif fetch:
                result = cur.fetchone()
            else:
                result = None
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _parse_itinerary(value) -> list[dict]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        data = json.loads(value)
        return data if isinstance(data, list) else data.get("days", [])
    return []


def list_trips() -> list[dict]:
    """SELECT all trips ordered by newest first."""
    rows = run_query(
        """
        SELECT id, destination, days, style, itinerary, created_at
        FROM trips
        ORDER BY created_at DESC
        """,
        fetch_all=True,
    )
    trips = []
    for row in rows or []:
        trips.append(
            {
                "id": int(row[0]),
                "destination": row[1],
                "days": int(row[2]),
                "style": row[3],
                "itinerary": _parse_itinerary(row[4]),
                "created_at": row[5],
            }
        )
    return trips


def get_trip_by_id(trip_id: int) -> dict | None:
    """Load one saved trip by id."""
    row = run_query(
        """
        SELECT id, destination, days, style, itinerary, created_at
        FROM trips
        WHERE id = %s
        """,
        (trip_id,),
        fetch=True,
    )
    if not row:
        return None
    return {
        "id": int(row[0]),
        "destination": row[1],
        "days": int(row[2]),
        "style": row[3],
        "itinerary": _parse_itinerary(row[4]),
        "created_at": row[5],
    }


def delete_trip(trip_id: int) -> None:
    """DELETE a trip row by id."""
    run_query("DELETE FROM trips WHERE id = %s", (trip_id,))


def save_trip(
    destination: str,
    days: int,
    style: str,
    itinerary: list[dict],
) -> int:
    """Insert a generated trip into the trips table and return its id."""
    row = run_query(
        """
        INSERT INTO trips (destination, days, style, itinerary)
        VALUES (%s, %s, %s, %s)
        RETURNING id
        """,
        (destination, days, style, json.dumps(itinerary)),
        fetch=True,
    )
    if not row:
        raise RuntimeError("Insert succeeded but no trip id was returned.")
    return int(row[0])


def apply_saved_trip(trip: dict) -> None:
    """Show a saved trip in the main area without calling OpenAI."""
    st.session_state.itinerary = trip["itinerary"]
    st.session_state.trip_meta = {
        "destination": trip["destination"],
        "days": trip["days"],
        "travel_style": trip["style"],
    }
    st.session_state.retrieved_tips = None
    st.session_state.weather_summary = None
    st.session_state.loaded_trip_id = trip["id"]
    st.session_state.db_save_message = None
    st.session_state.db_save_error = None


def _load_tip_lines() -> list[str]:
    if not TIPS_FILE.exists():
        raise FileNotFoundError(f"Tips file not found: {TIPS_FILE}")
    return [
        line.strip()
        for line in TIPS_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _embed_texts(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


@st.cache_resource
def init_tips_collection():
    """Load TIPS.txt into a persistent Chroma collection once."""
    api_key = require_openai_api_key()

    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)

    if collection.count() == 0:
        tips = _load_tip_lines()
        if not tips:
            raise ValueError("TIPS.txt is empty — nothing to embed.")

        openai_client = OpenAI(api_key=api_key)
        embeddings = _embed_texts(openai_client, tips)
        collection.add(
            ids=[f"tip-{i}" for i in range(len(tips))],
            documents=tips,
            embeddings=embeddings,
        )

    return collection


@st.cache_data(show_spinner=False, ttl=3600)
def retrieve_tips(preference: str, n_results: int = 3) -> list[str]:
    """Return the n most relevant local tips for a travel preference."""
    if not preference.strip():
        return []

    api_key = require_openai_api_key()
    collection = init_tips_collection()
    openai_client = OpenAI(api_key=api_key)
    query_embedding = _embed_texts(openai_client, [preference.strip()])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(n_results, max(collection.count(), 1)),
    )
    documents = results.get("documents") or [[]]
    return [doc for doc in documents[0] if doc]


async def _fetch_weather_mcp(city: str) -> str:
    """Call get_weather on weather_server.py over MCP stdio transport."""
    if not WEATHER_SERVER.exists():
        raise FileNotFoundError(f"Weather MCP server not found: {WEATHER_SERVER}")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(WEATHER_SERVER)],
        cwd=str(BASE_DIR),
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("get_weather", {"city": city})
            if getattr(result, "is_error", False):
                raise RuntimeError(f"get_weather failed: {result}")
            texts = [
                block.text
                for block in (result.content or [])
                if getattr(block, "text", None)
            ]
            if not texts:
                raise RuntimeError("get_weather returned no text content.")
            return "\n".join(texts).strip()


@st.cache_data(show_spinner=False, ttl=1800)
def get_weather_for_destination(city: str) -> str:
    """Cached weather lookup via MCP (fast on re-runs for the same city)."""
    return asyncio.run(_fetch_weather_mcp(city))


@st.cache_data(show_spinner=False, ttl=3600)
def generate_itinerary(
    destination: str,
    days: int,
    travel_style: str,
    tips: tuple[str, ...] = (),
    weather: str | None = None,
) -> list[dict]:
    """Cached OpenAI itinerary generation for identical trip inputs."""
    api_key = require_openai_api_key()

    tips_block = ""
    if tips:
        tips_list = "; ".join(tips)
        tips_block = f"\nUse these local tips where relevant: {tips_list}."

    weather_block = ""
    if weather:
        weather_block = (
            f"\nCurrent weather forecast: {weather}. "
            "Adapt the itinerary to suit these conditions "
            "(prefer indoor activities if rain or storms; outdoor plans if clear and mild)."
        )

    client = OpenAI(api_key=api_key)
    system_prompt = (
        "You are an expert travel planner. Return ONLY valid JSON with this shape:\n"
        '{"days":[{"day_number":1,"theme":"...","morning":"...","afternoon":"...",'
        '"evening":"...","food_tip":"..."}]}\n'
        "Include exactly one object per day. Keep each field concise and practical. "
        "When weather is provided, adjust activities accordingly."
    )
    user_prompt = (
        f"Create a {days}-day itinerary for {destination}. "
        f"Travel style: {travel_style}.{tips_block}{weather_block}"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    data = json.loads(content)
    days_data = data.get("days", data if isinstance(data, list) else [])
    if not isinstance(days_data, list) or not days_data:
        raise ValueError("The model did not return a usable day-by-day itinerary.")
    return days_data


def itinerary_to_text(
    itinerary: list[dict],
    meta: dict,
    weather: str | None = None,
    tips: list[str] | None = None,
) -> str:
    lines = [
        "AI Trip Planner — Itinerary",
        "=" * 40,
        f"Destination: {meta.get('destination', '')}",
        f"Total days: {meta.get('days', len(itinerary))}",
        f"Travel style: {meta.get('travel_style', '')}",
    ]
    if weather:
        lines.append(f"Weather: {weather}")
    lines.append("")

    for day in itinerary:
        lines.extend(
            [
                f"Day {day.get('day_number', '?')}: {day.get('theme', 'Explore')}",
                f"  Morning: {day.get('morning', '')}",
                f"  Afternoon: {day.get('afternoon', '')}",
                f"  Evening: {day.get('evening', '')}",
                f"  Food tip: {day.get('food_tip', '')}",
                "",
            ]
        )

    if tips:
        lines.append("Local insider tips")
        lines.append("-" * 20)
        lines.extend(f"- {tip}" for tip in tips)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def render_weather_badge(weather: str | None) -> None:
    if not weather:
        return
    safe = html.escape(weather)
    st.markdown(
        f"""
        <div class="weather-badge">
            <span class="label">Weather</span>
            <span>{safe}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_day_card(day: dict) -> None:
    day_number = day.get("day_number", "?")
    theme = day.get("theme", "Explore")
    with st.expander(f"Day {day_number}: {theme}", expanded=False):
        slots = [
            ("🌅", "Morning", day.get("morning", "")),
            ("☀️", "Afternoon", day.get("afternoon", "")),
            ("🌙", "Evening", day.get("evening", "")),
            ("🍽️", "Food tip", day.get("food_tip", "")),
        ]
        for icon, label, text in slots:
            st.markdown(
                f"""
                <div class="day-slot">
                    <strong>{icon} {label}</strong>
                    <p>{html.escape(str(text))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_tips_box(tips: list[str]) -> None:
    if not tips:
        return
    items = "".join(f"<li>{html.escape(tip)}</li>" for tip in tips)
    st.markdown(
        f"""
        <div class="tips-box">
            <h3>Local insider tips</h3>
            <ul>{items}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="empty-state">
            <div>
                <h2>Your next adventure starts here</h2>
                <p>
                    Pick a destination, choose how many days you have, and set your travel style.
                    Then hit <strong>Plan my trip</strong> — we’ll craft a day-by-day plan with
                    local tips and today’s weather in mind.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_summary_and_download(
    itinerary: list[dict],
    meta: dict,
    weather: str | None,
    tips: list[str] | None,
) -> None:
    total_days = meta.get("days") or len(itinerary)
    dest = html.escape(str(meta.get("destination", "")))
    style = html.escape(str(meta.get("travel_style", "")))
    day_label = "day" if int(total_days) == 1 else "days"

    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"""
            <div class="summary-bar">
                <p class="summary-text">
                    {total_days} {day_label} in {dest}
                    <span class="summary-sub">{style} · {len(itinerary)} planned day cards</span>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        filename = f"itinerary-{meta.get('destination', 'trip').replace(' ', '_').lower()}.txt"
        st.download_button(
            label="Download itinerary",
            data=itinerary_to_text(itinerary, meta, weather, tips),
            file_name=filename,
            mime="text/plain",
            use_container_width=True,
        )


if "itinerary" not in st.session_state:
    st.session_state.itinerary = None
if "trip_meta" not in st.session_state:
    st.session_state.trip_meta = None
if "retrieved_tips" not in st.session_state:
    st.session_state.retrieved_tips = None
if "weather_summary" not in st.session_state:
    st.session_state.weather_summary = None
if "db_save_message" not in st.session_state:
    st.session_state.db_save_message = None
if "db_save_error" not in st.session_state:
    st.session_state.db_save_error = None
if "loaded_trip_id" not in st.session_state:
    st.session_state.loaded_trip_id = None

# Build / load the tips vector store once on startup (cached across reruns).
try:
    if get_openai_api_key():
        init_tips_collection()
except MissingAPIKeyError:
    pass
except Exception as exc:
    st.sidebar.warning(f"Tips index unavailable: {exc}")

st.markdown(
    """
    <div class="hero-banner">
        <h1>AI Trip Planner</h1>
        <p>Craft a day-by-day journey tailored to your destination, pace, and travel style.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Trip details")
    destination = st.text_input("Destination", placeholder="e.g. Lisbon, Portugal")
    days = st.number_input("Number of days", min_value=1, max_value=14, value=5, step=1)
    travel_style = st.selectbox(
        "Travel style",
        options=["Budget", "Balanced", "Luxury"],
        index=1,
    )
    plan_clicked = st.button("Plan my trip", type="primary")

    if not get_openai_api_key():
        st.info(
            "Add your OpenAI API key to a `.env` file as OPENAI_API_KEY before planning."
        )

    st.markdown("---")
    st.markdown("### My past trips")

    if not get_database_url():
        st.caption("Set DATABASE_URL to browse saved trips.")
    else:
        try:
            past_trips = list_trips()
        except Exception as exc:
            past_trips = []
            st.caption(f"Could not load trips: {exc}")

        if not past_trips:
            st.caption("No saved trips yet.")
        else:
            for trip in past_trips:
                created = trip["created_at"]
                if hasattr(created, "strftime"):
                    date_label = created.strftime("%Y-%m-%d %H:%M")
                else:
                    date_label = str(created)

                label = f"{trip['destination']} · {date_label}"
                if st.session_state.loaded_trip_id == trip["id"]:
                    label = f"● {label}"

                cols = st.columns([3.2, 1])
                with cols[0]:
                    if st.button(
                        label,
                        key=f"load_trip_{trip['id']}",
                        use_container_width=True,
                    ):
                        fresh = get_trip_by_id(trip["id"])
                        if fresh:
                            apply_saved_trip(fresh)
                            st.rerun()
                        else:
                            st.session_state.db_save_error = (
                                f"Trip id={trip['id']} was not found."
                            )
                with cols[1]:
                    if st.button(
                        "Delete",
                        key=f"delete_trip_{trip['id']}",
                        use_container_width=True,
                    ):
                        try:
                            delete_trip(trip["id"])
                            if st.session_state.loaded_trip_id == trip["id"]:
                                st.session_state.itinerary = None
                                st.session_state.trip_meta = None
                                st.session_state.retrieved_tips = None
                                st.session_state.weather_summary = None
                                st.session_state.loaded_trip_id = None
                            st.session_state.db_save_message = (
                                f"Trip deleted successfully (id={trip['id']})."
                            )
                            st.session_state.db_save_error = None
                            st.rerun()
                        except Exception as exc:
                            st.session_state.db_save_error = (
                                f"Could not delete trip: {exc}"
                            )
                            st.session_state.db_save_message = None

if plan_clicked:
    if not destination.strip():
        st.warning("Please enter a destination to start planning.")
    elif not get_openai_api_key():
        st.error(
            "OpenAI API key is missing. Add OPENAI_API_KEY to your `.env` file "
            "(or Streamlit secrets), restart the app, then try again."
        )
    else:
        with st.spinner("Planning your trip..."):
            try:
                dest = destination.strip()
                preference = f"{travel_style} trip to {dest}"
                tips = retrieve_tips(preference)
                weather = get_weather_for_destination(dest)
                itinerary = generate_itinerary(
                    dest,
                    int(days),
                    travel_style,
                    tips=tuple(tips),
                    weather=weather,
                )
                st.session_state.itinerary = itinerary
                st.session_state.retrieved_tips = tips
                st.session_state.weather_summary = weather
                st.session_state.loaded_trip_id = None
                st.session_state.trip_meta = {
                    "destination": dest,
                    "days": int(days),
                    "travel_style": travel_style,
                }
                try:
                    trip_id = save_trip(dest, int(days), travel_style, itinerary)
                    st.session_state.loaded_trip_id = trip_id
                    st.session_state.db_save_message = (
                        f"Trip inserted successfully (id={trip_id})."
                    )
                    st.session_state.db_save_error = None
                except Exception as db_exc:
                    st.session_state.db_save_message = None
                    st.session_state.db_save_error = (
                        f"Trip planned, but could not save to the database: {db_exc}"
                    )
            except MissingAPIKeyError as exc:
                st.session_state.itinerary = None
                st.session_state.retrieved_tips = None
                st.session_state.weather_summary = None
                st.session_state.trip_meta = None
                st.session_state.loaded_trip_id = None
                st.session_state.db_save_message = None
                st.session_state.db_save_error = None
                st.error(str(exc))
            except Exception as exc:
                st.session_state.itinerary = None
                st.session_state.retrieved_tips = None
                st.session_state.weather_summary = None
                st.session_state.trip_meta = None
                st.session_state.loaded_trip_id = None
                st.session_state.db_save_message = None
                st.session_state.db_save_error = None
                st.error(f"Could not generate itinerary: {exc}")

if st.session_state.db_save_message:
    st.success(st.session_state.db_save_message)
if st.session_state.db_save_error:
    st.warning(st.session_state.db_save_error)

if st.session_state.trip_meta and st.session_state.itinerary:
    meta = st.session_state.trip_meta
    chips = [
        meta["destination"],
        f"{meta['days']} day{'s' if meta['days'] != 1 else ''}",
        meta["travel_style"],
    ]
    chips_html = "".join(
        f'<span class="meta-chip">{html.escape(str(chip))}</span>' for chip in chips
    )
    st.markdown(chips_html, unsafe_allow_html=True)
    render_weather_badge(st.session_state.weather_summary)
    render_summary_and_download(
        st.session_state.itinerary,
        meta,
        st.session_state.weather_summary,
        st.session_state.retrieved_tips or [],
    )
    for day in st.session_state.itinerary:
        render_day_card(day)
    render_tips_box(st.session_state.retrieved_tips or [])
else:
    render_empty_state()
