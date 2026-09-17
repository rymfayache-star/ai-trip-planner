"""MCP server exposing travel-related weather tools."""

from __future__ import annotations

import requests

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError:
    # mcp 2.x renamed FastMCP -> MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (Open-Meteo)
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "foggy",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}

mcp = FastMCP("Travel Tools")


@mcp.tool()
def get_weather(city: str) -> str:
    """Get today's weather summary for a travel destination city.

    Call this tool whenever the user asks about current or today's weather,
    temperature, or outdoor conditions for a city, or when planning a trip
    and you need to factor in local weather. Pass the city name (optionally
    with country), for example "Lisbon" or "Paris, France".

    Args:
        city: City name to look up (e.g. "Tokyo", "New York").

    Returns:
        A short plain-text summary with today's temperature (°C) and conditions.
    """
    city = (city or "").strip()
    if not city:
        return "Please provide a city name to look up the weather."

    geo = requests.get(
        GEOCODING_URL,
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=15,
    )
    geo.raise_for_status()
    geo_data = geo.json()
    results = geo_data.get("results") or []
    if not results:
        return f"Could not find a location matching '{city}'."

    place = results[0]
    lat = place["latitude"]
    lon = place["longitude"]
    label = place.get("name", city)
    country = place.get("country")
    if country:
        label = f"{label}, {country}"

    forecast = requests.get(
        FORECAST_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "weathercode,temperature_2m_max,temperature_2m_min",
            "timezone": "auto",
            "forecast_days": 1,
        },
        timeout=15,
    )
    forecast.raise_for_status()
    daily = forecast.json().get("daily") or {}

    try:
        code = int(daily["weathercode"][0])
        temp_max = daily["temperature_2m_max"][0]
        temp_min = daily["temperature_2m_min"][0]
    except (KeyError, IndexError, TypeError, ValueError):
        return f"Weather data for {label} is currently unavailable."

    conditions = WEATHER_CODES.get(code, f"weather code {code}")
    return (
        f"Today in {label}: {conditions}, "
        f"high {temp_max}°C / low {temp_min}°C."
    )


if __name__ == "__main__":
    mcp.run()
