"""Venue altitude modelling.

The real, documented venue effect in international football is altitude: teams
ascending to high-altitude venues suffer measurable physical decline, while
acclimatised home sides do not. (CONMEBOL qualifiers at La Paz/Quito/Bogotá and
African matches at Addis Ababa/Johannesburg are the textbook cases.)

For the 2026 World Cup, only the two Mexican venues are materially elevated:
Estadio Azteca, Mexico City (2,240 m) and Estadio Akron, Guadalajara (1,566 m).
Monterrey (~540 m) and every US/Canadian host city are effectively sea level.
(Denver was eliminated during host selection — it is not a 2026 venue.)

We model the effect as a per-team "altitude baseline" (the typical elevation a
team plays its home games at) and a per-match "altitude gap" = how far ABOVE
that baseline the venue sits. The penalty applies to ascending only; playing
lower than usual is not penalised.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

# City -> altitude in metres. Curated for (a) high-altitude football cities that
# appear in the historical record and (b) all 16 WC 2026 host cities. Any city
# not listed defaults to 0 (sea-level class) — accurate to first order, since
# altitude effects are negligible below ~1,000 m.
CITY_ALTITUDE_M = {
    # --- South America (CONMEBOL altitude venues) ---
    "La Paz": 3640, "El Alto": 4150, "Oruro": 3700, "Cochabamba": 2560,
    "Sucre": 2810, "Potosí": 4070, "Quito": 2850, "Ambato": 2580,
    "Riobamba": 2750, "Bogotá": 2640, "Bogota": 2640, "Pasto": 2527,
    "Tunja": 2820, "Cusco": 3400, "Cuzco": 3400, "Arequipa": 2335,
    "Huancayo": 3250, "Mérida": 1630,
    # --- Mexico ---
    "Mexico City": 2240, "Ciudad de México": 2240, "Toluca": 2660,
    "Puebla": 2135, "Pachuca": 2400, "Querétaro": 1820, "Queretaro": 1820,
    "Guadalajara": 1566, "León": 1815, "Leon": 1815, "San Luis Potosí": 1860,
    "Aguascalientes": 1880, "Monterrey": 540,
    # --- Africa ---
    "Addis Ababa": 2355, "Johannesburg": 1753, "Pretoria": 1339,
    "Bloemfontein": 1395, "Soweto": 1700, "Nairobi": 1795, "Kampala": 1190,
    "Asmara": 2325, "Windhoek": 1700, "Harare": 1490, "Antananarivo": 1280,
    "Sana'a": 2250, "Sanaa": 2250, "Ifrane": 1665, "Kigali": 1567,
    "Polokwane": 1310, "Rustenburg": 1500, "Mmabatho": 1500, "Mafikeng": 1500,
    "Lusaka": 1280, "Ndola": 1300, "Gaborone": 1010, "Maseru": 1600,
    "Mbabane": 1240, "Bujumbura": 774, "Lilongwe": 1050, "Blantyre": 1039,
    # --- Asia (high) ---
    "Tehran": 1200, "Mashhad": 985, "Ulaanbaatar": 1300, "Bishkek ": 800,
    # --- North/Central America ---
    "Denver": 1609, "Guatemala City": 1500, "San José": 1170, "San Jose": 1170,
    "Tegucigalpa": 990, "Calgary": 1045,
    # --- Asia/Europe high spots ---
    "Almaty": 850, "Bishkek": 800, "Kathmandu": 1400, "Thimphu": 2320,
    "Erzurum": 1900, "Addis": 2355, "Sucre ": 2810,
    # --- WC 2026 host cities (low-altitude) ---
    "Toronto": 76, "Vancouver": 4, "Atlanta": 320, "Boston": 43,
    "Foxborough": 90, "Dallas": 131, "Arlington": 184, "Houston": 24,
    "Kansas City": 277, "Los Angeles": 71, "Inglewood": 38, "Miami": 2,
    "Miami Gardens": 3, "East Rutherford": 3, "New York": 10,
    "Philadelphia": 12, "Santa Clara": 9, "San Francisco": 16, "Seattle": 53,
}

# WC 2026 elevated-venue group fixtures, keyed by the unordered pair of canonical
# team names. Verified from the official schedule (Azteca + Akron), 2026-06-13.
WC2026_VENUE = {
    frozenset({"Mexico", "South Africa"}): ("Mexico City", 2240),
    frozenset({"Uzbekistan", "Colombia"}): ("Mexico City", 2240),
    frozenset({"Mexico", "Czech Republic"}): ("Mexico City", 2240),
    frozenset({"South Korea", "Czech Republic"}): ("Guadalajara", 1566),
    frozenset({"Mexico", "South Korea"}): ("Guadalajara", 1566),
    frozenset({"Colombia", "DR Congo"}): ("Guadalajara", 1566),
    frozenset({"Uruguay", "Spain"}): ("Guadalajara", 1566),
}


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", str(s)).strip()


def altitude(city: str | float | None) -> float:
    """Altitude in metres for a city name (0 if unknown / sea-level class)."""
    if city is None or (isinstance(city, float) and np.isnan(city)):
        return 0.0
    return float(CITY_ALTITUDE_M.get(_nfc(city), 0.0))


def team_baselines(history: pd.DataFrame) -> dict[str, float]:
    """Each team's home-altitude baseline = median altitude of its home venues.

    Uses non-neutral matches (team playing in its own country). Teams that never
    play at altitude get 0; Mexico ~2240, Bolivia ~3600, Ecuador ~2850, etc.
    """
    alt = history["city"].map(altitude)
    home = history.loc[~history["neutral"].astype(bool)]
    home_alt = alt.loc[home.index]
    baselines: dict[str, float] = {}
    for team, idx in home.groupby("home_team").groups.items():
        vals = home_alt.loc[idx]
        baselines[team] = float(vals.median()) if len(vals) else 0.0
    return baselines


def wc2026_altitude(home_team: str, away_team: str) -> float:
    """Venue altitude for a WC 2026 fixture (0 for the sea-level majority)."""
    v = WC2026_VENUE.get(frozenset({home_team, away_team}))
    return float(v[1]) if v else 0.0


def wc2026_venue_city(home_team: str, away_team: str) -> str | None:
    v = WC2026_VENUE.get(frozenset({home_team, away_team}))
    return v[0] if v else None
