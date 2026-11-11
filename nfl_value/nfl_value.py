"""
NFL value finder
  1) Odds: The Odds API (TAB = soft book, Pinnacle = sharp book)
  2) Stats: nflverse via nflreadpy
       - EPA/play power ratings
       - WIN RATES: overall / home / away / last 5 / against the spread, plus a log5 matchup probability
       - KEY PLAYERS OUT: official injury report filtered to likely starters, weighted by position
  3) Compares TAB's price against (a) the de-vigged Pinnacle price and (b) our blended model.

Note on injuries: Pinnacle's price already contains public injury news, so injuries are applied
to OUR MODEL only. The edge to look for is TAB being slow to move after a key player is ruled out
(the model and the sharp price both move, TAB's price does not).

Setup:
    pip install -r requirements.txt
    export ODDS_API_KEY=your_key      # free tier at the-odds-api.com
Run:
    python nfl_value.py
"""
import os
import re
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests

API_KEY = os.environ["ODDS_API_KEY"]
SOFT, SHARP = "tab", "pinnacle"      # bookmaker keys used by The Odds API
SEASON = 2026

# ---- model constants (tune by backtesting) ----
SIGMA = 13.5            # std dev of NFL margin vs expectation (points)
HFA = 1.5               # home field advantage (points)
MARGIN_PER_EPA = 40     # points per 1.0 EPA/play rating gap
PRIOR_GAMES = 6         # shrink win rates toward 50% as if N coin-flip games were played
W_EPA, W_WINRATE = 0.65, 0.35   # blend weights for the two margin estimates

# ---- injury constants ----
STARTER_SNAP_PCT = 0.60         # avg snap share over recent games to count as a starter
STARTER_LOOKBACK_WEEKS = 4
MAX_INJ_POINTS = 10.0           # cap on total lineup penalty per team
STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.25}
# Rough points of margin lost when a starter at the position is missing.
POS_POINTS = {
    "QB": 4.0, "WR": 0.8, "RB": 0.6, "TE": 0.5,
    "T": 0.6, "G": 0.5, "C": 0.5, "OL": 0.5, "OT": 0.6, "OG": 0.5,
    "DE": 0.6, "DT": 0.6, "DL": 0.6, "EDGE": 0.7, "OLB": 0.5, "LB": 0.4, "ILB": 0.4, "MLB": 0.4,
    "CB": 0.7, "S": 0.4, "FS": 0.4, "SS": 0.4, "DB": 0.5,
}

ODDS_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
ND = NormalDist()

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
# Different data sources spell a few teams differently.
ABBR_ALIAS = {"LAR": "LA", "LVR": "LV", "WSH": "WAS", "JAC": "JAX", "ARZ": "ARI",
              "BLT": "BAL", "CLV": "CLE", "HST": "HOU", "SD": "LAC", "OAK": "LV", "STL": "LA"}


def abbr(team_name):
    code = TEAM_ABBR.get(team_name, team_name)
    return ABBR_ALIAS.get(code, code)


# ---------- odds maths ----------
def devig(prices):
    """Decimal odds -> fair probabilities (proportional margin removal)."""
    inv = [1 / p for p in prices]
    total = sum(inv)
    return [i / total for i in inv]


def ev(prob, decimal_odds):
    """Expected profit per $1 staked."""
    return prob * decimal_odds - 1


def log5(a, b):
    """Win probability of a team with win rate a vs a team with win rate b."""
    return a * (1 - b) / (a * (1 - b) + b * (1 - a))


# ---------- odds scraping ----------
def fetch_odds():
    resp = requests.get(
        ODDS_URL,
        params={
            "apiKey": API_KEY,
            "regions": "au,eu",        # au has TAB, eu has Pinnacle
            "markets": "h2h",
            "oddsFormat": "decimal",
            "bookmakers": f"{SOFT},{SHARP}",
        },
        timeout=20,
    )
    resp.raise_for_status()
    print("API credits remaining:", resp.headers.get("x-requests-remaining"))
    return resp.json()


def h2h_prices(game, book_key):
    for book in game["bookmakers"]:
        if book["key"] == book_key:
            for market in book["markets"]:
                if market["key"] == "h2h":
                    return {o["name"]: o["price"] for o in market["outcomes"]}
    return None


# ---------- stats: EPA ratings ----------
def build_epa_ratings(season):
    """EPA/play power rating per team: offence EPA minus EPA allowed."""
    import nflreadpy as nfl

    pbp = nfl.load_pbp([season]).to_pandas()
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()]
    off = plays.groupby("posteam")["epa"].mean()
    dfn = plays.groupby("defteam")["epa"].mean()   # lower = better defence
    return (off - dfn).to_dict()


def epa_margin(home, away, ratings):
    h, a = abbr(home), abbr(away)
    if h not in ratings or a not in ratings:
        return None
    return MARGIN_PER_EPA * (ratings[h] - ratings[a]) + HFA


# ---------- stats: WIN RATES ----------
def build_win_rates(season):
    """Per-team win rates and ATS record for the season so far (regular season)."""
    import nflreadpy as nfl

    sched = nfl.load_schedules([season]).to_pandas()
    played = sched[(sched["game_type"] == "REG") & sched["result"].notna()].copy()
    cols = ["team", "week", "home", "win", "cover"]
    if played.empty:
        return pd.DataFrame(columns=cols[1:]).rename_axis("team"), sched

    res = played["result"].to_numpy()                 # home score - away score
    spr = played["spread_line"].to_numpy()            # positive = home favoured
    home_win = np.where(res > 0, 1.0, np.where(res == 0, 0.5, 0.0))
    home_cover = np.where(res > spr, 1.0, np.where(res == spr, 0.5, 0.0))
    long = pd.concat([
        pd.DataFrame({"team": played["home_team"], "week": played["week"], "home": 1,
                      "win": home_win, "cover": home_cover}),
        pd.DataFrame({"team": played["away_team"], "week": played["week"], "home": 0,
                      "win": 1 - home_win, "cover": 1 - home_cover}),
    ])

    rows = []
    for team, g in long.sort_values("week").groupby("team"):
        n = len(g)
        rows.append({
            "team": team,
            "games": n,
            "win_pct": g["win"].mean(),
            "home_win_pct": g.loc[g["home"] == 1, "win"].mean(),
            "away_win_pct": g.loc[g["home"] == 0, "win"].mean(),
            "last5_win_pct": g["win"].tail(5).mean(),
            "ats_pct": g["cover"].mean(),
            "shrunk_win_pct": (g["win"].sum() + PRIOR_GAMES * 0.5) / (n + PRIOR_GAMES),
        })
    return pd.DataFrame(rows).set_index("team"), sched


def winrate_margin(home, away, rates):
    h, a = abbr(home), abbr(away)
    if h not in rates.index or a not in rates.index:
        return None
    p = log5(rates.loc[h, "shrunk_win_pct"], rates.loc[a, "shrunk_win_pct"])
    p = min(max(p, 0.02), 0.98)
    return SIGMA * ND.inv_cdf(p) + HFA


# ---------- stats: KEY PLAYERS OUT ----------
def norm_name(name):
    s = re.sub(r"[^a-z ]", "", str(name).lower())
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def current_week(sched, season):
    upcoming = sched[(sched["season"] == season) & sched["result"].isna()]
    return int(upcoming["week"].min()) if not upcoming.empty else None


def starter_names(season):
    """Players whose recent snap share marks them as starters (matched by name)."""
    import nflreadpy as nfl

    snaps = nfl.load_snap_counts([season]).to_pandas()
    snaps["pct"] = snaps[["offense_pct", "defense_pct"]].max(axis=1)
    if snaps["pct"].max() > 1.5:                      # stored as 0-100 rather than 0-1
        snaps["pct"] = snaps["pct"] / 100
    recent = snaps[snaps["week"] > snaps["week"].max() - STARTER_LOOKBACK_WEEKS]
    avg = recent.groupby("player")["pct"].mean()
    return {norm_name(p) for p in avg[avg >= STARTER_SNAP_PCT].index}


def build_injury_impacts(season, week):
    """{team_abbr: {points, notes, qb_out}} for starters Out / Doubtful / Questionable."""
    import nflreadpy as nfl

    inj = nfl.load_injuries([season]).to_pandas()
    inj = inj[(inj["week"] == week) & inj["report_status"].isin(STATUS_WEIGHT)]
    try:
        starters = starter_names(season)
    except Exception as e:                            # snap data missing -> count everyone
        print(f"(starter filter unavailable, counting all listed players: {e})")
        starters = set()

    impacts = {}
    for team, g in inj.groupby("team"):
        pts, notes, qb_out = 0.0, [], False
        for _, r in g.iterrows():
            if starters and norm_name(r["full_name"]) not in starters:
                continue
            status, pos = r["report_status"], r["position"]
            pts += POS_POINTS.get(pos, 0.2) * STATUS_WEIGHT[status]
            notes.append(f"{r['full_name']} ({pos}, {status})")
            if pos == "QB" and status in ("Out", "Doubtful"):
                qb_out = True
        impacts[ABBR_ALIAS.get(team, team)] = {
            "points": min(pts, MAX_INJ_POINTS), "notes": notes, "qb_out": qb_out}
    return impacts


# ---------- model assembly ----------
def blend(parts):
    parts = [(w, m) for w, m in parts if m is not None]
    if not parts:
        return None
    return sum(w * m for w, m in parts) / sum(w for w, _ in parts)


def safe(label, fn, default):
    try:
        return fn()
    except Exception as e:
        print(f"[warn] {label} unavailable: {e}")
        return default


def main():
    epa = safe("EPA ratings", lambda: build_epa_ratings(SEASON), {})
    rates, sched = safe("win rates", lambda: build_win_rates(SEASON),
                        (pd.DataFrame(columns=["shrunk_win_pct"]), None))
    week = current_week(sched, SEASON) if sched is not None else None
    injuries = safe("injuries", lambda: build_injury_impacts(SEASON, week), {}) if week else {}

    games = fetch_odds()
    value_rows, win_rows, lineup_rows = [], [], []
    seen_teams = set()

    for g in games:
        soft, sharp = h2h_prices(g, SOFT), h2h_prices(g, SHARP)
        if not soft or not sharp:
            continue
        home, away = g["home_team"], g["away_team"]
        h, a = abbr(home), abbr(away)
        sharp_fair = dict(zip(sharp, devig(list(sharp.values()))))

        inj_h = injuries.get(h, {"points": 0.0, "notes": [], "qb_out": False})
        inj_a = injuries.get(a, {"points": 0.0, "notes": [], "qb_out": False})

        base = blend([(W_EPA, epa_margin(home, away, epa)),
                      (W_WINRATE, winrate_margin(home, away, rates))])
        # Home margin: lineup penalties hurt the team missing players.
        margin = None if base is None else base + inj_a["points"] - inj_h["points"]
        p_home = None if margin is None else ND.cdf(margin / SIGMA)

        for t, ab in ((home, h), (away, a)):
            if ab not in seen_teams and ab in rates.index:
                seen_teams.add(ab)
                win_rows.append({"team": ab, **rates.loc[ab].round(3).to_dict()})
        if inj_h["notes"] or inj_a["notes"]:
            lineup_rows.append((f"{away} @ {home}", a, inj_a, h, inj_h))

        for team, price in soft.items():
            ab, inj_for, inj_opp = (h, inj_h, inj_a) if team == home else (a, inj_a, inj_h)
            p_sharp = sharp_fair[team]
            p_model = None if p_home is None else (p_home if team == home else 1 - p_home)
            flag = []
            if inj_for["qb_out"]:
                flag.append(f"{ab} QB OUT")
            if inj_opp["qb_out"]:
                flag.append(f"{abbr(away if team == home else home)} QB OUT")
            value_rows.append({
                "game": f"{away} @ {home}",
                "bet": team,
                "tab_odds": price,
                "sharp_fair_p": round(p_sharp, 3),
                "ev_vs_sharp": round(ev(p_sharp, price), 3),
                "model_p": None if p_model is None else round(p_model, 3),
                "ev_vs_model": None if p_model is None else round(ev(p_model, price), 3),
                "lineup_pts_lost": round(inj_for["points"], 1),
                "flags": ", ".join(flag),
            })

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)

    print(f"\n=== WIN RATES, {SEASON} regular season (shrunk toward 50% by {PRIOR_GAMES} games) ===")
    if win_rows:
        print(pd.DataFrame(win_rows).sort_values("shrunk_win_pct", ascending=False).to_string(index=False))
    else:
        print("no games played yet / no data")

    print(f"\n=== KEY PLAYERS OUT / DOUBTFUL / QUESTIONABLE (week {week}) ===")
    if lineup_rows:
        for title, a, inj_a, h, inj_h in lineup_rows:
            print(f"\n{title}")
            for ab, inj in ((a, inj_a), (h, inj_h)):
                if inj["notes"]:
                    print(f"  {ab} (-{inj['points']:.1f} pts): " + "; ".join(inj["notes"]))
    else:
        print("no starters listed on the injury report yet (reports usually land Wed-Fri)")

    print("\n=== VALUE: TAB price vs sharp fair price and vs model ===")
    if value_rows:
        print(pd.DataFrame(value_rows).sort_values("ev_vs_sharp", ascending=False).to_string(index=False))
    else:
        print("no games with both TAB and Pinnacle prices")


if __name__ == "__main__":
    main()
