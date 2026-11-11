# NFL value finder

Compares TAB head-to-head prices with Pinnacle's de-vigged price and a simple model, and prints
expected value per $1 staked.

Sections printed:
- **Win rates**: season win %, home/away, last 5, against the spread, shrunk toward 50%.
- **Key players out**: injury report (Out/Doubtful/Questionable) filtered to likely starters and
  weighted by position (QB counts most).
- **Value table**: TAB odds vs sharp fair probability vs model probability, with QB-out flags.

```bash
pip install -r requirements.txt
export ODDS_API_KEY=your_key   # the-odds-api.com
python nfl_value.py
```

Notes
- Pinnacle already prices in public injury news, so injuries only adjust the model. The edge to look
  for is TAB lagging after a key player is ruled out.
- Constants at the top of `nfl_value.py` (margin per EPA, home advantage, position weights, blend
  weights) are rough starting points. Backtest before trusting them.
- Not tested against the live APIs yet; check the bookmaker keys (`tab`, `pinnacle`) and the
  nflverse column names on first run.
