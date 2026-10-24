# Trading notebooks

## Running the ASX stock screener

When asked to "run the screener" (from a schedule, the phone app, or anywhere
else), run `ASX_Stock_Screener.ipynb` headlessly:

```bash
pip install -r requirements.txt
SCREENER_TARGET=<target> jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 ASX_Stock_Screener.ipynb \
  --output /tmp/ASX_Stock_Screener_run.ipynb
```

`<target>` comes from the request:

| Request says | SCREENER_TARGET |
|---|---|
| nothing specific / "ASX 200" / "full" | `asx200` (the default) |
| a watchlist name, e.g. "banking", "green energy", "construction" | that name (case-insensitive) |
| a single stock, e.g. "BHP" or "CBA.AX" | the ticker code |

Watchlist names live in the `watchlists` dict in the notebook's config cell.
An unrecognised target fails fast with the list of valid names.

The full ASX 200 run takes 10+ minutes; a watchlist takes a minute or two.

Off Windows the workbook is written to `output/ASX_Stock_Screener_<target>.xlsx`.

### After the run
1. Commit `output/*.xlsx` and push to `main` so it can be downloaded from GitHub.
   Do not commit the executed notebook copy.
2. Reply with a short summary, read from the workbook's `Summary` and
   `Candidates` sheets: how many tickers were analysed, the top 10 by
   `Composite Score (0-24)`, every ticker in `Candidates`, and any tickers listed
   on `Data_Issues`.

For the lender watchlists (`Banking`, `Big Four Banks`), the Graham current-ratio,
Leverage and FCF metrics don't suit lenders, so the notebook ranks them by
`Lender Composite (0-9)` (Lender Score + Momentum) instead of the Composite
Score, and adds a `Lenders` sheet. Use that column in place of
`Composite Score (0-24)` for the top 10, and lead with P/E, P/B, ROE, dividend
yield and momentum.
