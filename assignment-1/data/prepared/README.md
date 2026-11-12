# Prepared labels supplied by teammate

`labels/` holds 10,464 local label copies and is excluded from Git.
`removed_rows.txt` is the supplied log of 17 zero-area annotation rows removed.
The original dataset labels were not replaced. Verify these copies against the
originals before using them for training. `prepare_split.py` has now verified
all 10,464 copies: they preserve every valid annotation in order and remove
exactly the 17 zero-area rows listed in the supplied log. No other annotation
changes were found. This verifies content preservation, not visual correctness.
