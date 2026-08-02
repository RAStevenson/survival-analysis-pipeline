# datasets/

Real public data used by the repository's real-data demonstration. Nothing
here is synthetic.

## chicago_licences.csv.gz

Every City of Chicago business licence first issued after 2002-01-01:
342,453 licences across 150 licence types, from restaurants and taverns to
home repair, tobacco retail and one-off event permits. Each row is one
licence, observed from the day it was first issued until the business stopped
holding it (`closed` = 1, 84.6 percent) or the 2026-08-01 cutoff while still
current (`closed` = 0, 15.4 percent). Median observed life is 977 days.

Features are restricted to what was knowable on the first day: licence type,
application type, conditional-approval flag, ward, community area, police
district, zip code, latitude and longitude. Renewal count is deliberately
excluded, because more renewals means a longer life and it would simply be
the outcome in disguise.

**Source.** City of Chicago open data portal, "Business Licenses" dataset
(`r5kz-chrr`), pulled through the unauthenticated Socrata API:
https://data.cityofchicago.org/resource/r5kz-chrr.csv

**Cleaning.** The committed file is produced by
`python scripts/run_prepare_chicago.py`, which documents every step. In
short: group the 1.19 million issuance and renewal transactions by licence
number, take the earliest start date, end at the cancellation date where one
exists and otherwise at the latest expiry, and treat an end date before the
cutoff as an observed closure. No values are altered.

**Why the file is committed rather than fetched on demand.** The portal is
live and changes daily, so re-running the preparation script would return a
different dataset and the numbers in the report would no longer match. The
committed snapshot is what keeps every figure in the report checkable. The
fitted models are the opposite case: they are reproducible from this file
plus the code, so they are regenerated rather than stored.

**Suitability.** The pipeline evaluates on temporal folds and re-censors
training labels at each split date, which assumes every row is observed from
its own start. Chicago satisfies that because the city records issuance,
renewal and cancellation continuously in one system, so a licence issued in
2004 has its whole life on file.
