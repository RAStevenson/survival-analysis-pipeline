# datasets/

Real public data used by the repository's real-data demonstration. Nothing
here is synthetic.

## chicago_licences.csv.gz

Every City of Chicago business licence whose first issue falls after
2002-01-01: 262,763 licences across 149 licence types, from restaurants and
taverns to home repair, tobacco retail and one-off event permits. Each row is
one licence, observed from the day it was first issued until the business
stopped holding it (`closed` = 1, 83.6 percent) or the 2026-08-01 cutoff while
still current (`closed` = 0, 16.4 percent). Median observed life is 759 days.

Features are restricted to what was knowable on the first day: licence type,
conditional-approval flag, ward, community area, police district, zip code,
latitude and longitude. Renewal count is deliberately excluded, because more
renewals means a longer life and it would simply be the outcome in disguise.

**Source.** City of Chicago open data portal, "Business Licenses" dataset
(`r5kz-chrr`), pulled through the unauthenticated Socrata API:
https://data.cityofchicago.org/resource/r5kz-chrr.csv

**Terms of use.** The portal publishes this dataset under the city's open data
terms, which permit redistribution with attribution and disclaim any warranty
of accuracy. Attribution is the Source line above. The repository's MIT licence
covers the code, not this file: the data stays the city's, under the city's
terms. The file carries no
business name and no street address, only the administrative geography the
portal publishes. One residual worth naming rather than leaving unsaid: for a
sole proprietor the recorded latitude and longitude may be a home address, and
at six decimal places those coordinates plus the issue date are close to a
join key back into the portal, which does carry names. Nothing here identifies
anyone on its own, and re-identification would require deliberately going back
to the source, but the possibility is real and is the reason no finer
geography is carried.

**Cleaning.** The committed file is produced by
`python scripts/run_prepare_chicago.py`, which documents every step. In short:
group the issuance and renewal transactions by licence number, keep only
licences whose earliest transaction is a genuine first issue, take that start
date, end at the cancellation date where one exists and otherwise at the
latest expiry, and treat an end date before the cutoff as an observed closure.
No values are altered.

**Why the file is committed rather than fetched on demand.** The portal is
live and changes daily, so re-running the preparation script would return a
different dataset and the numbers in the report would no longer match. The
committed snapshot is what keeps every figure in the report checkable. The
fitted models are the opposite case: they are reproducible from this file plus
the code, so they are regenerated rather than stored.

**Suitability, and the filter that earns it.** The pipeline evaluates on
temporal folds and re-censors training labels at each split date, which
assumes every row is observed from its own start. Where that assumption fails
the failure is silent: nothing errors, the early folds simply run out of
events. Two other registries were built and rejected for exactly this reason
before Chicago was chosen. The EIA-860 power-generator file lists
commissioning dates back to 1891 while its retirement records effectively
start in 2001, so a unit commissioned in 1960 is unobserved for four decades.
The FDIC insured-institution register is the same shape and worse:
establishment dates from 1782, closure records only from 1970. Both are
snapshots, meaning what exists now plus recent endings. This pipeline needs a
cohort, meaning entities followed from their own time zero.

Chicago qualifies because the city logs issuance, renewal and cancellation
continuously in one system, so a licence issued in 2004 has its whole life on
file. But the same test has to be applied inside the dataset, not only across
candidate datasets. The pull begins at 2002 because that is where the portal's
status history becomes complete, and a licence already running in 2001 still
appears, entering at its first post-2002 renewal. Its recorded start is that
renewal date and its recorded span is only what was left of its life, given it
had already survived at least one term. That was 79,690 rows, 23 percent of
the original pull. The tell was a spike of 61,351 licences dated to 2002 while
genuine first issues ran flat at roughly 14,000 a year either side of it.
Keeping only licences whose earliest transaction is a first issue removes it,
and that filter is what makes the assumption above true rather than merely
asserted.

**One flag the fit command needs.** Ward, community area, police district and
zip code are administrative codes, not quantities. Nothing in a CSV
distinguishes the two, so they are read as numbers unless the fit command says
otherwise, and a linear model then fits them a single monotonic slope: the
claim that hazard rises steadily with ward number. Pass them to
`--categorical-cols` so they are one-hot encoded, as the command in the root
README does.
