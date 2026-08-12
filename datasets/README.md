# datasets/

Real public data used by the repository's real-data demonstration. Nothing
here is synthetic.

## chicago_licences.csv.gz

This dataset includes every City of Chicago business licence whose first
issue falls after 2002-01-01. It spans 262,763 licences across 149 licence
types, from restaurants and taverns to home repair, tobacco retail, and
one-off event permits. Each row is one licence, observed from the day it
was first issued until either the business stopped holding it or the
2026-08-01 cutoff. 83.6 percent had closed by the cutoff. The remaining
16.4 percent were still current. The `closed` column records which: 1 for
an observed closure, 0 for a licence still running at the cutoff. Median
observed life is 759 days.

Features are restricted to what was knowable on the first day. This
includes licence type, conditional-approval flag, ward, community area,
police district, zip code, latitude and longitude. Renewal count is
deliberately excluded, because more renewals means a longer life and it
would simply be the outcome in disguise.

**Source.** City of Chicago open data portal, "Business Licenses" dataset
(`r5kz-chrr`), pulled through the unauthenticated Socrata API:
https://data.cityofchicago.org/resource/r5kz-chrr.csv

**Terms of use.** The portal publishes this dataset under the city's open
data terms, which permit redistribution with attribution and disclaim any
warranty of accuracy. Attribution is the Source line above. The
repository's MIT licence covers the code, not this file. The data stays
the city's, under the city's terms. The file carries no business name and
no street address, only the administrative geography the portal publishes.
There is one residual risk worth naming rather than leaving unsaid. For a
sole proprietor the recorded latitude and longitude may be a home address,
and at six decimal places those coordinates plus the issue date are close
to a join key back into the portal, which does carry names. Nothing here
identifies anyone on its own, and re-identification would require
deliberately going back to the source, but the possibility is real and is
the reason no finer geography is carried.

**Cleaning.** The committed file is produced by
`python scripts/run_prepare_chicago.py`, which documents every step. In
short, it groups the issuance and renewal transactions by licence number,
keeps only licences whose earliest transaction is a genuine first issue,
takes that start date, ends at the cancellation date where one exists and
otherwise at the latest expiry, and treats an end date before the cutoff
as an observed closure. No values are altered.

**Why the file is committed rather than fetched on demand.** The portal is
live and changes daily, so re-running the preparation script would return
a different dataset and the numbers in the report would no longer match.
The committed snapshot is what keeps every figure in the report checkable.
The fitted models are the opposite case. They are reproducible from this
file plus the code, so they are regenerated rather than stored.

**Suitability, and the filter that earns it.** The pipeline evaluates on
temporal folds, training always on earlier rows than it tests, and
re-censors training labels at each split date, rewriting each one to what
was knowable then. Both steps assume every row is observed from its own
start. Where that assumption fails, the failure is silent. Nothing errors,
and the early folds simply run out of events. Two other registries were
tried and rejected for exactly this reason before Chicago was chosen. The
EIA-860 power-generator file lists commissioning dates back to 1891 while
its retirement records effectively start in 2001, so a unit commissioned
in 1960 is unobserved for four decades. The FDIC insured-institution
register is the same shape and worse, with establishment dates from 1782
and closure records only from 1970. Both are snapshots, meaning what
exists now plus recent endings. This pipeline needs a cohort, meaning
entities followed from their own time zero.

Chicago qualifies because the city logs issuance, renewal and cancellation
continuously in one system, so a licence issued in 2004 has its whole life
on file. But the same test has to be applied inside the dataset as well as
across candidate datasets. The pull begins at 2002 because that is where
the portal's status history becomes complete, and a licence already
running in 2001 still appears, entering at its first post-2002 renewal.
Its recorded start is that renewal date and its recorded span is only what
was left of its life, given it had already survived at least one term.
That was 79,690 rows, 23 percent of the original pull. The tell was a
spike of 61,351 licences dated to 2002 while genuine first issues ran flat
at roughly 14,000 a year either side of it. Keeping only licences whose
earliest transaction is a first issue removes it, and that filter is what
makes the assumption above true rather than merely asserted.

**One flag the fit command needs.** Ward, community area, police district
and zip code are labels, not quantities. A code's meaning lives in which
place it names, not in the size of the number. Nothing in a CSV marks the
difference, so these columns are read as numbers unless the fit command
says otherwise, and a linear model then fits each one as a straight-line
trend, as if the risk of a licence closing rose steadily from ward 1 to
ward 50. Passing them to `--categorical-cols` treats each code as a label
instead. Every distinct value gets its own yes/no column (one-hot
encoding), so ward 43 gets its own estimated effect rather than a point on
a line through all fifty. The command in the root README does this.
