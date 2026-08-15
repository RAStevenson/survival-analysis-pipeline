# datasets/

Real public data used by the repository's real-data demonstration. Nothing
here is synthetic. It is real data with real results.

## chicago_licences.csv.gz

This dataset includes every City of Chicago business licence whose first
issue falls after 2002-01-01, excluding licence types that are temporary by
construction (the exclusion is described under Cleaning below). It spans
239,721 licences across 137 licence types, from restaurants and taverns to
home repair and tobacco retail. Each row is one licence, observed from the
day it was first issued until either the business stopped holding it or the
2026-08-01 cutoff. 82.2 percent had closed by the cutoff. The remaining
17.8 percent were still current. The `closed` column records these events
with 1 for an observed closure and 0 for a licence still running at the cutoff.
Median observed licence life in the set is 904 days.

Features are restricted to what was knowable on the first day of licence issuance.
This includes licence type, conditional-approval flag, ward, community area,
police district, zip code, latitude, and longitude. Renewal count is
deliberately excluded, because more renewals means a longer life. It
contains look-ahead information about the outcome and including it would be
information leakage.

**Source.** 

City of Chicago open data portal, "Business Licenses" dataset
(`r5kz-chrr`), pulled through the unauthenticated Socrata API:
https://data.cityofchicago.org/resource/r5kz-chrr.csv

**Terms of use.** 

The portal publishes this dataset under the city's open
data terms, which permit redistribution with attribution and disclaim any
warranty of accuracy. Attribution is the Source line above. The
repository's MIT licence covers the code, not this file. The data stays
the city's, under the city's terms. The file carries no business name and
no street address, only the administrative geography the portal publishes.

**Cleaning.** 

The committed file is produced by `python scripts/run_prepare_chicago.py`, 
which documents every step. It groups the issuance and renewal transactions 
by licence number, keeps only licences whose earliest transaction is a genuine 
first issue, takes that start date, ends at the cancellation date where one 
exists and otherwise at the latest expiry, and treats an end date before the 
cutoff as an observed closure. No values are altered.

**Why the file is committed rather than fetched on demand.** 

The portal is
live and changes daily, so re-running the preparation script would return
a different dataset and the numbers in the report would no longer match.
The committed snapshot is what keeps every figure in the report checkable.
The fitted models are the opposite case. They are reproducible from this
file plus the code, so they are regenerated rather than stored.

**The report's dataset-specific prose.** 

The generated Chicago report makes several claims that are about this dataset 
rather than the method. This includes the exclusion of licences whose life 
began before the records do (the left-truncation exclusion), the provisional 
endings near the cutoff, and the term-boundary steps in the survival curves. 
All of them come from the notes folder beside the run,
`reports/chicago_demo/notes/`, one markdown file per report slot. The report
template asserts nothing about any particular dataset, so the notes folder is
the only channel for dataset-specific claims, and it survives every refit and
rebuild. The root README's own-data section explains how to write notes for
your own data.

**Why this dataset? - Two failed alternatives.** 

The pipeline evaluates on temporal folds, training always on earlier rows than 
it tests, and re-censors training labels at each split date, rewriting each 
one to what was knowable then. Both steps assume every row is observed from its 
own start. Where that assumption fails, the failure would be silent. Nothing
errors, and the early folds simply run out of events.

Because of this limitation, two other registries were tried and rejected
before the Chicago data was chosen. This wasn't hand picking. The
data genuinely was not reliable enough to qualify. The first candidate was
the EIA-860 power-generator file, which lists commissioning dates back to
1891, but its retirement records effectively start in 2001. This makes early dates
unusable for a survival study because large sections of data contain no death
records and we cannot know if older power generators are still in use or were
retired silently. Power generators survive for decades on average, so restricting the
data to units commissioned after 2001 leaves a group that is mostly
still running, with too few observed retirements to learn from. This
makes the data unsuited to this analysis.

The second candidate considered was the FDIC insured-institution register. It had
the same shape but worse, with establishment dates from 1782 and no closure records
until 1970. The data contains no death events until then. Banks also have relatively
long lives, and restricting to banks established after 1970 once again
leaves too few observed closures.

The Chicago business licence data is much better. The city has logged issuance,
renewal, and cancellation continuously in one system since 2002, so a licence
issued in 2004 has its whole life on file.

The dataset is retrieved from the portal's API starting at 2002-01-01,
because that is where the portal's records become complete. But a business
that opened before 2002 still shows up and its first renewal after 2002
gets downloaded and falsely looks like an opening. Such a row lists the wrong
start date and only the last part of what likely was a longer life.

What makes this obvious is observed volume. Taken at face value, the
data shows 61,351 supposed openings in 2002. But from the remaining
records we can observe that a normal year only has about 14,000 new
entries.

To correct this, we drop every licence whose earliest downloaded record
is a renewal rather than a first issue, which removed 79,690 rows (23 percent).
That step is what guarantees every remaining row is watched from its true start
but still leaves us with hundreds of thousands of valid rows.

One further cut removes licence types that are temporary by construction. 
This includes the Special Event, Pop-Up, and Itinerant variants. That is 
23,042 licences across 12 types with a median recorded life of four days. Their 
short lives are intent, not failure. A temporary food licence was always going to 
expire within days, while the handyman whose business folded never meant it to end. 
Only the second kind belongs in a study of business survival. A model that learns 
to spot event permits learns nothing about why businesses fail, so these rows are 
dropped before the file is written.

**One flag the fit command needs.** 

As explained in the root README.md, categorical columns must be named using 
`--categorical-cols` by the user. Ward, community area, police district, and zip 
code are labels, not quantities. A code's meaning lives in which place it names, 
not in the size of the number. Nothing in a CSV marks the difference, so these 
columns are read as numbers unless the fit command line says otherwise. This allows
every distinct value to get its own yes/no column (one-hot encoding), so ward 43 
gets its own estimated effect rather than a point on a line through all fifty.

The commands needed to reproduce this dataset and report are included and explained
in detail in the project's root README.md.
