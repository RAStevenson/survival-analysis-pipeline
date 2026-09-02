# datasets/

Real public data used by the repository's real-data demonstrations.

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

This registry was chosen because the city has logged issuance, renewal, and
cancellation continuously in one system since 2002, so a licence issued in
2004 has its whole life on file. Two registries were tried and rejected
before it. The EIA-860 power-generator file lists commissioning dates back
to 1891 but records retirements only from 2001, and the FDIC
insured-institution register lists establishments from 1782 with no closures
before 1970. In both, restricting to rows that start after the death records
begin leaves a population mostly still alive, with too few observed endings
to learn from.

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
by licence number, takes the first issue as the start date, ends at the
cancellation date where one exists and otherwise at the latest expiry, and
treats an end date before the cutoff as an observed closure. No values are
altered. Two kinds of licence are dropped before the file is written.

The first is any licence already running when the portal's records begin.
The portal is complete only from 2002, so a business that opened earlier
shows its first post-2002 renewal as if it were an opening, with the wrong
start date and only the tail of its life. Taken at face value the data
shows 61,351 openings in 2002 against roughly 14,000 in a normal year,
which is what exposed them. Dropping every licence whose earliest record
is a renewal rather than a first issue removed 79,690 rows, 23 percent of
the download, and guarantees every remaining row is watched from its true
start.

The second is licence types that are temporary by construction, the
Special Event, Pop-Up, and Itinerant variants, 23,042 licences across 12
types with a median recorded life of four days. Their short lives are
intent, not failure, and a model that learns to spot event permits learns
nothing about why businesses close.

Ward, community area, police district, and zip code are administrative
codes rather than quantities, and the fit command names them as
categorical with `--categorical-cols`. The full command is in the root
README.

**Why the file is committed rather than fetched on demand.**

The portal is
live and changes daily, so re-running the preparation script would return
a different dataset and the numbers in the report would no longer match.
The committed snapshot is what keeps every figure in the report checkable.
The fitted models are the opposite case. They are reproducible from this
file plus the code, so they are regenerated rather than stored.

## flchain.csv.gz

The serum free light chain cohort distributed with R's survival package: a
stratified random half-sample of a Mayo Clinic study of the relation
between serum free light chain and mortality, covering residents of
Olmsted County, Minnesota aged 50 and over. After preparation the file
holds 7,871 subjects, one row per subject, observed from blood sample to
death or the end of follow-up. 27.5 percent died during follow-up and the
rest are censored. Features are the intake measurements: age, sex, the
kappa and lambda free light chain concentrations, serum creatinine, the
assay decile group, and an MGUS diagnosis flag. The file also carries
flc_band, a three-level banding of the decile group derived during
preparation for the report's survival figure; the fit command excludes it
from the features with --drop-cols.

**Source.**

The flchain dataset ships with R's survival package. The committed file
was retrieved as CSV from the Rdatasets mirror:
https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/survival/flchain.csv
Benchmark figure quoted in the run's interpretation note: Tarkhan and
Simon, BigSurvSGD (arXiv:2003.00116). Original study: Dispenzieri et al., "Use of nonclonal serum
immunoglobulin free light chains to predict overall survival in the
general population," Mayo Clinic Proceedings 87:512-523 (2012).

**Terms of use.**

The dataset ships with R's survival package and is redistributed by the
Rdatasets project. It is de-identified study data, and it carries no names, no locations finer than the
county, and no dates finer than the sample year. The repository's MIT
licence covers the code, not this file.

**Cleaning.**

Produced by `python scripts/run_prepare_flchain.py`, which documents every
step. Columns are renamed to explicit names. The cause-of-death chapter
column is dropped because it is recorded at death and would hand the
model the outcome. Three subjects with zero days of follow-up are
removed. The flc_band column is derived from the decile group, and the
rows are shuffled within each sample year with a fixed seed. The source file lists subjects
in age order within each year, and with year-only dates that hidden order
would decide which subjects fall either side of a fold boundary, so the
shuffle makes the unknowable within-year order genuinely arbitrary. No
values are altered.
