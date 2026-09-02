# datasets/

Real public data used by the repository's real-data demonstrations. Nothing
here is synthetic. It is real data with real results.

## chicago_licences.csv.gz

Every City of Chicago business licence whose first issue falls after
2002-01-01, excluding licence types that are temporary by construction
(the exclusion is described under Cleaning). The file spans 239,721
licences across 137 licence types, from restaurants and taverns to home
repair and tobacco retail. Each row is one licence, observed from the day
it was first issued until either the business stopped holding it or the
2026-08-01 cutoff. 82.2 percent had closed by the cutoff and 17.8 percent
were still current. The `closed` column records these with 1 for an
observed closure and 0 for a licence still running at the cutoff. Median
observed licence life is 904 days.

Features are restricted to what was knowable on the first day of
issuance: licence type, conditional-approval flag, ward, community area,
police district, zip code, latitude, and longitude. Renewal count is
deliberately excluded, because more renewals means a longer life, so it
carries the outcome.

Two registries were tried before this one and rejected, because their
death records begin long after their birth records. The EIA-860
power-generator file lists commissioning dates back to 1891 but its
retirement records effectively start in 2001, and the FDIC
insured-institution register has establishment dates from 1782 and no
closures until 1970. Restricting either to the years with death records
leaves a population too young to have died. Chicago has logged issuance,
renewal, and cancellation in one system since 2002, so a licence issued
in 2004 has its whole life on file.

**Source.**

City of Chicago open data portal, "Business Licenses" dataset
(`r5kz-chrr`), pulled through the unauthenticated Socrata API:
https://data.cityofchicago.org/resource/r5kz-chrr.csv

**Terms of use.**

The portal publishes this dataset under the city's open data terms, which
permit redistribution with attribution and disclaim any warranty of
accuracy. Attribution is the Source line above. The repository's MIT
licence covers the code, not this file. The data stays the city's, under
the city's terms. The file carries no business name and no street
address, only the administrative geography the portal publishes.

**Cleaning.**

Produced by `python scripts/run_prepare_chicago.py`, which documents
every step. The portal is queried from 2002-01-01, where its records
become complete. A business that opened before 2002 still appears, with
its first post-2002 renewal looking like an opening, so the row carries
the wrong start date and only the tail of a longer life. The volume
exposes them. Taken at face value the data shows 61,351 openings in 2002
against about 14,000 first issues in a normal year. The script groups the
issuance and renewal transactions by licence number and keeps only
licences whose earliest transaction is a genuine first issue. That
removed 79,690 rows, 23 percent of the pull, and guarantees every
remaining row is watched from its true start.

A second cut removes licence types that are temporary by construction,
the Special Event, Pop-Up, and Itinerant variants. That is 23,042
licences across 12 types with a median recorded life of four days. Their
short lives are intent, not failure. A temporary food licence was always
going to expire within days, while the handyman whose business folded
never meant it to end, and only the second kind belongs in a study of
business survival.

Each remaining licence starts at its first issue date, ends at the
cancellation date where one exists and otherwise at the latest expiry,
and an end date before the cutoff counts as an observed closure. No
values are altered.

**Why the file is committed rather than fetched on demand.**

The portal is live and changes daily, so re-running the preparation
script would return a different dataset and the numbers in the report
would no longer match. The committed snapshot is what keeps every figure
in the report checkable. The fitted models are the opposite case. They
are reproducible from this file plus the code, so they are regenerated
rather than stored.

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
