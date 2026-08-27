Each row is one subject in the Mayo Clinic serum free light chain study,
observed from their blood sample until death or the end of follow-up. The
features are the intake measurements: age at sampling, sex, the kappa and
lambda free light chain concentrations, serum creatinine, and the assay
decile group, the study's own one-to-ten grouping of the light-chain
result. A flag for a prior diagnosis of MGUS, a benign
precursor blood condition (monoclonal gammopathy of undetermined
significance), completes the set. Only
@val{dataset.event_rate:.0%} of subjects died during follow-up. For
everyone else, the data records how long they had lived so far, and
nothing more.

The recorded start date is a year, with no month or day. Two subjects
sampled in the same year cannot be put in time order. The preparation
script therefore shuffles within each year with a fixed seed. That makes
the order genuinely arbitrary instead of the source file's hidden age
sort. The limitations section carries the consequences for folds and
labels.

The measured clock starts at the blood sample for every subject by
construction, so no subject is first seen partway through the interval
being measured, and the left-truncation fault described above does not
arise in this dataset.
