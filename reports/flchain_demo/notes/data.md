Each row is one subject in the Mayo Clinic serum free light chain study,
observed from their blood sample until death or the end of follow-up. Only
@val{dataset.event_rate:.0%} of subjects died during follow-up. The
features are: age at sampling, sex, the kappa and
lambda free light chain concentrations, serum creatinine, and the assay
decile group, the study's own one-to-ten grouping of the light-chain
result. It also includes a flag for a prior diagnosis of MGUS, a benign
precursor blood condition (monoclonal gammopathy of undetermined
significance).

The recorded start date is a year, with no month or day. This
complicates preparation. Two subjects sampled in the same year cannot be
put in time order, and there are only a few distinct start dates to cut
folds at. Asking for too many folds cut the same-year rows by their
position in the file, so two folds could share a split date and differ
only in which rows they held. Because the source file lists subjects in
age order within each year, that cut gave different folds different age
groups rather than a fair mix of ages. And because age is a major driver
of survival, some folds scored far better than others. To prevent that,
the preparation script shuffles within each year with a fixed seed. That
makes the order genuinely arbitrary instead of the source file's hidden
age sort. The pipeline also merges any requested folds that land on the
same split date into one. The interpretation section gives the numbers.

Survival time starts at the blood sample for every subject by
construction, so no subject is first seen partway through the interval
being measured, and the left-truncation fault described above does not
arise in this dataset.
