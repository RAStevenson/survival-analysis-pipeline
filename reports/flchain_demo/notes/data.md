Each row is one subject in the Mayo Clinic serum free light chain study,
observed from their blood sample until death or the end of follow-up. Only
@val{dataset.event_rate:.0%} of subjects died during follow-up. The
features are: age at sampling, sex, the kappa and
lambda free light chain concentrations, serum creatinine, and the assay
decile group, the study's own one-to-ten grouping of the light-chain
result. It also includes a flag for a prior diagnosis of MGUS, a benign
precursor blood condition (monoclonal gammopathy of undetermined
significance).

The recorded start date is a year, with no month or day. This causes some complications when preparing the data.
Two subjects sampled in the same year cannot be put in time order and there are a limited number
of row start dates and groups. Too many folds causes these groups to be split 
arbitrarily with the same start date for each fold but different rows. 
But because rows came presorted by age, the split caused different folds to contain different
age groups without a true distribution. And because age is a major factor in survival, 
this caused some folds to perform much better than others. 
To prevent unintended ordering of folds, 
the preparation script shuffles within each year with a fixed seed. That makes the order genuinely arbitrary instead of the source file's hidden age sort. The pipeline also merges any requested folds that land on the same split date into one. The issues this caused are discussed further in the interpretation section.

Survival time starts at the blood sample for every subject by
construction, so no subject is first seen partway through the interval
being measured, and the left-truncation fault described above does not
arise in this dataset.
