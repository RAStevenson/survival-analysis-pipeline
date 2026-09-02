Two generator settings matter for interpreting the results.
@val{generator.admin_censor_rate:.0%} of rows are administratively
retired independent of performance, which installs the censoring the
evaluation must handle correctly. And lifetimes are drawn log-normally
around what their drivers predict, at noise scale
@val{generator.log_time_sigma:g}, the irreducible noise that keeps even
the oracle in section @sec:results short of a perfect score.

The generator intentionally installs detectable structure in the data, so that recovering it is a
test rather than an interpretation. Each of the six feature families shifts
log survival time by a fixed amount, microstructure the most fragile and
value-carry the most durable. Among the four asset classes, crypto is built
to decay fastest, a shift of @val{generator.installed.asset_log_time_effect.crypto:.2f} on log time against the fx-majors
baseline. It is drawn for @val{generator.installed.asset_class_weights.crypto:.0%} of rows, so the class effect
has to be recovered from a minority slice.

Censoring concentrates among recently discovered strategies, which have had
the least time to fail before the observation cutoff, so the final fold's
test block is far more censored than the population overall.
