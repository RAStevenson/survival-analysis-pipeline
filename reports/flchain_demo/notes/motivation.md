The serum free light chain cohort is one of the most widely taught
survival datasets. It ships with R's survival package, it appears in the
standard Python tutorials, and a reader who has studied survival analysis
likely arrives already knowing roughly what a model scores on it. That
makes it a useful public test. The pipeline's numbers can be checked
against reported ones, and where they differ, the difference has to be
explainable.

The data is also a hard test. Nearly three quarters of the subjects were
still alive when the study stopped watching, so a model here must learn
to predict lifetimes from lifetimes it mostly never saw end. How each of
the two models holds up under that much censoring is the run's main
finding, and the interpretation section reports it.
