For this dataset the preparation step did perform that exclusion. Licences
already running when the portal's history begins enter at their first
recorded renewal rather than their true start; 79,690 of them, 23 percent of
the pull, were removed by keeping only licences whose earliest transaction is
a genuine issue. The tell was a spike of 61,351 supposed starts dated 2002,
against roughly 14,000 genuine first issues in a normal year. The rule and
the counts are recorded in `scripts/run_prepare_chicago.py`.

Licence types that are event-scoped by construction, the Special Event,
Pop-Up, and Itinerant variants, are excluded as well. That removed 23,042
licences across 12 types, permits that typically expire within their first
week. Those lives are short because the licence says so, and a lifespan that
is intent rather than failure is the wrong thing to ask a survival model
about.
