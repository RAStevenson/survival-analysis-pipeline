For this dataset the preparation step did perform that exclusion. The
portal's history is complete only from 2002, so a licence already running
then shows its first post-2002 renewal as its recorded start, not its true
start, and nothing in the data labels these rows. Keeping only licences
whose earliest transaction is a genuine issue removed 79,690 of them, 23
percent of the pull. What exposed them was a spike of 61,351 supposed starts
dated 2002, against roughly 14,000 genuine first issues in a normal year. No
licence in this file starts before 2002. The rule and the counts are
recorded in `scripts/run_prepare_chicago.py`.

Licence types that are event-scoped by construction, the Special Event,
Pop-Up, and Itinerant variants, are excluded as well. That removed 23,042
licences across 12 types, permits that typically expire within their first
week. A permit like that expires on a date set when it is issued, so its
recorded lifespan measures the permit's term, not the business's survival.
Training on those rows teaches the model to recognize permit types, not to
predict failure.

Licence terms are visible in the curves below. The vertical drop in the
(other) curve at about two years, and the smaller steps at term multiples in
the others, are terms expiring. A business that does not renew exits at a
term boundary, so endings cluster there.
