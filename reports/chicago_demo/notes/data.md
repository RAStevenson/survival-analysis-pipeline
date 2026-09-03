For this dataset the preparation step did perform left-truncation exclusion. The
portal's history is complete only starting from 2002, so a license already running
then shows its first post-2002 renewal as its recorded start, not its true
start, and nothing in the data labels these rows. Keeping only licenses
whose earliest transaction is a genuine first issue removed 79,690 of them, 23
percent of the licenses downloaded from the portal. What exposed them was a spike of 61,351 supposed starts
dated 2002, against roughly 14,000 usual first issues in a normal year. No
license in this file starts before 2002. The rule and the counts are
recorded in `scripts/run_prepare_chicago.py`.

License types that are short or event-scoped by construction are excluded as well.
This includes the Special Event, Pop-Up, and Itinerant variants. That removed 23,042
licenses across 12 types. These permits are intended to be temporary. Their lack of
survival is obvious and expected and they do not contain useful general information
about which types of business have a greater chance to survive or not. Training on
those rows teaches the model to recognize short-term permit types, not to generalize.

License terms are visible in the curves below. The vertical drop at about
two years, sharpest in the Peddler License curve, and the smaller steps at
term multiples in the others, are terms expiring. A business that does not renew exits at a
term boundary, so endings cluster there.
