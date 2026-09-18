"""The fixed sampling spec behind the site's playground.

Kept apart from sample.py so publish_results.py can check published samples against it
without importing torch. The site says every prompt here is shown at these settings, and
the publish step refuses a samples file whose prompts or settings differ, so that sentence
is a checked fact rather than a promise. Changing either is a code change that shows up in
the diff, not a command-line flag.
"""

# Neutral, everyday openings: the point is to show how a model this size writes, not to
# find the prompts it happens to handle best.
PROMPTS = [
    "The reason the sky appears blue is",
    "The most important step in making bread is",
    "In 1905, Albert Einstein published",
    "A computer program is",
]

SETTINGS = {"temperature": 0.8, "top_k": 40, "max_new_tokens": 60, "seed": 1337}
