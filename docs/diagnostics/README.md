# Diagnostics

Official-safe diagnostic reports live here. These reports summarize local
Training Set and Internal Validation Split behavior without adding external
weather images or treating manual review notes as authoritative labels.

Generated reports may depend on ignored Experiment Run artifacts under
`results/runs/`. Keep report inputs reproducible through documented `just`
recipes and keep bulky generated assets out of this directory unless they are
needed for durable review.
