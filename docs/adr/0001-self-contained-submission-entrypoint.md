# Keep the Submission Entrypoint Self-Contained

The momodel platform runs `main.py` as the submission prediction entrypoint, while local and cloud training need an importable model definition for faster iteration. We keep `main.py` self-contained for platform reliability, keep `weather_model.py` as the training-side source of truth for baseline labels, image size, and model structure, and use `just check` to detect drift between the two. This deliberately accepts duplicated submission code to avoid platform import/path surprises while keeping training code maintainable.

Update: the self-contained submission entrypoint remains, but the `weather_model.py` drift-check portion was superseded by ADR-0003 after the project adopted Model Candidates and Submission Confirmation.
