# Use Submission Confirmation Over Entrypoint Drift Checks

The project now compares multiple Model Candidates, so comparing `main.py` to one training-side baseline would not prove that the Submission Artifact matches the selected Experiment Run. We keep `main.py` self-contained for platform reliability, remove training-candidate drift checks from `just check`, and rely on Promote to Submission plus Submission Confirmation before a Platform Run.
