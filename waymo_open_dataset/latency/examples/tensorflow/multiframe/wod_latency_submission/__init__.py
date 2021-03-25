"""Example __init__.py to wrap the wod_latency_submission module imports."""
from . import model

initialize_model = model.initialize_model
run_model = model.run_model
DATA_FIELDS = model.DATA_FIELDS
