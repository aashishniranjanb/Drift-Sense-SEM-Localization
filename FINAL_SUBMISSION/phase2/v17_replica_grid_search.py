import os
import sys
import numpy as np
import pandas as pd

# Load the audit data from V17_REPLICA_35_AUDIT or pre-extracted features
# We want to optimize conditional Top-1 accuracy across all present cases
# Let's write a fast optimizer that tests weight combinations on candidates
