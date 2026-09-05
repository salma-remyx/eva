#!/usr/bin/env python3
"""Run the perturbation analysis end to end: data → stats → audit.

Reads perturbations_config.yaml and writes results_*.csv to its configured output_dir.
"""

from data_perturbations import main as data_main
from stats_perturbations import main as stats_main
from transition_audits import main as audit_main

if __name__ == "__main__":
    data_main()
    stats_main()
    audit_main()
