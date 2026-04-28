"""python -m keiba_ai entrypoint — delegates to ingest job."""

import sys

from keiba_ai.jobs.ingest import cli_main

sys.exit(cli_main())
