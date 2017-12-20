#!/usr/bin/env python3
import sys
import logging
import yaml
import attrdict
from dataset import BabiDatasetLoader
from dnc import DNC
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CONFIG_FILE = 'config.yml'
# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
log = logging.getLogger("main")
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(handler)
# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Here we go...
    log.debug('Loading configuration')

    # Step 01: Load configuration
    try:
        with open(CONFIG_FILE, 'r') as fp:
            config = attrdict.AttrDict(yaml.load(fp))
    except IOError:
        log.error('Could not load configuration file: {}'.format(CONFIG_FILE))
        sys.exit(1)

    # Step 02: Load the training and testing data
    try:
        data = BabiDatasetLoader.load(
            config.data.cache_dir,
            config.data.data_dir,
            config.dataset
        )
        if not data:
            log.error('Could not load or reprocess the data. Aborting')
            sys.exit(1)
    except IOError as exc:
        log.error('Failed to load the bAbI data set')
        print(exc)
        sys.exit(1)

    # Step 03: Train the model
#    dnc = DNC(data, config.model)
