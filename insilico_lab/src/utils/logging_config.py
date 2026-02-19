import logging
import os

def setup_logging(log_file='app.log', level=logging.INFO):
    """
    Setup basic logging configuration.
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file,
        filemode='a'
    )
    console = logging.StreamHandler()
    console.setLevel(level)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
