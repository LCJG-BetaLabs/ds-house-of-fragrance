# Databricks notebook source
import sys
import logging

_configured = False
_logger_name = "spe"


def _configure_logger():
    global _configured
    logger = logging.getLogger(_logger_name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)-7s %(message)s")
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    _configured = True


def get_logger() -> logging.Logger:
    if not _configured:
        _configure_logger()
    return logging.getLogger(_logger_name)
