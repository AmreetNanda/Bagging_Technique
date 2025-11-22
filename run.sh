#!/bin/bash

# Set environment variables
export FLASK_APP=app.py
export FLASK_RUN_HOST=0.0.0.0
export FLASK_ENV=production

# Run Flask app
flask run
