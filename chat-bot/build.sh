#!/usr/bin/env bash
# exit on error
set -o errexit

pip install --upgrade pip
pip install -r requirements.txt

# Don't load ML models during collectstatic
python manage.py collectstatic --no-input
python manage.py migrate

echo "Build completed successfully!"
