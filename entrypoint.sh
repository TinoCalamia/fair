#!/bin/sh

cd /home/app
pip3 install -r app/requirements.txt

# May as well do this too, while we're here.
python3 manage.py migrate

exec "$@"