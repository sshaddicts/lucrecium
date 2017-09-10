#!/bin/bash

mvn verify -B

if [ $? -ne 0 ]; then
    echo build failed
    exit 1
    curl -X POST https://api.telegram.org/bot425164421:AAF28SLXHQbPJkQ2MJs8CeiSwYbRajWvd_Q/sendMessage -H 'content-type: application/x-www-form-urlencoded' -d 'chat_id=-1001143086398' -d 'text=Your build on CircleCi has failed!'
else
    echo good
    curl -X POST https://api.telegram.org/bot425164421:AAF28SLXHQbPJkQ2MJs8CeiSwYbRajWvd_Q/sendMessage -H 'content-type: application/x-www-form-urlencoded' -d 'chat_id=-1001143086398' -d 'text=Your build on CircleCi has succeded!.'
    exit 0
fi
