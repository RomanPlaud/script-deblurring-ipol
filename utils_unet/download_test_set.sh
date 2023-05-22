#!/bin/bash

FILE_ID="1lkBj2xzikfVxbQ_LGwwTep0POtW8_9lg"
DESTINATION="test_set.tar"

URL="https://drive.google.com/uc?export=download&id=${FILE_ID}"

wget -O "${DESTINATION}" --no-check-certificate "${URL}"

unzip test_set.tar

rm test_set.tar