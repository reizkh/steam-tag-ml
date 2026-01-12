#!/bin/bash
mkdir -p raw
curl -L -o ./raw/steam-reviews.zip\
  https://www.kaggle.com/api/v1/datasets/download/andrewmvd/steam-reviews
unzip ./raw/steam-reviews.zip -d ./raw
