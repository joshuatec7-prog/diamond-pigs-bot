#!/bin/bash
# Start bot en agent tegelijk
python3 agent.py &
python3 diamond_bot.py
