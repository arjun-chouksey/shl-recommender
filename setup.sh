#!/bin/bash

# Create sample data if it doesn't exist
if [ ! -f "shl_assessments.json" ]; then
    echo "Creating sample assessments data..."
    python -c "from recommender import create_sample_data; create_sample_data()"
fi

echo "Setup completed!" 