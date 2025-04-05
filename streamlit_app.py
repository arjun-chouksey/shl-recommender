import streamlit as st
import os
import json

# Create sample data if needed
def ensure_sample_data():
    if not os.path.exists("sample_assessments.json"):
        print("Creating sample data...")
        sample_assessments = [
            {
                "name": "Verify Numerical Reasoning Test",
                "url": "https://example.com/verify-numerical",
                "remote_testing": True,
                "adaptive_irt": False,
                "duration": 25,
                "test_type": ["Numerical Reasoning", "Cognitive Ability"]
            },
            {
                "name": "Verify Verbal Reasoning Test",
                "url": "https://example.com/verify-verbal",
                "remote_testing": True,
                "adaptive_irt": False,
                "duration": 25,
                "test_type": ["Verbal Reasoning", "Cognitive Ability"]
            },
            {
                "name": "Verify General Ability Test",
                "url": "https://example.com/verify-ga",
                "remote_testing": True,
                "adaptive_irt": False,
                "duration": 36,
                "test_type": ["Numerical Reasoning", "Verbal Reasoning", "Inductive Reasoning", "Cognitive Ability"]
            },
            {
                "name": "Work Strengths Questionnaire",
                "url": "https://example.com/work-strengths",
                "remote_testing": True,
                "adaptive_irt": True,
                "duration": 25,
                "test_type": ["Personality Assessment"]
            },
            {
                "name": "ADEPT-15 Personality Assessment",
                "url": "https://example.com/adept15",
                "remote_testing": True,
                "adaptive_irt": True,
                "duration": 25,
                "test_type": ["Personality Assessment"]
            }
        ]
        
        with open("sample_assessments.json", "w") as f:
            json.dump(sample_assessments, f, indent=2)
        
        print("Sample data created")

# Make sure sample data exists
ensure_sample_data()

# Import and run the main app
from app import main
main() 