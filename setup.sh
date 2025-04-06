#!/bin/bash
# Setup script for SHL Assessment Recommender

echo "Setting up SHL Assessment Recommender..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv/Scripts/activate
else
    # Linux/macOS
    source venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Install Playwright browsers
echo "Installing Playwright browsers..."
playwright install

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file..."
    echo "# Google Gemini API key - Get yours at https://makersuite.google.com/app/apikey" > .env
    echo "GOOGLE_API_KEY=your_api_key_here" >> .env
    echo "API_URL=http://localhost:8000" >> .env
    
    echo "⚠️ Please edit the .env file to add your Google API key!"
fi

# Generate sample data if not present
echo "Generating sample data..."
python -c "
import asyncio, os, json
from datetime import datetime

async def ensure_sample_data():
    # Create sample assessments if no data exists
    if not os.path.exists('shl_assessments.json'):
        print('Creating sample assessment data...')
        
        sample_assessments = [
            {
                'name': 'Verify Numerical Reasoning Test',
                'url': 'https://www.shl.com/solutions/products/verify-numerical-reasoning-test/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 25,
                'test_type': ['Ability & Aptitude', 'Numerical Reasoning']
            },
            {
                'name': 'Verify Verbal Reasoning Test',
                'url': 'https://www.shl.com/solutions/products/verify-verbal-reasoning-test/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 25,
                'test_type': ['Ability & Aptitude', 'Verbal Reasoning']
            },
            {
                'name': 'Verify General Ability Test',
                'url': 'https://www.shl.com/solutions/products/verify-general-ability-test/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 36,
                'test_type': ['Ability & Aptitude', 'Numerical Reasoning', 'Verbal Reasoning', 'Inductive Reasoning']
            },
            {
                'name': 'Work Strengths Questionnaire',
                'url': 'https://www.shl.com/solutions/products/work-strengths-questionnaire/',
                'remote_testing': True,
                'adaptive_irt': True,
                'duration': 25,
                'test_type': ['Personality & Behavior']
            },
            {
                'name': 'ADEPT-15 Personality Assessment',
                'url': 'https://www.shl.com/solutions/products/adept-15-personality-assessment/',
                'remote_testing': True,
                'adaptive_irt': True,
                'duration': 25,
                'test_type': ['Personality & Behavior']
            },
            {
                'name': 'Verify Coding Assessment',
                'url': 'https://www.shl.com/solutions/products/verify-coding-assessment/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 60,
                'test_type': ['Knowledge & Skills', 'Technical Skills']
            },
            {
                'name': 'Java Programming Skills Test',
                'url': 'https://www.shl.com/solutions/products/java-programming-skills-test/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 45,
                'test_type': ['Knowledge & Skills', 'Technical Skills']
            },
            {
                'name': 'Python Programming Skills Test',
                'url': 'https://www.shl.com/solutions/products/python-programming-skills-test/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 45,
                'test_type': ['Knowledge & Skills', 'Technical Skills']
            },
            {
                'name': 'JavaScript Programming Skills Test',
                'url': 'https://www.shl.com/solutions/products/javascript-programming-skills-test/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 40,
                'test_type': ['Knowledge & Skills', 'Technical Skills']
            },
            {
                'name': 'Situational Judgment Test',
                'url': 'https://www.shl.com/solutions/products/situational-judgment-test/',
                'remote_testing': True,
                'adaptive_irt': False,
                'duration': 30,
                'test_type': ['Situational Judgment', 'Competencies']
            }
        ]
        
        with open('shl_assessments.json', 'w', encoding='utf-8') as f:
            json.dump(sample_assessments, f, indent=2)
        
    # Create benchmark queries if not exists
    if not os.path.exists('benchmark_queries.json'):
        print('Creating benchmark queries...')
        
        benchmark_queries = [
            {
                'query': 'Frontend Developer with React experience',
                'relevant_assessments': [
                    'JavaScript Programming Skills Test',
                    'Verify Coding Assessment',
                    'Verify General Ability Test',
                    'Situational Judgment Test'
                ]
            },
            {
                'query': 'Data Scientist with machine learning experience',
                'relevant_assessments': [
                    'Python Programming Skills Test',
                    'Verify Numerical Reasoning Test',
                    'Verify General Ability Test'
                ]
            },
            {
                'query': 'Project Manager for software development team',
                'relevant_assessments': [
                    'Situational Judgment Test',
                    'ADEPT-15 Personality Assessment',
                    'Verify Verbal Reasoning Test'
                ]
            },
            {
                'query': 'Sales Executive for enterprise software',
                'relevant_assessments': [
                    'Work Strengths Questionnaire',
                    'ADEPT-15 Personality Assessment',
                    'Situational Judgment Test'
                ]
            },
            {
                'query': 'Customer Service Representative for call center',
                'relevant_assessments': [
                    'Work Strengths Questionnaire',
                    'Situational Judgment Test',
                    'Verify Verbal Reasoning Test'
                ]
            }
        ]
        
        with open('benchmark_queries.json', 'w', encoding='utf-8') as f:
            json.dump(benchmark_queries, f, indent=2)

asyncio.run(ensure_sample_data())
"

echo "Setup complete!"
echo ""
echo "To run the system:"
echo "  - Ensure you've set your Google API key in the .env file"
echo "  - Run the scraper: python main.py --scrape"
echo "  - Run both API and web app: python main.py --all"
echo "  - Access the web UI at: http://localhost:8501"
echo "  - Access the API docs at: http://localhost:8000/docs"
echo ""
echo "Happy recommending! 🚀" 