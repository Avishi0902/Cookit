Smart Recipe AI

An ingredient-based recipe recommendation system built with Python, 
hybrid machine learning, and Groq's LLaMA 3.3 AI.

What it does
- Enter the ingredients you have and get ranked recipe suggestions
- AI-powered substitution for unknown or international ingredients
- Meal planner, similar recipe finder, and personal recipe book
- Evaluated across 18 structured test cases

Dataset
8,009 real-world recipes from Archanaskitchen.com via Kaggle  
77 cuisines | 10 diet types | 1,804 unique ingredients

Tech Stack
- Python
- Sentence Transformers (all-MiniLM-L6-v2)
- scikit-learn
- Groq API (LLaMA 3.3-70B)
- pandas, numpy, fuzzywuzzy

How to run

### 1. Install dependencies
pip install sentence-transformers scikit-learn fuzzywuzzy 
pandas numpy python-dotenv groq

### 2. Add your Groq API key
Create a .env file in the project folder:
GROQ_API_KEY=your_key_here

Get a free key at console.groq.com

### 3. Run
python recommender.py

Project Structure
Smart-Recipe-AI/
├── recommender.py       # Main ML engine
├── evaluate.py          # Evaluation script (18 test cases)
├── food_recipes.csv     # Dataset
├── .env                 # API key (not uploaded)
└── README.md


Authors
Avishi 
