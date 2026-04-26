# ============================================================
#  Smart Recipe AI  -  recommender.py
# ============================================================

import pandas as pd
import numpy as np
import re
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from fuzzywuzzy import process
from groq import Groq

# Load .env from the same folder as this script file
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ============================================================
# 1. LOAD AND CLEAN DATA
# ============================================================

CSV_PATH = "food_recipes.csv"

def load_data(path=CSV_PATH):
    df = pd.read_csv(path)

    # Drop completely empty columns
    df.drop(columns=[c for c in df.columns if df[c].isna().all()], inplace=True)

    # Fill missing text fields with placeholder
    text_cols = ['description', 'cuisine', 'course', 'diet',
                 'tags', 'ingredients', 'category']
    for col in text_cols:
        df[col] = df[col].fillna("unknown")

    # Lowercase, remove special characters, collapse whitespace
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z0-9\s\|]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    for col in text_cols + ['recipe_title']:
        df[col] = df[col].apply(clean)

    return df

# ============================================================
# 2. TIME PARSING
# ============================================================

def parse_minutes(text):
    """
    Converts time strings like '15 M', '1 H 30 M', '45' into
    total integer minutes.
    """
    text = str(text).upper()
    hours   = re.findall(r'(\d+)\s*H', text)
    minutes = re.findall(r'(\d+)\s*M', text)
    total = 0
    if hours:   total += int(hours[0]) * 60
    if minutes: total += int(minutes[0])
    if not hours and not minutes:
        nums = re.findall(r'\d+', text)
        total = int(nums[0]) if nums else 0
    return total

def add_time_columns(df):
    df['prep_minutes'] = df['prep_time'].apply(parse_minutes)
    df['cook_minutes'] = df['cook_time'].apply(parse_minutes)
    df['total_time']   = df['prep_minutes'] + df['cook_minutes']
    return df

# ============================================================
# 3. NUTRITION PARSING
# ============================================================

NUTRITION_PATTERNS = {
    'calories':    r'(\d+)\s*kcal',
    'total_fat':   r'Total Fat\s+([\d.]+)g',
    'protein':     r'Protein\s+([\d.]+)g',
    'carbs':       r'Total Carbohydrate\s+([\d.]+)g',
    'fiber':       r'Dietary Fiber\s+([\d.]+)g',
    'sodium':      r'Sodium\s+([\d.]+)mg',
    'sugar':       r'Total Sugars\s+([\d.]+)g',
    'cholesterol': r'Cholesterol\s+([\d.]+)mg',
}

def parse_nutrition(text):
    """
    Extracts structured nutritional values from a raw nutrition string.
    Returns a dict with numeric values (or None where data is missing).
    """
    result = {}
    for key, pattern in NUTRITION_PATTERNS.items():
        m = re.search(pattern, str(text), re.IGNORECASE)
        result[key] = float(m.group(1)) if m else None
    return result

def add_nutrition_columns(df):
    parsed = df['nutrition'].apply(parse_nutrition)
    nutrition_df = pd.DataFrame(parsed.tolist(), index=df.index)
    return pd.concat([df, nutrition_df], axis=1)

# ============================================================
# 4. BAYESIAN RATING
#
#  Problem: A recipe with 5.0 stars from 15 votes should not
#  rank above one with 4.9 stars from 10,000 votes.
#
#  Solution - Bayesian average:
#    bayesian_rating = (v * R + C * m) / (v + C)
#
#    v = number of votes for this recipe
#    R = raw rating of this recipe
#    C = mean vote count across all recipes
#    m = global mean rating across all recipes
# ============================================================

def add_bayesian_rating(df):
    C = df['vote_count'].mean()
    m = df['rating'].mean()
    df['bayesian_rating'] = (
        (df['vote_count'] * df['rating'] + C * m) / (df['vote_count'] + C)
    )
    return df

# ============================================================
# 5. INGREDIENT UTILITIES
# ============================================================

def get_ingredient_list(ingredients_str):
    """
    Splits a pipe-separated ingredient string into a clean list.
    Example: 'Egg|Tomato|Cheese' -> ['egg', 'tomato', 'cheese']
    """
    return [i.strip().lower() for i in str(ingredients_str).split('|') if i.strip()]

def ingredient_coverage(user_ingredients, recipe_ingredients_str):
    """
    Computes what fraction of a recipe's ingredients the user has.

    Returns:
        coverage_ratio  - matched / total  (float between 0 and 1)
        matched_count   - number of recipe ingredients the user has
        total_count     - total ingredients in the recipe
    """
    recipe_ings = get_ingredient_list(recipe_ingredients_str)
    if not recipe_ings:
        return 0.0, 0, 0

    user_ings = [u.lower().strip() for u in user_ingredients]
    matched = 0
    for r_ing in recipe_ings:
        for u_ing in user_ings:
            if u_ing in r_ing or r_ing in u_ing:
                matched += 1
                break

    ratio = matched / len(recipe_ings)
    return ratio, matched, len(recipe_ings)

def build_ingredient_vocabulary(df):
    """
    Collects all unique ingredient names from the dataset.
    Used by the fuzzy correction step to fix typos in user input.
    """
    vocab = set()
    for row in df['ingredients'].dropna():
        for ing in get_ingredient_list(row):
            vocab.add(ing)
    return list(vocab)

# ============================================================
# 6. INGREDIENT VALIDATION
#
#  Checks each user ingredient against the dataset vocabulary.
#  An ingredient is considered "known" if it exactly matches or
#  is a meaningful substring of any ingredient in the vocabulary.
#  Unknown ingredients are sent to the AI for substitution.
# ============================================================

def classify_ingredients(user_ingredients, vocab):
    """
    Splits user ingredients into two groups:
        known   - found in the dataset vocabulary
        unknown - not found at all in the dataset vocabulary

    Returns:
        known   : list of ingredients found in vocabulary
        unknown : list of ingredients not found in vocabulary
    """
    vocab_set = set(vocab)
    known   = []
    unknown = []

    for ing in user_ingredients:
        ing_clean = ing.lower().strip()

        # Check exact match first
        if ing_clean in vocab_set:
            known.append(ing_clean)
            continue

        # Partial match - only count if the matching portion is
        # at least 5 characters long to avoid short false matches
        # e.g. 'acai' should NOT match 'achari masala'
        partial_match = any(
            (ing_clean in v and len(ing_clean) >= 5) or
            (v in ing_clean and len(v) >= 5)
            for v in vocab_set
        )

        if partial_match:
            known.append(ing_clean)
        else:
            unknown.append(ing_clean)

    return known, unknown

# ============================================================
# 7. AI SUBSTITUTION  (Groq API - Free, no daily limits)
#
#  When one or more user ingredients are not in the dataset,
#  this function calls the Groq API (LLaMA 3 model) to suggest
#  the closest substitute ingredients that ARE in the dataset
#  vocabulary.
#
#  The AI receives:
#    - The unknown ingredient names
#    - A sample of 300 valid ingredients from the dataset
#  The AI returns:
#    - A JSON mapping of unknown -> best substitute from vocabulary
#    - A brief explanation of why each substitute was chosen
# ============================================================

def get_ai_substitutes(unknown_ingredients, vocab, api_key=None):
    """
    Calls Groq API (LLaMA 3) to find substitutes for unknown ingredients.

    Parameters:
        unknown_ingredients : list of ingredient strings not in dataset
        vocab               : full list of valid dataset ingredients
        api_key             : Groq API key (optional if set in .env)

    Returns:
        substitutions : dict  {unknown_ingredient: substitute_ingredient}
        explanations  : dict  {unknown_ingredient: reason_string}
        success       : bool  True if API call succeeded
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")

    if not key:
        print("  Note: No GROQ_API_KEY found. Skipping AI substitution.")
        print("  Add GROQ_API_KEY=your_key to your .env file to enable this.")
        return {}, {}, False

    # Send 300 shortest/most common ingredients to keep prompt compact
    vocab_sample = sorted(vocab, key=len)[:300]
    vocab_str    = ", ".join(vocab_sample)

    prompt = f"""You are helping a recipe recommendation system.
The following ingredients were entered by the user but are NOT present in the recipe dataset:
{", ".join(unknown_ingredients)}

The dataset contains these ingredients (sample of 300):
{vocab_str}

For each unknown ingredient, suggest the single best substitute ingredient from the dataset list above.
Choose substitutes that are culinarily similar - same texture, flavour profile, or cooking role.

Respond ONLY with a valid JSON object in exactly this format, nothing else:
{{
  "substitutions": {{
    "unknown_ingredient": "substitute_from_dataset"
  }},
  "explanations": {{
    "unknown_ingredient": "one sentence reason"
  }}
}}"""

    try:
        client   = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
        )
        text = response.choices[0].message.content.strip()

        # Strip markdown code fences if model wraps the JSON
        text = re.sub(r'^```json\s*', '', text)
        text = re.sub(r'\s*```$',     '', text)

        parsed        = json.loads(text)
        substitutions = parsed.get("substitutions", {})
        explanations  = parsed.get("explanations",  {})
        return substitutions, explanations, True

    except json.JSONDecodeError:
        print("  AI returned invalid JSON. Skipping substitution.")
        return {}, {}, False
    except Exception as e:
        print(f"  AI API error: {e}")
        return {}, {}, False

# ============================================================
# 8. COMBINED FEATURE TEXT
#
#  Each recipe is converted into a single string that captures
#  all its relevant attributes. Ingredients are repeated twice
#  to give them more weight in the embedding.
# ============================================================

def build_combined_text(df):
    def row_text(r):
        ings = ' '.join(get_ingredient_list(r['ingredients']))
        return (
            f"{r['recipe_title']} "
            f"{ings} {ings} "
            f"{r['cuisine']} {r['course']} {r['diet']} "
            f"{r['tags']} {r['description']}"
        )
    df['combined'] = df.apply(row_text, axis=1)
    return df

# ============================================================
# 9. SENTENCE EMBEDDINGS
# ============================================================

def get_embeddings(df, model):
    print("Encoding recipe embeddings. This may take a few minutes...")
    embeddings = model.encode(df['combined'].tolist(), show_progress_bar=True)
    print("Embeddings ready.")
    return embeddings

# ============================================================
# 10. TF-IDF MATRIX
# ============================================================

def get_tfidf(df):
    print("Building TF-IDF matrix...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df['combined'])
    print("TF-IDF matrix ready.")
    return {'vectorizer': vectorizer, 'matrix': matrix}

# ============================================================
# 11. FUZZY QUERY CORRECTION
#
#  Only runs on known or AI-substituted ingredients.
#  Never runs on raw unknown ingredients to prevent words like
#  'acai' from being corrupted into 'achari'.
# ============================================================

def correct_query(query, vocab, threshold=90):
    """
    Corrects a word only if fuzzy match confidence >= threshold.
    Threshold is set high (90) to avoid false corrections.
    """
    words = query.split()
    corrected = []
    for w in words:
        if len(w) < 4:
            corrected.append(w)
            continue
        match, score = process.extractOne(w, vocab)
        corrected.append(match if score >= threshold else w)
    return ' '.join(corrected)

# ============================================================
# 12. MAIN RECOMMEND FUNCTION
#
#  Hybrid scoring formula:
#    final_score = 0.45 * semantic_similarity
#                + 0.20 * tfidf_similarity
#                + 0.20 * ingredient_coverage
#                + 0.15 * bayesian_rating
# ============================================================

def recommend(
    query,
    model,
    embeddings,
    tfidf_cache,
    df,
    ingredient_vocab,
    max_time=None,
    diet=None,
    cuisine=None,
    course=None,
    top_n=5,
    semantic_weight=0.45,
    tfidf_weight=0.20,
    coverage_weight=0.20,
    rating_weight=0.15,
    api_key=None,
):
    """
    Main recommendation function with AI fallback for unknown ingredients.

    When all ingredients are known  -> standard hybrid search.
    When some ingredients are unknown -> AI suggests substitutes from the
    dataset vocabulary, then runs hybrid search with substituted query.
    When ALL ingredients are unknown -> AI substitutes everything and searches.
    If AI is unavailable and all ingredients are unknown -> returns empty
    result with a clear message instead of a wrong result.
    """

    # --- Parse user ingredients ---
    raw_ings = [i.strip() for i in re.split(r'[,\s]+', query) if i.strip()]

    # --- Classify ingredients: known vs unknown ---
    known_ings, unknown_ings = classify_ingredients(raw_ings, ingredient_vocab)

    substitutions = {}
    explanations  = {}
    ai_was_used   = False
    final_ings    = list(raw_ings)

    if unknown_ings:
        print(f"\nUnknown ingredients detected: {', '.join(unknown_ings)}")
        print("Calling AI to find substitutes from dataset vocabulary...")

        substitutions, explanations, ai_was_used = get_ai_substitutes(
            unknown_ings, ingredient_vocab, api_key=api_key
        )

        if ai_was_used and substitutions:
            print("\nAI substitutions:")
            for orig, sub in substitutions.items():
                reason = explanations.get(orig, "similar culinary profile")
                print(f"  '{orig}' -> '{sub}'  ({reason})")

            # Replace unknown ingredients with AI substitutes
            final_ings = []
            for ing in raw_ings:
                ing_clean = ing.lower().strip()
                if ing_clean in substitutions:
                    final_ings.append(substitutions[ing_clean])
                else:
                    final_ings.append(ing_clean)
        else:
            # AI unavailable or returned nothing
            # Drop unknowns - never pass them to fuzzy correction
            final_ings = known_ings if known_ings else []
            if not known_ings:
                print("  No known ingredients and AI unavailable.")
                print("  Cannot search without valid ingredients.")
                empty = df.head(0).copy()
                empty.attrs['ai_was_used']   = False
                empty.attrs['substitutions'] = {}
                empty.attrs['explanations']  = {}
                empty.attrs['known_ings']    = known_ings
                empty.attrs['unknown_ings']  = unknown_ings
                empty.attrs['final_ings']    = []
                return empty

    # --- Build search query ---
    search_query = ' '.join(final_ings).lower()
    search_query = re.sub(r'[^a-zA-Z\s]', ' ', search_query)

    # Skip fuzzy correction if there were unknowns that AI did not substitute
    # to prevent any remaining unknown words from being corrupted
    if unknown_ings and not ai_was_used:
        corrected_query = search_query
    else:
        corrected_query = correct_query(search_query, ingredient_vocab)

    print(f"\nOriginal input : {', '.join(raw_ings)}")
    if substitutions:
        print(f"After AI sub   : {', '.join(final_ings)}")
    print(f"Search query   : {corrected_query}")

    # --- Semantic similarity ---
    query_vec  = model.encode([corrected_query])
    sem_scores = cosine_similarity(query_vec, embeddings).flatten()

    # --- TF-IDF similarity ---
    vectorizer   = tfidf_cache['vectorizer']
    tfidf_matrix = tfidf_cache['matrix']
    query_tfidf  = vectorizer.transform([corrected_query])
    tfidf_scores = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # --- Ingredient coverage (uses final_ings after substitution) ---
    coverage_scores = df['ingredients'].apply(
        lambda x: ingredient_coverage(final_ings, x)[0]
    ).values

    # --- Normalise Bayesian rating to [0, 1] ---
    rating_scores = (df['bayesian_rating'] - df['bayesian_rating'].min()) / (
        df['bayesian_rating'].max() - df['bayesian_rating'].min() + 1e-9
    )

    # --- Compute final blended score ---
    df = df.copy()
    df['sem_score']      = sem_scores
    df['tfidf_score']    = tfidf_scores
    df['coverage_score'] = coverage_scores
    df['rating_norm']    = rating_scores.values

    df['final_score'] = (
        semantic_weight  * df['sem_score']      +
        tfidf_weight     * df['tfidf_score']    +
        coverage_weight  * df['coverage_score'] +
        rating_weight    * df['rating_norm']
    )

    # --- Apply hard filters ---
    mask = pd.Series([True] * len(df), index=df.index)

    if max_time:
        mask &= (df['total_time'] > 0) & (df['total_time'] <= max_time)
    if diet:
        mask &= df['diet'].str.contains(diet.lower(), case=False, na=False)
    if cuisine:
        mask &= df['cuisine'].str.contains(cuisine.lower(), case=False, na=False)
    if course:
        mask &= df['course'].str.contains(course.lower(), case=False, na=False)

    filtered = df[mask]

    if filtered.empty:
        print("No recipes match the given filters. Showing unfiltered results.")
        filtered = df

    results = filtered.sort_values('final_score', ascending=False).head(top_n)

    # --- Attach coverage details ---
    coverage_details = results['ingredients'].apply(
        lambda x: ingredient_coverage(final_ings, x)
    )
    results = results.copy()
    results['matched_ings'] = [c[1] for c in coverage_details]
    results['total_ings']   = [c[2] for c in coverage_details]
    results['coverage_pct'] = (
        results['matched_ings'] / results['total_ings'].replace(0, 1) * 100
    ).round(1)

    # --- Attach metadata ---
    results.attrs['ai_was_used']   = ai_was_used
    results.attrs['substitutions'] = substitutions
    results.attrs['explanations']  = explanations
    results.attrs['known_ings']    = known_ings
    results.attrs['unknown_ings']  = unknown_ings
    results.attrs['final_ings']    = final_ings

    return results

# ============================================================
# 13. DISPLAY RESULTS
# ============================================================

def display_results(results, user_ingredients):
    SEP = "-" * 65

    if results.attrs.get('ai_was_used') and results.attrs.get('substitutions'):
        print(f"\n{'=' * 65}")
        print("  AI INGREDIENT SUBSTITUTION APPLIED")
        print(f"{'=' * 65}")
        for orig, sub in results.attrs['substitutions'].items():
            reason = results.attrs['explanations'].get(orig, "similar culinary profile")
            print(f"  '{orig}' was substituted with '{sub}'")
            print(f"  Reason: {reason}")
        print()

    if results.attrs.get('unknown_ings') and not results.attrs.get('ai_was_used'):
        print(f"\n{'=' * 65}")
        print("  UNKNOWN INGREDIENTS - AI UNAVAILABLE")
        print(f"{'=' * 65}")
        print(f"  The following ingredients were not found in the dataset:")
        print(f"  {', '.join(results.attrs['unknown_ings'])}")
        print(f"  To get AI-powered substitutions, add your GROQ_API_KEY")
        print(f"  to the .env file in your project folder.")
        if not results.attrs.get('known_ings'):
            print(f"\n  No known ingredients to search with. Please try again")
            print(f"  with ingredients that are more commonly used in Indian cooking.")
            print(f"{'=' * 65}\n")
            return
        else:
            print(f"\n  Showing results based on known ingredients only:")
            print(f"  {', '.join(results.attrs['known_ings'])}")
        print()

    if results.empty:
        print("\n  No results to display.\n")
        return

    final_ings = results.attrs.get('final_ings', user_ingredients)

    print(f"\n{'=' * 65}")
    print(f"  TOP {len(results)} RECIPE RECOMMENDATIONS")
    print(f"{'=' * 65}")

    for rank, (_, row) in enumerate(results.iterrows(), 1):
        ing_list     = get_ingredient_list(row['ingredients'])
        user_has     = [i for i in ing_list
                        if any(u.lower() in i or i in u.lower()
                               for u in final_ings)]
        user_missing = [i for i in ing_list if i not in user_has]

        print(f"\n  #{rank}  {row['recipe_title'].title()}")
        print(SEP)
        print(f"  Cuisine  : {row['cuisine'].title()}")
        print(f"  Course   : {row['course'].title()}")
        print(f"  Diet     : {row['diet'].title()}")
        print(f"  Time     : {int(row['total_time'])} min "
              f"(prep {int(row['prep_minutes'])} + cook {int(row['cook_minutes'])})")
        print(f"  Rating   : {row['rating']:.2f} / 5.00  ({int(row['vote_count'])} votes)")
        print(f"  Score    : {row['final_score']:.3f}  |  "
              f"Coverage : {row['coverage_pct']}% "
              f"({row['matched_ings']} of {row['total_ings']} ingredients)")
        print(f"  Have     : {', '.join(user_has[:6]) or 'none matched'}")
        if user_missing:
            missing_str = ', '.join(user_missing[:5])
            if len(user_missing) > 5:
                missing_str += ' ...'
            print(f"  Missing  : {missing_str}")

        if pd.notna(row.get('calories')):
            print(f"  Nutrition: {int(row['calories'])} kcal  |  "
                  f"Protein {row.get('protein', 'N/A')}g  |  "
                  f"Carbs {row.get('carbs', 'N/A')}g  |  "
                  f"Fat {row.get('total_fat', 'N/A')}g")

        tags = [t.strip().title() for t in str(row['tags']).split('|')
                if t.strip() and t.strip().lower() != 'unknown'][:4]
        if tags:
            print(f"  Tags     : {', '.join(tags)}")

        print(f"  URL      : {row['url']}")

    print(f"\n{'=' * 65}\n")

# ============================================================
# 14. MEAL PLANNER
# ============================================================

def generate_meal_plan(user_ingredients, model, embeddings,
                        tfidf_cache, df, ingredient_vocab, days=5,
                        api_key=None):
    """
    Generates a Breakfast / Lunch / Dinner plan for N days.
    No recipe is repeated across the plan.
    """
    courses = {
        'Breakfast': 'breakfast',
        'Lunch':     'lunch',
        'Dinner':    'dinner',
    }
    used_titles = set()
    plan = {}

    for day in range(1, days + 1):
        plan[f"Day {day}"] = {}
        for meal_label, course_kw in courses.items():
            results = recommend(
                query=user_ingredients,
                model=model,
                embeddings=embeddings,
                tfidf_cache=tfidf_cache,
                df=df,
                ingredient_vocab=ingredient_vocab,
                course=course_kw,
                top_n=20,
                api_key=api_key,
            )
            for _, row in results.iterrows():
                title = row['recipe_title']
                if title not in used_titles:
                    plan[f"Day {day}"][meal_label] = title.title()
                    used_titles.add(title)
                    break
            else:
                plan[f"Day {day}"][meal_label] = "No unique recipe found"

    return plan

def display_meal_plan(plan):
    print(f"\n{'=' * 55}")
    print("  MEAL PLAN")
    print(f"{'=' * 55}")
    for day, meals in plan.items():
        print(f"\n  {day}")
        for meal, recipe in meals.items():
            print(f"    {meal:12s}: {recipe}")
    print(f"\n{'=' * 55}\n")

# ============================================================
# 15. SIMILAR RECIPE FINDER
# ============================================================

def find_similar(title_query, model, embeddings, df, top_n=5, api_key=None):
    """
    Given a recipe name, finds the most similar recipes using
    embedding cosine similarity.
    If no match is found, AI suggests the closest real recipe name.
    """
    mask = df['recipe_title'].str.contains(title_query.lower(), case=False, na=False)

    if not mask.any():
        print(f"No recipe found matching '{title_query}'.")

        key = api_key or os.environ.get("GROQ_API_KEY", "")
        if key:
            print("Asking AI to suggest the closest matching recipe...")

            sample_titles = df['recipe_title'].sample(
                min(100, len(df)), random_state=42
            ).tolist()
            titles_str = ", ".join([t.title() for t in sample_titles])

            prompt = f"""A user searched for a recipe using the keyword: "{title_query}"
No exact match was found in the dataset.

Here are some real recipe names from the dataset:
{titles_str}

Suggest the single most likely recipe the user was looking for from this list.
Also suggest 2-3 related recipes from the list if available.

Respond ONLY with valid JSON in exactly this format:
{{
  "best_match": "exact recipe name from the list",
  "related": ["recipe name 1", "recipe name 2"],
  "explanation": "one sentence explaining why this matches the user intent"
}}"""

            try:
                client   = Groq(api_key=key)
                response = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                )
                text = response.choices[0].message.content.strip()
                text = re.sub(r'^```json\s*', '', text)
                text = re.sub(r'\s*```$',     '', text)

                parsed      = json.loads(text)
                best_match  = parsed.get("best_match", "")
                related     = parsed.get("related", [])
                explanation = parsed.get("explanation", "")

                print(f"\nAI Suggestion:")
                print(f"  Did you mean: '{best_match}'?")
                print(f"  Reason      : {explanation}")
                if related:
                    print(f"  Related     : {', '.join(related)}")

                if best_match:
                    mask2 = df['recipe_title'].str.contains(
                        best_match.lower().split()[0],
                        case=False, na=False
                    )
                    if mask2.any():
                        print(f"\nShowing results for '{best_match}':\n")
                        seed_idx = df[mask2].index[0]
                        seed_vec = embeddings[seed_idx].reshape(1, -1)
                        scores   = cosine_similarity(seed_vec, embeddings).flatten()
                        df2      = df.copy()
                        df2['sim_score'] = scores
                        results = df2[~mask2].sort_values(
                            'sim_score', ascending=False
                        ).head(top_n)
                        return results[['recipe_title', 'cuisine', 'course',
                                        'diet', 'total_time',
                                        'bayesian_rating', 'sim_score', 'url']]

            except Exception as e:
                print(f"  AI suggestion failed: {e}")
        else:
            print("  Tip: Add GROQ_API_KEY to .env for AI-powered suggestions.")

        return pd.DataFrame()

    seed_idx  = df[mask].index[0]
    seed_vec  = embeddings[seed_idx].reshape(1, -1)
    scores    = cosine_similarity(seed_vec, embeddings).flatten()
    df = df.copy()
    df['sim_score'] = scores
    results = df[~mask].sort_values('sim_score', ascending=False).head(top_n)
    return results[['recipe_title', 'cuisine', 'course', 'diet',
                     'total_time', 'bayesian_rating', 'sim_score', 'url']]

# ============================================================
# 16. PERSONAL RECIPE BOOK
#
#  Lets the user add, view, and delete their own recipes.
#  Stored in a local JSON file (my_recipes.json) in the same
#  folder as the script. No database required.
# ============================================================

RECIPE_BOOK_PATH = Path(__file__).parent / "my_recipes.json"

def load_recipe_book():
    """Load saved recipes from JSON file. Returns list of recipe dicts."""
    if RECIPE_BOOK_PATH.exists():
        with open(RECIPE_BOOK_PATH, 'r') as f:
            return json.load(f)
    return []

def save_recipe_book(recipes):
    """Save recipe list to JSON file."""
    with open(RECIPE_BOOK_PATH, 'w') as f:
        json.dump(recipes, f, indent=2)

def add_recipe():
    """
    Interactively collects a recipe from the user and saves it
    to the personal recipe book.
    """
    SEP = "-" * 55
    print(f"\n{'=' * 55}")
    print("  ADD A NEW RECIPE")
    print(f"{'=' * 55}")

    recipe = {}

    recipe['title'] = input("Recipe name          : ").strip()
    if not recipe['title']:
        print("Recipe name cannot be empty. Cancelled.")
        return

    recipe['ingredients'] = input("Ingredients          : ").strip()
    recipe['cuisine']     = input("Cuisine              : ").strip() or "Not specified"
    recipe['course']      = input("Course (e.g. dinner) : ").strip() or "Not specified"
    recipe['diet']        = input("Diet (e.g. vegetarian): ").strip() or "Not specified"
    recipe['prep_time']   = input("Prep time (e.g. 15 min): ").strip() or "Not specified"
    recipe['cook_time']   = input("Cook time (e.g. 30 min): ").strip() or "Not specified"
    recipe['servings']    = input("Servings             : ").strip() or "Not specified"
    recipe['difficulty']  = input("Difficulty (Easy/Medium/Hard): ").strip() or "Not specified"

    print("\nEnter instructions (type each step and press Enter).")
    print("Type 'done' when finished.\n")
    steps = []
    step_num = 1
    while True:
        step = input(f"Step {step_num}: ").strip()
        if step.lower() == 'done':
            break
        if step:
            steps.append(f"Step {step_num}: {step}")
            step_num += 1

    recipe['instructions'] = "\n".join(steps) if steps else "Not provided"
    recipe['notes']        = input("\nAny notes or tips    : ").strip() or ""

    # Auto-assign ID and timestamp
    import datetime
    recipe['id']        = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    recipe['added_on']  = datetime.datetime.now().strftime("%d %b %Y, %I:%M %p")

    recipes = load_recipe_book()
    recipes.append(recipe)
    save_recipe_book(recipes)

    print(f"\n{'=' * 55}")
    print(f"  Recipe '{recipe['title']}' saved successfully.")
    print(f"  Total recipes in your book: {len(recipes)}")
    print(f"{'=' * 55}\n")

def view_recipes():
    """
    Display all saved recipes in a structured, readable format.
    Options to view all, view one in detail, or delete one.
    """
    recipes = load_recipe_book()
    SEP = "-" * 55

    if not recipes:
        print(f"\n{'=' * 55}")
        print("  Your recipe book is empty.")
        print("  Use option 5 to add your first recipe.")
        print(f"{'=' * 55}\n")
        return

    while True:
        print(f"\n{'=' * 55}")
        print(f"  YOUR RECIPE BOOK  ({len(recipes)} recipes)")
        print(f"{'=' * 55}")

        for i, r in enumerate(recipes, 1):
            cuisine = r.get('cuisine', 'N/A')
            course  = r.get('course',  'N/A')
            diet    = r.get('diet',    'N/A')
            print(f"  {i:2}. {r['title'].title()}")
            print(f"      {cuisine} | {course} | {diet} | Added: {r.get('added_on','N/A')}")

        print(f"\n{'=' * 55}")
        print("  Options:")
        print("    Enter a recipe number to view full details")
        print("    Type 'd <number>' to delete a recipe  (e.g. d 2)")
        print("    Press Enter to go back")
        print(f"{'=' * 55}")

        choice = input("\nYour choice: ").strip()

        if choice == "":
            break

        # Delete option
        if choice.lower().startswith('d '):
            try:
                idx = int(choice.split()[1]) - 1
                if 0 <= idx < len(recipes):
                    deleted_title = recipes[idx]['title']
                    recipes.pop(idx)
                    save_recipe_book(recipes)
                    print(f"\n  '{deleted_title}' deleted from your recipe book.")
                    if not recipes:
                        print("  Your recipe book is now empty.")
                        break
                else:
                    print("  Invalid number. Try again.")
            except (ValueError, IndexError):
                print("  Invalid input. Use format: d 2")
            continue

        # View detail option
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(recipes):
                r = recipes[idx]
                print(f"\n{'=' * 55}")
                print(f"  {r['title'].upper()}")
                print(f"{'=' * 55}")
                print(f"  Cuisine    : {r.get('cuisine', 'N/A')}")
                print(f"  Course     : {r.get('course', 'N/A')}")
                print(f"  Diet       : {r.get('diet', 'N/A')}")
                print(f"  Prep Time  : {r.get('prep_time', 'N/A')}")
                print(f"  Cook Time  : {r.get('cook_time', 'N/A')}")
                print(f"  Servings   : {r.get('servings', 'N/A')}")
                print(f"  Difficulty : {r.get('difficulty', 'N/A')}")
                print(f"  Added on   : {r.get('added_on', 'N/A')}")
                print(f"\n  Ingredients:")
                for ing in r.get('ingredients', '').split(','):
                    if ing.strip():
                        print(f"    - {ing.strip().title()}")
                print(f"\n  Instructions:")
                for line in r.get('instructions', 'Not provided').split('\n'):
                    if line.strip():
                        print(f"    {line}")
                if r.get('notes'):
                    print(f"\n  Notes: {r['notes']}")
                print(f"\n{'=' * 55}")
                input("  Press Enter to go back to recipe list...")
            else:
                print("  Invalid number. Try again.")
        except ValueError:
            print("  Invalid input. Enter a number or 'd <number>'.")

# ============================================================
# 17. BOOTSTRAP
# ============================================================

def bootstrap(csv_path=CSV_PATH):
    """
    Loads and prepares everything needed by the recommender.
    Call this once at startup.
    """
    print("Loading dataset...")
    df = load_data(csv_path)
    df = add_time_columns(df)
    df = add_nutrition_columns(df)
    df = add_bayesian_rating(df)
    df = build_combined_text(df)
    df = df.reset_index(drop=True)
    print(f"{len(df)} recipes loaded.")

    print("\nBuilding ingredient vocabulary...")
    ingredient_vocab = build_ingredient_vocabulary(df)
    print(f"{len(ingredient_vocab)} unique ingredients found.")

    print("\nLoading sentence-transformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings  = get_embeddings(df, model)
    tfidf_cache = get_tfidf(df)

    print("\nSystem ready.\n")
    return df, model, embeddings, tfidf_cache, ingredient_vocab

# ============================================================
# 17. MAIN - Interactive CLI
# ============================================================

if __name__ == "__main__":
    API_KEY = os.environ.get("GROQ_API_KEY", "")

    df, model, embeddings, tfidf_cache, vocab = bootstrap()

    print("=" * 55)
    print("  SMART RECIPE AI")
    print("=" * 55)
    if API_KEY:
        print("  AI substitution : ENABLED  (Groq - LLaMA 3)")
    else:
        print("  AI substitution : DISABLED")
        print("  Add GROQ_API_KEY to your .env file to enable it.")
    print("=" * 55)

    while True:
        print("\nOptions:")
        print("  1. Search by ingredients")
        print("  2. Generate meal plan")
        print("  3. Find similar recipes")
        print("  4. My Recipe Book")
        print("  5. Add a recipe")
        print("  6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == "1":
            query    = input("Ingredients (comma-separated): ").strip()
            max_time = input("Max cooking time in minutes (press Enter to skip): ").strip()
            diet     = input("Diet preference e.g. vegetarian, vegan (press Enter to skip): ").strip()
            cuisine  = input("Cuisine e.g. indian, italian (press Enter to skip): ").strip()
            course   = input("Course e.g. lunch, dinner, snack (press Enter to skip): ").strip()

            results = recommend(
                query=query,
                model=model,
                embeddings=embeddings,
                tfidf_cache=tfidf_cache,
                df=df,
                ingredient_vocab=vocab,
                max_time=int(max_time) if max_time else None,
                diet=diet or None,
                cuisine=cuisine or None,
                course=course or None,
                top_n=5,
                api_key=API_KEY or None,
            )
            user_ings = [i.strip() for i in re.split(r'[,\s]+', query) if i.strip()]
            display_results(results, user_ings)

        elif choice == "2":
            query = input("Ingredients (comma-separated): ").strip()
            days  = input("Number of days (default 5): ").strip()
            plan  = generate_meal_plan(
                user_ingredients=query,
                model=model,
                embeddings=embeddings,
                tfidf_cache=tfidf_cache,
                df=df,
                ingredient_vocab=vocab,
                days=int(days) if days else 5,
                api_key=API_KEY or None,
            )
            display_meal_plan(plan)

        elif choice == "3":
            title   = input("Enter a recipe name or keyword: ").strip()
            similar = find_similar(title, model, embeddings, df,
                                   api_key=API_KEY or None)
            if not similar.empty:
                print("\nSimilar recipes:\n")
                print(similar[['recipe_title', 'cuisine', 'course',
                                'total_time', 'bayesian_rating']].to_string(index=False))

        elif choice == "4":
            view_recipes()

        elif choice == "5":
            add_recipe()

        elif choice == "6":
            print("\nExiting.\n")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")