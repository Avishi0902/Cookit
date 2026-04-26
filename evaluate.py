# ============================================================
#  Smart Recipe AI  -  evaluate.py
#  Authors: Avishi & Nancy
#  Minor Project - 3rd Year B.Tech
#
#  Test case categories:
#
#  VALID   - all ingredients confirmed present in dataset
#            Expected: high coverage, no AI triggered
#
#  INVALID - all ingredients confirmed NOT in dataset
#            Expected: AI substitution triggered for all
#
#  MIXED   - some known, some unknown (confirmed)
#            Expected: AI substitutes only the unknown ones
#
#  TYPO    - misspelled ingredients (some correctable by fuzzy,
#            some so wrong they go to AI)
#            Expected: fuzzy corrects mild typos, AI handles rest
#
#  Each test records: coverage, hit rate, score, rating,
#  AI usage, substitutions made, query time.
#  All results are saved to evaluation_results.csv for report.
# ============================================================

import pandas as pd
import numpy as np
import re
import time
from recommender import (
    bootstrap,
    recommend,
    classify_ingredients,
    ingredient_coverage,
)

# ============================================================
# TEST CASES
# All ingredients verified against the actual dataset.
# ============================================================

TEST_CASES = [

    # ----------------------------------------------------------
    # CATEGORY 1: VALID
    # All ingredients confirmed EXACT or PARTIAL match in dataset.
    # These test the core ML recommendation quality.
    # ----------------------------------------------------------
    {
        "id"       : "V1",
        "category" : "VALID",
        "name"     : "Basic Indian vegetarian",
        "query"    : "tomato, onion, garlic, ginger",
        "max_time" : None,
        "diet"     : "vegetarian",
        "cuisine"  : None,
        "course"   : None,
        "expected" : "High coverage, no AI, vegetarian recipes"
    },
    {
        "id"       : "V2",
        "category" : "VALID",
        "name"     : "Quick rice dish under 30 min",
        "query"    : "rice, mustard seeds, green chilli",
        "max_time" : 30,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "Results under 30 min, no AI triggered"
    },
    {
        "id"       : "V3",
        "category" : "VALID",
        "name"     : "Indian breakfast with filters",
        "query"    : "tomato, onion, mustard seeds, green chilli",
        "max_time" : None,
        "diet"     : "vegetarian",
        "cuisine"  : "indian",
        "course"   : "breakfast",
        "expected" : "Indian vegetarian breakfast, high coverage"
    },
    {
        "id"       : "V4",
        "category" : "VALID",
        "name"     : "Western salad ingredients",
        "query"    : "avocado, broccoli, bacon",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "Good results, all known, no AI"
    },
    {
        "id"       : "V5",
        "category" : "VALID",
        "name"     : "Single well-known ingredient",
        "query"    : "tomato",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "Many results, high semantic relevance"
    },

    # ----------------------------------------------------------
    # CATEGORY 2: INVALID
    # All ingredients confirmed NOT in dataset vocabulary.
    # These test the AI substitution pipeline end to end.
    # ----------------------------------------------------------
    {
        "id"       : "I1",
        "category" : "INVALID",
        "name"     : "Japanese fermented ingredients",
        "query"    : "miso, tempeh, edamame",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "AI substitutes all 3, returns relevant recipes"
    },
    {
        "id"       : "I2",
        "category" : "INVALID",
        "name"     : "Western processed meats",
        "query"    : "pepperoni, salami, chorizo",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "AI substitutes all 3 with similar proteins"
    },
    {
        "id"       : "I3",
        "category" : "INVALID",
        "name"     : "Single unknown ingredient",
        "query"    : "acai",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "AI substitutes acai with similar fruit"
    },
    {
        "id"       : "I4",
        "category" : "INVALID",
        "name"     : "Korean and hot sauce",
        "query"    : "gochujang, sriracha",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "AI substitutes with similar spicy condiments"
    },
    {
        "id"       : "I5",
        "category" : "INVALID",
        "name"     : "Unknown with course filter",
        "query"    : "prosciutto, dragonfruit",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : "dinner",
        "expected" : "AI substitutes both, dinner filter applied"
    },

    # ----------------------------------------------------------
    # CATEGORY 3: MIXED
    # Mix of confirmed known and confirmed unknown ingredients.
    # Tests that AI substitutes only the unknown ones.
    # ----------------------------------------------------------
    {
        "id"       : "M1",
        "category" : "MIXED",
        "name"     : "Indian base + unknown protein",
        "query"    : "tomato, onion, garlic, tempeh",
        "max_time" : None,
        "diet"     : "vegetarian",
        "cuisine"  : None,
        "course"   : None,
        "expected" : "tomato/onion/garlic known, tempeh unknown, AI substitutes tempeh"
    },
    {
        "id"       : "M2",
        "category" : "MIXED",
        "name"     : "Rice dish + unknown sauce",
        "query"    : "rice, mustard seeds, gochujang",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : "indian",
        "course"   : None,
        "expected" : "rice/mustard known, gochujang unknown, AI substitutes"
    },
    {
        "id"       : "M3",
        "category" : "MIXED",
        "name"     : "Western known + unknown",
        "query"    : "broccoli, avocado, miso",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "broccoli/avocado known, miso unknown, AI substitutes miso"
    },
    {
        "id"       : "M4",
        "category" : "MIXED",
        "name"     : "Mostly known one unknown",
        "query"    : "tomato, onion, garlic, ginger, acai",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "4 known, acai unknown, AI substitutes only acai"
    },
    {
        "id"       : "M5",
        "category" : "MIXED",
        "name"     : "Breakfast mixed",
        "query"    : "green chilli, onion, flaxseeds",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : "breakfast",
        "expected" : "green chilli/onion known, flaxseeds unknown, AI substitutes"
    },

    # ----------------------------------------------------------
    # CATEGORY 4: TYPO
    # Misspelled ingredient names.
    # Mild typos (tomatoe, potatoe) are handled by fuzzy correction.
    # Severe typos (eggg, garic) are unknown and go to AI.
    # Tests the full correction pipeline.
    # ----------------------------------------------------------
    {
        "id"       : "T1",
        "category" : "TYPO",
        "name"     : "Mild typos - correctable by fuzzy",
        "query"    : "tomatoe, potatoe",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "Fuzzy corrects tomatoe->tomato, potatoe->potato, no AI needed"
    },
    {
        "id"       : "T2",
        "category" : "TYPO",
        "name"     : "Severe typos - go to AI",
        "query"    : "eggg, garic, onoin",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "All 3 unknown, AI substitutes with egg/garlic/onion equivalents"
    },
    {
        "id"       : "T3",
        "category" : "TYPO",
        "name"     : "Mixed typos and valid",
        "query"    : "tomatoe, onion, eggg",
        "max_time" : None,
        "diet"     : None,
        "cuisine"  : None,
        "course"   : None,
        "expected" : "tomatoe partial match, onion exact, eggg unknown -> AI"
    },
]

# ============================================================
# METRIC FUNCTIONS
# ============================================================

def compute_mean_coverage(results, final_ings):
    if results.empty:
        return 0.0
    coverages = [
        ingredient_coverage(final_ings, row['ingredients'])[0]
        for _, row in results.iterrows()
    ]
    return round(np.mean(coverages) * 100, 2)

def compute_hit_rate(results):
    if results.empty:
        return 0.0
    return round((results['coverage_score'] > 0).mean() * 100, 2)

def compute_mean_score(results):
    if results.empty:
        return 0.0
    return round(results['final_score'].mean(), 4)

def compute_mean_rating(results):
    if results.empty:
        return 0.0
    return round(results['bayesian_rating'].mean(), 4)

# ============================================================
# RUN EVALUATION
# ============================================================

def run_evaluation(df, model, embeddings, tfidf_cache, vocab,
                   api_key=None, output_csv="evaluation_results.csv"):

    all_rows = []

    print("\n" + "=" * 70)
    print("  EVALUATION REPORT  -  Smart Recipe AI")
    print("=" * 70)

    for tc in TEST_CASES:
        print(f"\n[{tc['id']}] {tc['category']} - {tc['name']}")
        print(f"  Query    : {tc['query']}")
        print(f"  Expected : {tc['expected']}")

        raw_ings = [i.strip() for i in re.split(r'[,\s]+', tc['query']) if i.strip()]
        known, unknown = classify_ingredients(raw_ings, vocab)
        print(f"  Known    : {known if known else 'none'}")
        print(f"  Unknown  : {unknown if unknown else 'none'}")

        start = time.time()

        try:
            results = recommend(
                query=tc['query'],
                model=model,
                embeddings=embeddings,
                tfidf_cache=tfidf_cache,
                df=df,
                ingredient_vocab=vocab,
                max_time=tc['max_time'],
                diet=tc['diet'],
                cuisine=tc['cuisine'],
                course=tc['course'],
                top_n=5,
                api_key=api_key,
            )
            elapsed = round(time.time() - start, 2)

            final_ings    = results.attrs.get('final_ings', raw_ings)
            ai_used       = results.attrs.get('ai_was_used', False)
            substitutions = results.attrs.get('substitutions', {})

            avg_coverage = compute_mean_coverage(results, final_ings)
            hit_rate     = compute_hit_rate(results)
            avg_score    = compute_mean_score(results)
            avg_rating   = compute_mean_rating(results)
            n_results    = len(results)
            top_recipe   = results.iloc[0]['recipe_title'].title() if n_results > 0 else "None"

            print(f"  Results  : {n_results}")
            print(f"  Coverage : {avg_coverage}%")
            print(f"  Hit Rate : {hit_rate}%")
            print(f"  Avg Score: {avg_score}")
            print(f"  Avg Rating:{avg_rating}")
            print(f"  AI Used  : {ai_used}")
            if substitutions:
                for orig, sub in substitutions.items():
                    print(f"    '{orig}' -> '{sub}'")
            print(f"  Time(s)  : {elapsed}")
            print(f"  Top Result: {top_recipe}")

            all_rows.append({
                "ID"                  : tc['id'],
                "Category"            : tc['category'],
                "Test Name"           : tc['name'],
                "Query"               : tc['query'],
                "Known Ingredients"   : ", ".join(known) if known else "none",
                "Unknown Ingredients" : ", ".join(unknown) if unknown else "none",
                "AI Used"             : ai_used,
                "Substitutions"       : str(substitutions) if substitutions else "none",
                "Num Results"         : n_results,
                "Avg Coverage %"      : avg_coverage,
                "Hit Rate %"          : hit_rate,
                "Avg Final Score"     : avg_score,
                "Avg Bayesian Rating" : avg_rating,
                "Query Time (s)"      : elapsed,
                "Top Result"          : top_recipe,
                "Max Time Filter"     : tc['max_time'] if tc['max_time'] else "none",
                "Diet Filter"         : tc['diet'] if tc['diet'] else "none",
                "Cuisine Filter"      : tc['cuisine'] if tc['cuisine'] else "none",
                "Course Filter"       : tc['course'] if tc['course'] else "none",
                "Expected Behaviour"  : tc['expected'],
            })

        except Exception as e:
            elapsed = round(time.time() - start, 2)
            print(f"  ERROR: {e}")
            all_rows.append({
                "ID"                  : tc['id'],
                "Category"            : tc['category'],
                "Test Name"           : tc['name'],
                "Query"               : tc['query'],
                "Known Ingredients"   : ", ".join(known),
                "Unknown Ingredients" : ", ".join(unknown),
                "AI Used"             : False,
                "Substitutions"       : "error",
                "Num Results"         : 0,
                "Avg Coverage %"      : 0.0,
                "Hit Rate %"          : 0.0,
                "Avg Final Score"     : 0.0,
                "Avg Bayesian Rating" : 0.0,
                "Query Time (s)"      : elapsed,
                "Top Result"          : "error",
                "Max Time Filter"     : tc['max_time'],
                "Diet Filter"         : tc['diet'],
                "Cuisine Filter"      : tc['cuisine'],
                "Course Filter"       : tc['course'],
                "Expected Behaviour"  : tc['expected'],
            })

    # --------------------------------------------------------
    # SUMMARY BY CATEGORY
    # --------------------------------------------------------
    results_df = pd.DataFrame(all_rows)

    print("\n\n" + "=" * 70)
    print("  SUMMARY BY CATEGORY")
    print("=" * 70)

    for cat in ["VALID", "INVALID", "MIXED", "TYPO"]:
        subset = results_df[results_df["Category"] == cat]
        if subset.empty:
            continue
        print(f"\n  {cat} ({len(subset)} tests)")
        print(f"    Avg Coverage   : {subset['Avg Coverage %'].mean():.2f}%")
        print(f"    Avg Hit Rate   : {subset['Hit Rate %'].mean():.2f}%")
        print(f"    Avg Score      : {subset['Avg Final Score'].mean():.4f}")
        print(f"    Avg Rating     : {subset['Avg Bayesian Rating'].mean():.4f}")
        print(f"    Avg Query Time : {subset['Query Time (s)'].mean():.2f}s")
        print(f"    AI Triggered   : {subset['AI Used'].sum()}/{len(subset)} tests")

    # --------------------------------------------------------
    # OVERALL SUMMARY
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  OVERALL SUMMARY")
    print("=" * 70)
    print(f"  Total test cases        : {len(results_df)}")
    print(f"  Overall avg coverage    : {results_df['Avg Coverage %'].mean():.2f}%")
    print(f"  Overall avg hit rate    : {results_df['Hit Rate %'].mean():.2f}%")
    print(f"  Overall avg score       : {results_df['Avg Final Score'].mean():.4f}")
    print(f"  Tests where AI triggered: {results_df['AI Used'].sum()}/{len(results_df)}")
    print(f"  Avg query time          : {results_df['Query Time (s)'].mean():.2f}s")

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CATEGORY COMPARISON TABLE")
    print("=" * 70)
    print(f"  {'Category':<10} {'Tests':<7} {'Coverage':<12} "
          f"{'Hit Rate':<11} {'Avg Score':<12} {'AI Used'}")
    print(f"  {'-' * 60}")
    for cat in ["VALID", "INVALID", "MIXED", "TYPO"]:
        subset = results_df[results_df["Category"] == cat]
        if subset.empty:
            continue
        print(f"  {cat:<10} {len(subset):<7} "
              f"{subset['Avg Coverage %'].mean():<12.2f} "
              f"{subset['Hit Rate %'].mean():<11.2f} "
              f"{subset['Avg Final Score'].mean():<12.4f} "
              f"{subset['AI Used'].sum()}/{len(subset)}")

    # --------------------------------------------------------
    # SAVE CSV
    # --------------------------------------------------------
    results_df.to_csv(output_csv, index=False)
    print(f"\n  Full results saved to: {output_csv}")
    print("  Open this file in Excel to generate charts for your report.\n")
    print("=" * 70)

    return results_df

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    import os
    load_dotenv(dotenv_path=Path(__file__).parent / ".env")
    API_KEY = os.environ.get("GROQ_API_KEY", "")

    if not API_KEY:
        print("Note: GROQ_API_KEY not set.")
        print("INVALID and MIXED tests will run without AI substitution.")
        print("Add GROQ_API_KEY to your .env file for full results.\n")

    df, model, embeddings, tfidf_cache, vocab = bootstrap()
    run_evaluation(
        df, model, embeddings, tfidf_cache, vocab,
        api_key=API_KEY or None,
        output_csv="evaluation_results.csv"
    )