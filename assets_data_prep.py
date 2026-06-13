import pandas as pd
import numpy as np
import ast


def prepare_data(df):
    data = df.copy()

    # ===================================
    # Step 1: Remove leakage columns
    # ===================================
    leakage_cols = ['tconst', 'numVotes', 'BoxOffice', 'averageRating']
    data = data.drop(
        columns=[c for c in leakage_cols if c in data.columns],
        errors='ignore'
    )

    # ===================================
    # Step 2: Clean numeric columns
    # ===================================
    if 'startYear' in data.columns:
        data['startYear'] = pd.to_numeric(data['startYear'], errors='coerce')
        data.loc[~data['startYear'].between(1900, 2025), 'startYear'] = np.nan

    if 'runtimeMinutes' in data.columns:
        data['runtimeMinutes'] = pd.to_numeric(data['runtimeMinutes'], errors='coerce')
        data.loc[~data['runtimeMinutes'].between(1, 500), 'runtimeMinutes'] = np.nan

    if 'budget' in data.columns:
        data['budget'] = pd.to_numeric(data['budget'], errors='coerce')
        data.loc[~data['budget'].between(1, 1_000_000_000), 'budget'] = np.nan

    # ===================================
    # Step 3: Clean genres
    # ===================================
    if 'genres' in data.columns:
        def clean_genres(x):
            if pd.isna(x):
                return np.nan
            x = str(x).strip()
            if x.startswith('['):
                try:
                    parsed = ast.literal_eval(x)
                    if isinstance(parsed, list):
                        return ','.join(sorted(parsed))
                except:
                    pass
            cleaned = x.replace("'", "").replace('"', "").strip()
            if cleaned in ['\\N', 'None', 'nan', 'Not Found', '']:
                return np.nan
            return cleaned

        data['genres'] = data['genres'].apply(clean_genres)

    # ===================================
    # Step 4: Clean Language
    # ===================================
    if 'Language' in data.columns:
        data['Language'] = data['Language'].astype(str).str.strip()
        data['Language'] = data['Language'].replace(
            ['\\N', 'None', 'nan', 'Not Found', 'N', 'none'], np.nan
        )

    # ===================================
    # Step 5: Clean Country
    # ===================================
    if 'Country' in data.columns:
        data['Country'] = data['Country'].astype(str).str.strip()
        data['Country'] = data['Country'].str.replace(
            r"[\[\]'\n]", '', regex=True
        )
        data['Country'] = data['Country'].replace(
            ['\\N', 'None', 'nan', 'Not Found', 'N',
             'Earth', 'International', 'none'], np.nan
        )

    # ===================================
    # Step 6: Feature Engineering
    # ===================================

    if 'startYear' in data.columns:
        data['movie_age'] = 2025 - data['startYear']

    if 'genres' in data.columns:
        data['num_genres'] = data['genres'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else np.nan
        )

    if 'runtimeMinutes' in data.columns:
        data['runtime_category'] = pd.cut(
            data['runtimeMinutes'],
            bins=[0, 90, 120, np.inf],
            labels=['short', 'medium', 'long']
        )

    if 'Language' in data.columns:
        data['is_english'] = data['Language'].apply(
            lambda x: 1 if pd.notna(x) and 'English' in str(x) else 0
        )

    if 'runtimeMinutes' in data.columns and 'num_genres' in data.columns:
        data['runtime_per_genre'] = (
            data['runtimeMinutes'] / data['num_genres'].replace(0, np.nan)
        )

    if 'Country' in data.columns:
        data['is_usa'] = data['Country'].apply(
            lambda x: 1 if pd.notna(x) and
            any(c in str(x) for c in ['United States', 'USA', 'U.S.'])
            else 0
        )
        data['is_europe'] = data['Country'].apply(
            lambda x: 1 if pd.notna(x) and
            any(c in str(x) for c in ['France', 'Germany', 'Italy', 'Spain'])
            else 0
        )
        data['is_india'] = data['Country'].apply(
            lambda x: 1 if pd.notna(x) and 'India' in str(x) else 0
        )
        data['is_east_asian'] = data['Country'].apply(
            lambda x: 1 if pd.notna(x) and
            any(c in str(x) for c in ['Japan', 'South Korea', 'China'])
            else 0
        )

    top_genres = [
        'Drama', 'Comedy', 'Documentary', 'Horror',
        'Action', 'Romance', 'Thriller', 'Crime'
    ]
    for genre in top_genres:
        if 'genres' in data.columns:
            data[f'genre_{genre}'] = data['genres'].apply(
                lambda x: 1 if pd.notna(x) and genre in str(x) else 0
            )
        else:
            data[f'genre_{genre}'] = 0

    # ===================================
    # Step 7: Drop columns no longer needed
    # ===================================
    cols_to_remove = [
        'primaryTitle',
        'plot',
        'lead_actors_ids',
        'genres',
        'startYear',
        'budget',
        'Language',
        'Country'
    ]
    data = data.drop(
        columns=[c for c in cols_to_remove if c in data.columns],
        errors='ignore'
    )

    return data
