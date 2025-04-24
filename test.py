import joblib
mba = joblib.load("model/mba_rules.joblib")
print(mba.shape)
print(mba.head())



def train_mba_model():
    print("\U0001F6D2 Entrenando modelo de reglas de asociación (MBA)...")
    df = pd.read_csv(DATA_PATH)
    grouped = df.groupby('order_id')['product_id'].apply(list)

    te = TransactionEncoder()
    te_ary = te.fit(grouped).transform(grouped)
    df_tf = pd.DataFrame(te_ary, columns=te.columns_)

    frequent_itemsets = apriori(df_tf, min_support=0.0001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.0001)

    rules = rules[rules['antecedents'].apply(len) == 1]
    rules = rules[rules['consequents'].apply(len) == 1]
    rules['antecedents'] = rules['antecedents'].apply(lambda x: list(x)[0])
    rules['consequents'] = rules['consequents'].apply(lambda x: list(x)[0])

    mba_df = rules[['antecedents', 'consequents', 'confidence']]
    os.makedirs(MODEL_DIR, exist_ok=True)

    if mba_df.empty:
        mba_df = pd.DataFrame(columns=['antecedents', 'consequents', 'confidence'])

    joblib.dump(mba_df, MBA_PATH)
    print(f"\U0001F4C1 Reglas MBA guardadas en {MBA_PATH}")
