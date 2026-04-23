import os
import re
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)

warnings.filterwarnings("ignore")

DATA_PATH    = "data/processed/restmex_train.csv"
OUTPUT_DIR   = "artifacts/modelo_b_tfidf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5

STOPWORDS_ES = [
    "a","al","algo","algunas","algunos","ante","antes","como","con","contra",
    "cual","cuando","de","del","desde","donde","durante","e","el","ella","ellas",
    "ellos","en","entre","era","erais","eran","eras","eres","es","esa","esas",
    "ese","eso","esos","esta","estaba","estado","estais","estamos","estan",
    "estoy","esta","este","esto","estos","fue","fueron","fui","fuimos","ha",
    "han","has","hasta","hay","he","hemos","hubo","la","las","le","les","lo",
    "los","mas","me","mi","mia","mias","mio","mios","mis","mucho","muchos",
    "muy","ni","no","nos","nosotras","nosotros","nuestra","nuestras","nuestro",
    "nuestros","o","os","otra","otras","otro","otros","para","pero","poco",
    "por","porque","que","quien","quienes","se","sea","seais","seamos","sean",
    "seas","ser","si","sin","sobre","su","sus","suya","suyas","suyo","suyos",
    "también","tanto","te","teneis","tenemos","tener","tengo","ti","tiene",
    "tienen","toda","todas","todo","todos","tu","tus","tuya","tuyas","tuyo",
    "tuyos","u","un","una","unas","uno","unos","vosotras","vosotros","vuestra",
    "vuestras","vuestro","vuestros","y","ya","yo",
]


df = pd.read_csv(DATA_PATH)
print(f"   Filas totales : {len(df):,}")
print(f"\n   Distribución sentiment:\n{df['sentiment'].value_counts()}")
print(f"\n   Distribución label:\n{df['label'].value_counts().sort_index()}")

def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", " ", texto)
    texto = re.sub(r"[^a-záéíóúüñ\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    # Quitar stopwords manualmente
    palabras = [p for p in texto.split() if p not in STOPWORDS_ES]
    return " ".join(palabras)

df["texto_limpio"] = df["texto"].apply(limpiar_texto)
df = df[df["texto_limpio"].str.len() > 5].reset_index(drop=True)
print(f"   Filas tras limpieza: {len(df):,}")

X = df["texto_limpio"]
y_sentiment = df["sentiment"]
y_label     = df["label"]

(X_train, X_test,
 y_sent_train, y_sent_test,
 y_lab_train,  y_lab_test) = train_test_split(
    X, y_sentiment, y_label,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_sentiment
)
print(f"   Train: {len(X_train):,}  |  Test: {len(X_test):,}")

def crear_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=50_000,
            sublinear_tf=True,
            min_df=3,
            max_df=0.95,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",   # ← Balanceo de clases
            random_state=RANDOM_STATE,
        )),
    ])

param_grid = {
    "tfidf__ngram_range" : [(1, 1), (1, 2)],
    "tfidf__max_features": [30_000, 50_000],
    "clf__C"             : [0.1, 1.0, 10.0],
}

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

grid_sentiment = GridSearchCV(
    crear_pipeline(),
    param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)
grid_sentiment.fit(X_train, y_sent_train)
print(f"Mejores parámetros : {grid_sentiment.best_params_}")
print(f"Mejor F1 (CV)      : {grid_sentiment.best_score_:.4f}")
pipeline_sentiment = grid_sentiment.best_estimator_

grid_label = GridSearchCV(
    crear_pipeline(),
    param_grid,
    cv=cv,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
)
grid_label.fit(X_train, y_lab_train)
print(f"Mejores parámetros : {grid_label.best_params_}")
print(f"Mejor F1 (CV)      : {grid_label.best_score_:.4f}")
pipeline_label = grid_label.best_estimator_

def evaluar_modelo(nombre, pipeline, X_test, y_test, output_dir):
    print(f"EVALUACIÓN — {nombre}")

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="weighted")

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 Score : {f1:.4f}")
    print(f"\n{classification_report(y_test, y_pred)}")

    cm     = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax
    )
    ax.set_title(f"Matriz de Confusión — {nombre}", fontsize=14, pad=12)
    ax.set_xlabel("Predicho", fontsize=12)
    ax.set_ylabel("Real", fontsize=12)
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"confusion_{nombre.lower().replace(' ', '_')}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  Figura guardada en: {fig_path}")

    return {"accuracy": acc, "f1_weighted": f1}


metricas_sentiment = evaluar_modelo(
    "Sentiment", pipeline_sentiment, X_test, y_sent_test, OUTPUT_DIR
)
metricas_label = evaluar_modelo(
    "Label (Rating)", pipeline_label, X_test, y_lab_test, OUTPUT_DIR
)

with open(os.path.join(OUTPUT_DIR, "pipeline_sentiment.pkl"), "wb") as f:
    pickle.dump(pipeline_sentiment, f)
with open(os.path.join(OUTPUT_DIR, "pipeline_label.pkl"), "wb") as f:
    pickle.dump(pipeline_label, f)

results_sent  = pd.DataFrame(grid_sentiment.cv_results_)
results_label = pd.DataFrame(grid_label.cv_results_)
results_sent.to_csv(os.path.join(OUTPUT_DIR, "grid_results_sentiment.csv"), index=False)
results_label.to_csv(os.path.join(OUTPUT_DIR, "grid_results_label.csv"), index=False)

print(f"Modelos y resultados guardados en: {OUTPUT_DIR}/")


print(f"  Sentiment  →  Accuracy: {metricas_sentiment['accuracy']:.4f}  |  F1: {metricas_sentiment['f1_weighted']:.4f}")
print(f"  Label      →  Accuracy: {metricas_label['accuracy']:.4f}  |  F1: {metricas_label['f1_weighted']:.4f}")
print(f"\n  Optimizaciones aplicadas:")
print(f"Stopwords en español eliminadas")
print(f"class_weight='balanced'")
print(f"GridSearchCV con {CV_FOLDS} folds")
print(f"Mejor C sentiment : {grid_sentiment.best_params_['clf__C']}")
print(f"Mejor C label     : {grid_label.best_params_['clf__C']}")+

ejemplos = [
    "El hotel fue increíble, todo perfecto, volvería sin duda.",
    "Pésimo servicio, habitaciones sucias y personal maleducado.",
    "Las instalaciones estaban bien, nada especial pero correcto.",
]

for texto in ejemplos:
    texto_limpio = limpiar_texto(texto)
    sent = pipeline_sentiment.predict([texto_limpio])[0]
    lab  = pipeline_label.predict([texto_limpio])[0]
    print(f"\n  Reseña   : {texto[:60]}...")
    print(f"  Sentiment: {sent}  |  Rating predicho: {lab} estrellas")

print("\ncompletado.")
