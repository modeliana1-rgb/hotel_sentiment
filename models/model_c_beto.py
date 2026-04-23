
# ============================================================================
# 1. INSTALACIÓN DE DEPENDENCIAS
# ============================================================================
# Descomentar la siguiente línea si estás en Google Colab:
# !pip install transformers datasets accelerate scikit-learn matplotlib seaborn -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import warnings
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)
from torch.utils.data import Dataset

warnings.filterwarnings("ignore")

# ============================================================================
# 2. CARGA Y EXPLORACIÓN DEL DATASET
# ============================================================================
print("=" * 60)
print("CARGA Y EXPLORACIÓN DEL DATASET")
print("=" * 60)

# Si clonaste el repo en Colab:
# !git clone https://github.com/cristiancpr9211/hotel_sentiment.git
# import os; os.chdir('hotel_sentiment')

df = pd.read_csv("restmex_train.csv")

print(f"Forma del dataset: {df.shape}")
print(f"Columnas: {df.columns.tolist()}")
print(f"\nDistribución de sentimientos:")
print(df["sentiment"].value_counts())
print(f"\nLongitud promedio de texto: {df['texto'].str.len().mean():.0f} caracteres")
print(f"Longitud mediana de texto: {df['texto'].str.len().median():.0f} caracteres")

# Gráfico de distribución
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

colors = {"positivo": "#2ecc71", "neutro": "#f39c12", "negativo": "#e74c3c"}
sentiment_counts = df["sentiment"].value_counts()
axes[0].bar(
    sentiment_counts.index,
    sentiment_counts.values,
    color=[colors[s] for s in sentiment_counts.index],
)
axes[0].set_title("Distribución de Sentimientos", fontsize=14)
axes[0].set_ylabel("Cantidad")

for sent in ["positivo", "neutro", "negativo"]:
    subset = df[df["sentiment"] == sent]["texto"].str.len()
    axes[1].hist(subset, bins=50, alpha=0.5, label=sent, color=colors[sent])
axes[1].set_title("Distribución de Longitud de Texto", fontsize=14)
axes[1].set_xlabel("Longitud (caracteres)")
axes[1].set_xlim(0, 2000)
axes[1].legend()

plt.tight_layout()
plt.savefig("01_exploracion_datos.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: 01_exploracion_datos.png")

# ============================================================================
# 3. PREPROCESAMIENTO DE DATOS
# ============================================================================
print("\n" + "=" * 60)
print("PREPROCESAMIENTO DE DATOS")
print("=" * 60)

# --- CONFIGURACIÓN (ajusta según tus recursos) ---
SAMPLE_SIZE = 9000    # Total de muestras (3000 por clase)
TEST_SIZE = 0.2       # 20% para test
VAL_SIZE = 0.1        # 10% para validación
RANDOM_SEED = 42
# --------------------------------------------------

# Mapeo de sentimiento a etiqueta numérica
label_map = {"negativo": 0, "neutro": 1, "positivo": 2}
label_names = ["negativo", "neutro", "positivo"]

df["label_encoded"] = df["sentiment"].map(label_map)


def limpiar_texto(texto):
    """Limpieza básica del texto."""
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"\\x[0-9a-fA-F]{2}", "", texto)
    texto = re.sub(r"_x[0-9a-fA-F]{4}_", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


df["texto_limpio"] = df["texto"].apply(limpiar_texto)

# Muestreo estratificado balanceado
samples_per_class = SAMPLE_SIZE // 3
df_sample = (
    df.groupby("sentiment")
    .apply(lambda x: x.sample(n=min(samples_per_class, len(x)), random_state=RANDOM_SEED))
    .reset_index(drop=True)
)

print(f"Tamaño del subconjunto: {len(df_sample)}")
print(f"\nDistribución balanceada:")
print(df_sample["sentiment"].value_counts())

# División en train / validation / test
X = df_sample["texto_limpio"].values
y = df_sample["label_encoded"].values

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
)

val_ratio = VAL_SIZE / (1 - TEST_SIZE)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=val_ratio, random_state=RANDOM_SEED, stratify=y_train_val
)

print(f"\n--- Splits ---")
print(f"Train: {len(X_train)} muestras")
print(f"Validación: {len(X_val)} muestras")
print(f"Test: {len(X_test)} muestras")

# ============================================================================
# 4. TOKENIZACIÓN CON BETO
# ============================================================================
print("\n" + "=" * 60)
print("TOKENIZACIÓN CON BETO")
print("=" * 60)

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"
MAX_LENGTH = 256

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Modelo: {MODEL_NAME}")
print(f"Vocabulario: {tokenizer.vocab_size} tokens")
print(f"Max length configurado: {MAX_LENGTH}")

ejemplo = "El hotel es excelente, muy buena atención y limpieza."
tokens = tokenizer.tokenize(ejemplo)
print(f"\nEjemplo de tokenización:")
print(f"Texto: {ejemplo}")
print(f"Tokens: {tokens}")
print(f"Cantidad de tokens: {len(tokens)}")


class SentimentDataset(Dataset):
    """Dataset para análisis de sentimientos con BETO."""

    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


train_dataset = SentimentDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = SentimentDataset(X_val, y_val, tokenizer, MAX_LENGTH)
test_dataset = SentimentDataset(X_test, y_test, tokenizer, MAX_LENGTH)

print(f"\nTrain dataset: {len(train_dataset)} muestras")
print(f"Val dataset: {len(val_dataset)} muestras")
print(f"Test dataset: {len(test_dataset)} muestras")

# ============================================================================
# 5. CONFIGURACIÓN DEL MODELO
# ============================================================================
print("\n" + "=" * 60)
print("CONFIGURACIÓN DEL MODELO")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

NUM_LABELS = 3

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label={0: "negativo", 1: "neutro", 2: "positivo"},
    label2id={"negativo": 0, "neutro": 1, "positivo": 2},
)

model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nParámetros totales: {total_params:,}")
print(f"Parámetros entrenables: {trainable_params:,}")

# ============================================================================
# 6. ENTRENAMIENTO (FINE-TUNING)
# ============================================================================
print("\n" + "=" * 60)
print("ENTRENAMIENTO (FINE-TUNING)")
print("=" * 60)


def compute_metrics(eval_pred):
    """Calcula métricas de evaluación durante el entrenamiento."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted"
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


training_args = TrainingArguments(
    output_dir="./beto_sentiment_results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="./logs",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=RANDOM_SEED,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("Configuración del entrenamiento:")
print(f"  Epochs: {training_args.num_train_epochs}")
print(f"  Batch size (train): {training_args.per_device_train_batch_size}")
print(f"  Learning rate: {training_args.learning_rate}")
print(f"  Weight decay: {training_args.weight_decay}")
print(f"  Warmup ratio: {training_args.warmup_ratio}")
print(f"  FP16: {training_args.fp16}")

print("\nIniciando entrenamiento...")
print("=" * 50)

train_result = trainer.train()

print("\n" + "=" * 50)
print("Entrenamiento completado!")
print(f"Tiempo total: {train_result.metrics['train_runtime']:.1f} segundos")
print(f"Loss final: {train_result.metrics['train_loss']:.4f}")

# ============================================================================
# 7. EVALUACIÓN DEL MODELO
# ============================================================================
print("\n" + "=" * 60)
print("RESULTADOS EN EL CONJUNTO DE TEST")
print("=" * 60)

predictions = trainer.predict(test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)

print(f"\nAccuracy: {predictions.metrics['test_accuracy']:.4f}")
print(f"Precision (weighted): {predictions.metrics['test_precision']:.4f}")
print(f"Recall (weighted): {predictions.metrics['test_recall']:.4f}")
print(f"F1-Score (weighted): {predictions.metrics['test_f1']:.4f}")

print("\n" + "-" * 60)
print("Classification Report:")
print("-" * 60)
print(classification_report(y_test, preds, target_names=label_names, digits=4))

# Matriz de confusión
cm = confusion_matrix(y_test, preds)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=label_names, yticklabels=label_names, ax=axes[0],
)
axes[0].set_title("Matriz de Confusión (Absoluta)", fontsize=14)
axes[0].set_ylabel("Etiqueta Real")
axes[0].set_xlabel("Predicción")

cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(
    cm_norm, annot=True, fmt=".2%", cmap="Blues",
    xticklabels=label_names, yticklabels=label_names, ax=axes[1],
)
axes[1].set_title("Matriz de Confusión (Normalizada)", fontsize=14)
axes[1].set_ylabel("Etiqueta Real")
axes[1].set_xlabel("Predicción")

plt.tight_layout()
plt.savefig("02_matriz_confusion.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: 02_matriz_confusion.png")

# Evolución del entrenamiento
log_history = trainer.state.log_history

train_losses = [x["loss"] for x in log_history if "loss" in x and "eval_loss" not in x]
eval_entries = [x for x in log_history if "eval_loss" in x]
eval_losses = [x["eval_loss"] for x in eval_entries]
eval_f1s = [x["eval_f1"] for x in eval_entries]
eval_accs = [x["eval_accuracy"] for x in eval_entries]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(train_losses, label="Train Loss", color="#3498db")
axes[0].set_title("Training Loss", fontsize=14)
axes[0].set_xlabel("Steps (cada 50)")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

epochs = range(1, len(eval_losses) + 1)
axes[1].plot(epochs, eval_losses, "o-", label="Val Loss", color="#e74c3c")
axes[1].set_title("Validation Loss por Época", fontsize=14)
axes[1].set_xlabel("Época")
axes[1].set_ylabel("Loss")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, eval_f1s, "o-", label="F1 (weighted)", color="#2ecc71")
axes[2].plot(epochs, eval_accs, "s-", label="Accuracy", color="#9b59b6")
axes[2].set_title("Métricas de Validación por Época", fontsize=14)
axes[2].set_xlabel("Época")
axes[2].set_ylabel("Score")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("03_evolucion_entrenamiento.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: 03_evolucion_entrenamiento.png")

# ============================================================================
# 8. PREDICCIÓN SOBRE NUEVOS TEXTOS
# ============================================================================
print("\n" + "=" * 60)
print("PREDICCIONES SOBRE NUEVOS TEXTOS")
print("=" * 60)

sentiment_pipeline = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    top_k=None,
)

textos_prueba = [
    "El hotel fue increíble, la atención del personal fue excelente y las vistas al mar espectaculares.",
    "Pésimo servicio, la habitación estaba sucia y el personal fue muy grosero. Jamás regresaría.",
    "El restaurante está bien, la comida es aceptable pero nada extraordinario. Precios normales.",
    "¡Me encantó la experiencia! La mejor vacación de mi vida, todo perfecto.",
    "No me gustó nada. La comida era mala, el lugar estaba descuidado y era carísimo.",
    "El lugar es bonito pero la comida tardó mucho. Algunas cosas estaban bien, otras no tanto.",
]

for texto in textos_prueba:
    resultado = sentiment_pipeline(texto[:512])[0]
    resultado_sorted = sorted(resultado, key=lambda x: x["score"], reverse=True)
    prediccion = resultado_sorted[0]

    texto_display = f'"{texto[:80]}..."' if len(texto) > 80 else f'"{texto}"'
    print(f"\nTexto: {texto_display}")
    print(f"  → Predicción: {prediccion['label']} (confianza: {prediccion['score']:.2%})")
    print(f"    Probabilidades: ", end="")
    for r in resultado_sorted:
        print(f"{r['label']}: {r['score']:.2%}  ", end="")
    print()

# ============================================================================
# 9. ANÁLISIS DE ERRORES
# ============================================================================
print("\n" + "=" * 60)
print("ANÁLISIS DE ERRORES")
print("=" * 60)

errores_idx = np.where(preds != y_test)[0]
print(f"Total de errores: {len(errores_idx)} de {len(y_test)} ({len(errores_idx)/len(y_test):.1%})")
print(f"\nEjemplos de errores (primeros 10):")
print("-" * 70)

for i, idx in enumerate(errores_idx[:10]):
    texto = X_test[idx][:120]
    real = label_names[y_test[idx]]
    pred = label_names[preds[idx]]
    print(f'\n[{i+1}] Texto: "{texto}..."')
    print(f"     Real: {real} | Predicción: {pred}")

# Análisis de confianza
softmax = torch.nn.Softmax(dim=-1)
probs = softmax(torch.tensor(predictions.predictions)).numpy()
max_probs = np.max(probs, axis=1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(max_probs[preds == y_test], bins=30, alpha=0.7, label="Correctas", color="#2ecc71")
axes[0].hist(max_probs[preds != y_test], bins=30, alpha=0.7, label="Incorrectas", color="#e74c3c")
axes[0].set_title("Distribución de Confianza", fontsize=14)
axes[0].set_xlabel("Confianza (max probabilidad)")
axes[0].set_ylabel("Frecuencia")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

conf_por_clase = {}
for i, name in enumerate(label_names):
    mask = y_test == i
    conf_por_clase[name] = {
        "correctas": max_probs[(preds == y_test) & mask].mean()
        if ((preds == y_test) & mask).sum() > 0
        else 0,
        "incorrectas": max_probs[(preds != y_test) & mask].mean()
        if ((preds != y_test) & mask).sum() > 0
        else 0,
    }

x_pos = np.arange(len(label_names))
width = 0.35
axes[1].bar(
    x_pos - width / 2,
    [conf_por_clase[n]["correctas"] for n in label_names],
    width, label="Correctas", color="#2ecc71",
)
axes[1].bar(
    x_pos + width / 2,
    [conf_por_clase[n]["incorrectas"] for n in label_names],
    width, label="Incorrectas", color="#e74c3c",
)
axes[1].set_title("Confianza Promedio por Clase", fontsize=14)
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(label_names)
axes[1].set_ylabel("Confianza promedio")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("04_analisis_confianza.png", dpi=150, bbox_inches="tight")
plt.show()
print("Gráfico guardado: 04_analisis_confianza.png")

# ============================================================================
# 10. GUARDAR MODELO
# ============================================================================
OUTPUT_DIR = "./beto_sentiment_model"

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nModelo guardado en: {OUTPUT_DIR}")
print(f"\nPara cargar el modelo después:")
print(f"  model = AutoModelForSequenceClassification.from_pretrained('{OUTPUT_DIR}')")
print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")

print("\n" + "=" * 60)
print("¡PROCESO COMPLETADO!")
print("=" * 60)
