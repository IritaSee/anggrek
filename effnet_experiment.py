"""
effnet_experiment.py
=====================
Dedicated EfficientNetV2S training experiment — Coelogyne orchid classifier.

Perbedaan vs notebook utama:
  1. Dropout          : 0.3 (vs 0.4 / 0.2 di notebook)
  2. Fine-tune layers : hanya top-15 backbone (vs 40)
  3. Loss             : CategoricalCrossentropy(label_smoothing=0.1)
  4. Saving           : catatan training saja (JSON + CSV) — TIDAK simpan bobot model
  5. Stagnan guard    : jika val_accuracy tidak naik >1% dalam 5 epoch terakhir phase 2,
                        otomatis masuk Round 2 dengan class-weighted oversampling +
                        augmentasi lebih agresif

Cara pakai:
    cd /Users/iganarendra/anggrek
    source .venv/bin/activate
    python effnet_experiment.py
"""

import os
import json
import platform
import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')          # headless — simpan plot ke file tanpa display
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks, regularizers
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ── Apple Silicon Metal GPU ───────────────────────────────────────────────────
if platform.system() == "Darwin" and platform.machine() == "arm64":
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    print(f"GPU (Metal) : {[g.name for g in gpus]}")

print(f"TensorFlow  : {tf.__version__}")
print(f"Keras       : {keras.__version__}")

# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_DIR = './data'
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 8
VAL_SPLIT   = 0.2
SEED        = 42

CLASS_NAMES = ['Asperata', 'Dayana', 'Pandurata', 'Rumphii', 'Swaniana']
NUM_CLASSES = len(CLASS_NAMES)

EPOCHS_HEAD     = 15
EPOCHS_FINETUNE = 20
LR_HEAD         = 1e-3
LR_FINETUNE     = 1e-5

# ── Experiment-specific overrides ─────────────────────────────────────────────
EFF_DROPOUT         = 0.3    # lebih kecil dari default 0.4
EFF_FINETUNE_LAYERS = 15     # freeze semua kecuali 15 layer terakhir backbone
EFF_LABEL_SMOOTHING = 0.1    # regularisasi label

STAGNAN_THRESHOLD = 0.01     # improvement val_acc < 1% dalam window terakhir → stagnan
STAGNAN_WINDOW    = 5        # lihat 5 epoch terakhir phase 2

# ── Log / output directories ──────────────────────────────────────────────────
EXPERIMENT_NAME = "effnet_exp_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_BASE_DIR    = os.path.join('./logs', EXPERIMENT_NAME)
NOTES_FILE      = os.path.join(LOG_BASE_DIR, 'training_notes.json')

os.makedirs(LOG_BASE_DIR, exist_ok=True)
print(f"\nLog dir     : {LOG_BASE_DIR}")
print(f"Notes file  : {NOTES_FILE}")

# ── Data Pipeline — Standard augmentation ─────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale            = 1. / 255,
    validation_split   = VAL_SPLIT,
    rotation_range     = 30,
    width_shift_range  = 0.15,
    height_shift_range = 0.15,
    shear_range        = 0.1,
    zoom_range         = 0.2,
    horizontal_flip    = True,
    brightness_range   = [0.8, 1.2],
    fill_mode          = 'nearest',
)

val_datagen = ImageDataGenerator(
    rescale          = 1. / 255,
    validation_split = VAL_SPLIT,
)


def make_generators(datagen_train, datagen_val, seed=SEED):
    """Buat pasangan train/val generator dari direktori dataset."""
    tg = datagen_train.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='training',
        shuffle=True, seed=seed,
    )
    vg = datagen_val.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', subset='validation',
        shuffle=False, seed=seed,
    )
    return tg, vg


train_gen, val_gen = make_generators(train_datagen, val_datagen)

print(f"\nTrain samples : {train_gen.samples}")
print(f"Val samples   : {val_gen.samples}")
print(f"Class indices : {train_gen.class_indices}")

# ── Class weights — untuk oversampling efek inverse-frequency ─────────────────
try:
    from sklearn.utils.class_weight import compute_class_weight

    cw_values = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_gen.classes),
        y=train_gen.classes,
    )
    class_weight_dict = dict(enumerate(cw_values))
except ImportError:
    # Fallback manual jika sklearn tidak tersedia
    counts = np.bincount(train_gen.classes)
    total  = len(train_gen.classes)
    class_weight_dict = {i: total / (NUM_CLASSES * c) for i, c in enumerate(counts)}

print("\nClass weights (inverse-frequency, dipakai di Round 2):")
for idx, name in enumerate(CLASS_NAMES):
    print(f"  [{idx}] {name}: {class_weight_dict[idx]:.3f}")


# ── Model Builder ─────────────────────────────────────────────────────────────
def build_effnet_experiment(num_classes, img_size=IMG_SIZE, dropout=EFF_DROPOUT):
    """
    EfficientNetV2S dengan:
    - dropout uniform = EFF_DROPOUT (bukan 0.4/0.2)
    - backbone beku saat build (Phase 1)
    - Phase 2 buka top-EFF_FINETUNE_LAYERS layer dari luar fungsi ini
    """
    inputs = keras.Input(shape=(*img_size, 3), name='input_image')

    # Rescale [0,1] → [0,255] agar kompatibel dengan preprocessing internal EfficientNet
    x = layers.Rescaling(255.0, name='to_uint8_range')(inputs)

    backbone = EfficientNetV2S(
        input_shape=(*img_size, 3),
        include_top=False,
        weights='imagenet',
        include_preprocessing=True,   # normalisasi internal ke [-1, 1]
    )
    backbone.trainable = False

    x = backbone(x, training=False)   # (B, 7, 7, 1280) untuk input 224×224

    # Classification head — dropout uniform EFF_DROPOUT di kedua layer
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(name='bn_head')(x)
    x = layers.Dropout(dropout, name='dropout_head')(x)
    x = layers.Dense(256, activation='relu', name='dense_256',
                     kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(dropout, name='dropout_2')(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='predictions')(x)

    model = keras.Model(inputs, outputs, name='EfficientNetV2S_Experiment')
    return model, backbone


# ── Callbacks — tanpa ModelCheckpoint (tidak simpan bobot) ───────────────────
def get_experiment_callbacks(phase_name):
    """EarlyStopping + ReduceLR + TensorBoard + CSVLogger per fase."""
    csv_path = os.path.join(LOG_BASE_DIR, f'{phase_name}_history.csv')
    return [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-7, verbose=1,
        ),
        callbacks.TensorBoard(
            log_dir=os.path.join(LOG_BASE_DIR, phase_name),
            histogram_freq=0,
        ),
        callbacks.CSVLogger(csv_path, separator=',', append=False),
    ]


# ── Stagnan detector ──────────────────────────────────────────────────────────
def is_stagnan(history_obj, threshold=STAGNAN_THRESHOLD, window=STAGNAN_WINDOW):
    """
    Returns True jika val_accuracy hampir tidak bergerak dalam
    `window` epoch terakhir (improvement < threshold).
    """
    val_acc = history_obj.history.get('val_accuracy', [])
    if len(val_acc) < window:
        return False
    recent      = val_acc[-window:]
    improvement = max(recent) - min(recent)
    print(
        f"\nStagnan check — val_acc window {window} epochs: "
        f"min={min(recent):.4f}, max={max(recent):.4f}, "
        f"improvement={improvement:.4f} (threshold={threshold})"
    )
    return improvement < threshold


# ── Helpers ───────────────────────────────────────────────────────────────────
def history_to_dict(h):
    """Konversi Keras History ke dict yang JSON-serializable."""
    return {k: [float(v) for v in vals] for k, vals in h.history.items()}


def save_notes(notes: dict):
    with open(NOTES_FILE, 'w') as f:
        json.dump(notes, f, indent=2)
    print(f"\n✅  Catatan training disimpan  →  {NOTES_FILE}")


def plot_and_save(h_head, h_ft, title_suffix, filename):
    """Simpan kurva akurasi + loss ke PNG."""
    n1 = len(h_head.history['accuracy'])
    n2 = len(h_ft.history['accuracy'])
    x1 = range(1, n1 + 1)
    x2 = range(n1 + 1, n1 + n2 + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'EfficientNetV2S — {title_suffix}', fontweight='bold')

    # Accuracy
    ax = axes[0]
    ax.plot(x1, h_head.history['accuracy'],     'b-',  label='Train (head)')
    ax.plot(x1, h_head.history['val_accuracy'], 'b--', label='Val (head)')
    ax.plot(x2, h_ft.history['accuracy'],       'r-',  label='Train (finetune)')
    ax.plot(x2, h_ft.history['val_accuracy'],   'r--', label='Val (finetune)')
    ax.axvline(n1, color='gray', linestyle=':', alpha=0.6, label='FT start')
    ax.set_title('Accuracy')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(x1, h_head.history['loss'],     'b-',  label='Train (head)')
    ax.plot(x1, h_head.history['val_loss'], 'b--', label='Val (head)')
    ax.plot(x2, h_ft.history['loss'],       'r-',  label='Train (finetune)')
    ax.plot(x2, h_ft.history['val_loss'],   'r--', label='Val (finetune)')
    ax.axvline(n1, color='gray', linestyle=':', alpha=0.6, label='FT start')
    ax.set_title('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(LOG_BASE_DIR, filename)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot disimpan  →  {save_path}")


# ════════════════════════════════════════════════════════════════════════════════
#  ROUND 1 — Training normal
#  dropout=0.3 | fine-tune top-15 | label_smoothing=0.1
# ════════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("ROUND 1 — Phase 1 : Training Head (backbone beku)")
print("=" * 70)

effnet_model, effnet_backbone = build_effnet_experiment(NUM_CLASSES)

loss_fn = keras.losses.CategoricalCrossentropy(label_smoothing=EFF_LABEL_SMOOTHING)

effnet_model.compile(
    optimizer=optimizers.Adam(LR_HEAD),
    loss=loss_fn,
    metrics=['accuracy'],
)

print(f"Trainable params (head only) : {effnet_model.count_params():,}")

train_gen.reset(); val_gen.reset()
history_r1_head = effnet_model.fit(
    train_gen,
    epochs=EPOCHS_HEAD,
    validation_data=val_gen,
    callbacks=get_experiment_callbacks('r1_phase1_head'),
    verbose=1,
)
train_gen.reset(); val_gen.reset()

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print(f"ROUND 1 — Phase 2 : Fine-tune top {EFF_FINETUNE_LAYERS} layers backbone")
print("=" * 70)

effnet_backbone.trainable = True
freeze_until = len(effnet_backbone.layers) - EFF_FINETUNE_LAYERS
for layer in effnet_backbone.layers[:freeze_until]:
    layer.trainable = False

trainable_n = sum(1 for l in effnet_backbone.layers if l.trainable)
print(f"Total backbone layers      : {len(effnet_backbone.layers)}")
print(f"Frozen sampai layer index  : {freeze_until}")
print(f"Trainable backbone layers  : {trainable_n}")

effnet_model.compile(
    optimizer=optimizers.Adam(LR_FINETUNE),
    loss=loss_fn,
    metrics=['accuracy'],
)

train_gen.reset(); val_gen.reset()
history_r1_ft = effnet_model.fit(
    train_gen,
    epochs=EPOCHS_FINETUNE,
    validation_data=val_gen,
    callbacks=get_experiment_callbacks('r1_phase2_finetune'),
    verbose=1,
)
train_gen.reset(); val_gen.reset()

# ── Simpan catatan Round 1 ────────────────────────────────────────────────────
notes = {
    'experiment'  : EXPERIMENT_NAME,
    'timestamp'   : datetime.datetime.now().isoformat(),
    'config': {
        'dropout'         : EFF_DROPOUT,
        'finetune_layers' : EFF_FINETUNE_LAYERS,
        'label_smoothing' : EFF_LABEL_SMOOTHING,
        'epochs_head'     : EPOCHS_HEAD,
        'epochs_finetune' : EPOCHS_FINETUNE,
        'lr_head'         : LR_HEAD,
        'lr_finetune'     : LR_FINETUNE,
        'batch_size'      : BATCH_SIZE,
        'img_size'        : list(IMG_SIZE),
        'oversampling'    : False,
    },
    'round1': {
        'phase1_head'          : history_to_dict(history_r1_head),
        'phase2_finetune'      : history_to_dict(history_r1_ft),
        'best_val_acc_head'    : float(max(history_r1_head.history.get('val_accuracy', [0]))),
        'best_val_acc_finetune': float(max(history_r1_ft.history.get('val_accuracy', [0]))),
    },
}
save_notes(notes)
plot_and_save(
    history_r1_head, history_r1_ft,
    title_suffix='Round 1 (dropout=0.3 | top-15 finetune | label_smooth=0.1)',
    filename='round1_curve.png',
)

# ════════════════════════════════════════════════════════════════════════════════
#  Stagnan Detection → Round 2 dengan Oversampling
# ════════════════════════════════════════════════════════════════════════════════
stagnan_detected = is_stagnan(history_r1_ft)

if not stagnan_detected:
    print("\n✅  Training tidak stagnan — Round 2 tidak diperlukan.")
    print(f"   Best val_accuracy Round 1 : {notes['round1']['best_val_acc_finetune']:.4f}")

else:
    print("\n" + "!" * 70)
    print("⚠️   STAGNAN TERDETEKSI — Masuk Round 2")
    print("!" * 70)
    print("""
Strategi Round 2:
  1. class_weight     : inverse-frequency weighting per kelas
                        → loss kelas minoritas lebih besar, efek seperti oversampling
  2. Augmentasi ++    : rotation ±45°, zoom 0.35, brightness [0.6,1.4],
                        vertical_flip=True, channel_shift=20
  3. Model fresh      : bobot direset — tidak lanjut dari Round 1
    """)

    # ── Heavy augmentation generator ─────────────────────────────────────────
    heavy_datagen = ImageDataGenerator(
        rescale            = 1. / 255,
        validation_split   = VAL_SPLIT,
        rotation_range     = 45,
        width_shift_range  = 0.20,
        height_shift_range = 0.20,
        shear_range        = 0.20,
        zoom_range         = 0.35,
        horizontal_flip    = True,
        vertical_flip      = True,
        brightness_range   = [0.6, 1.4],
        channel_shift_range= 20.0,
        fill_mode          = 'nearest',
    )

    train_gen2, val_gen2 = make_generators(heavy_datagen, val_datagen)

    # ── Fresh model ───────────────────────────────────────────────────────────
    effnet_model2, effnet_backbone2 = build_effnet_experiment(NUM_CLASSES)

    print("\n" + "=" * 70)
    print("ROUND 2 — Phase 1 : Head training + class_weight + heavy aug")
    print("=" * 70)

    effnet_model2.compile(
        optimizer=optimizers.Adam(LR_HEAD),
        loss=loss_fn,
        metrics=['accuracy'],
    )

    train_gen2.reset(); val_gen2.reset()
    history_r2_head = effnet_model2.fit(
        train_gen2,
        epochs=EPOCHS_HEAD,
        validation_data=val_gen2,
        callbacks=get_experiment_callbacks('r2_phase1_head'),
        class_weight=class_weight_dict,   # ← oversampling effect
        verbose=1,
    )
    train_gen2.reset(); val_gen2.reset()

    print("\n" + "=" * 70)
    print(f"ROUND 2 — Phase 2 : Fine-tune top {EFF_FINETUNE_LAYERS} layers + class_weight")
    print("=" * 70)

    effnet_backbone2.trainable = True
    ft2_freeze_until = len(effnet_backbone2.layers) - EFF_FINETUNE_LAYERS
    for layer in effnet_backbone2.layers[:ft2_freeze_until]:
        layer.trainable = False

    effnet_model2.compile(
        optimizer=optimizers.Adam(LR_FINETUNE),
        loss=loss_fn,
        metrics=['accuracy'],
    )

    train_gen2.reset(); val_gen2.reset()
    history_r2_ft = effnet_model2.fit(
        train_gen2,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen2,
        callbacks=get_experiment_callbacks('r2_phase2_finetune'),
        class_weight=class_weight_dict,
        verbose=1,
    )
    train_gen2.reset(); val_gen2.reset()

    # ── Update catatan dengan Round 2 ────────────────────────────────────────
    notes['config']['oversampling'] = True
    notes['round2'] = {
        'phase1_head'          : history_to_dict(history_r2_head),
        'phase2_finetune'      : history_to_dict(history_r2_ft),
        'best_val_acc_head'    : float(max(history_r2_head.history.get('val_accuracy', [0]))),
        'best_val_acc_finetune': float(max(history_r2_ft.history.get('val_accuracy', [0]))),
        'class_weights'        : {str(k): float(v) for k, v in class_weight_dict.items()},
    }
    save_notes(notes)
    plot_and_save(
        history_r2_head, history_r2_ft,
        title_suffix='Round 2 (class_weight + heavy aug)',
        filename='round2_curve.png',
    )

    # ── Perbandingan Round 1 vs Round 2 ──────────────────────────────────────
    r1_best = notes['round1']['best_val_acc_finetune']
    r2_best = notes['round2']['best_val_acc_finetune']
    delta   = r2_best - r1_best
    print("\n" + "=" * 70)
    print("Perbandingan Round 1 vs Round 2")
    print("=" * 70)
    print(f"  Round 1  best val_accuracy : {r1_best:.4f}")
    print(f"  Round 2  best val_accuracy : {r2_best:.4f}")
    print(f"  Delta                      : {delta:+.4f}  {'✅' if delta >= 0 else '❌'}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(f"Selesai. Semua output tersimpan di  →  {LOG_BASE_DIR}")
print(f"  {NOTES_FILE}")
for fname in os.listdir(LOG_BASE_DIR):
    if fname.endswith('.csv') or fname.endswith('.png'):
        print(f"  {os.path.join(LOG_BASE_DIR, fname)}")
print(f"{'=' * 70}")
