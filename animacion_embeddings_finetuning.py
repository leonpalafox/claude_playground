# -*- coding: utf-8 -*-
"""
Animación: cómo los embeddings de ResNet-18 forman clusters durante el
fine-tuning en CIFAR-10 (proyectados con UMAP).

Mismo experimento que la práctica (ResNet-18 de ImageNet, 10 000 imágenes de
entrenamiento, 2 épocas, layer4 + fc entrenables, Adam lr=1e-4, imágenes a
224x224), pero tomando "fotografías" de los embeddings durante el
entrenamiento. Cada fotografía se proyecta con UMAP inicializado con la
proyección anterior y alineada con Procrustes, y luego se interpola entre
fotografías para que el movimiento sea continuo.

La única salida es el video: embeddings_finetuning.mp4

Requisitos: pip install torch torchvision umap-learn datasets scikit-learn scipy matplotlib
            y ffmpeg instalado (en Colab ya viene).
"""

import os
import warnings
import logging

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms
from torchvision import models

import datasets as hf_datasets
from datasets import load_dataset
from sklearn.metrics import silhouette_samples
from scipy.linalg import orthogonal_procrustes
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import CubicSpline
import umap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.animation import FuncAnimation, FFMpegWriter

hf_datasets.disable_progress_bars()

# ----------------------------------------------------------------------------
# Configuración (igual que la práctica)
# ----------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_TRAIN = 10000
NUM_TEST = 2000
BATCH_SIZE = 64
EPOCHS = 2
NUM_UMAP_SAMPLES = 1200

# Animación
NUM_SNAPSHOTS = 48          # fotografías de los embeddings durante el entrenamiento
FPS = 30
FRAMES_PER_SNAPSHOT = 14    # frames interpolados entre fotografías consecutivas
HOLD_START_S = 1.5
HOLD_END_S = 2.5
OUTPUT = "embeddings_finetuning.mp4"

class_names = [
    "avión", "auto", "pájaro", "gato", "venado",
    "perro", "rana", "caballo", "barco", "camión"
]

# ----------------------------------------------------------------------------
# Datos (CIFAR-10 desde Hugging Face, mismos subconjuntos que la práctica)
# ----------------------------------------------------------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])


class HFCIFAR10(Dataset):
    def __init__(self, split, transform=None):
        self.data = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[int(idx)]
        img, label = item["img"], item["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


train_full = HFCIFAR10("train", transform=transform)
test_full = HFCIFAR10("test", transform=transform)

train_indices = np.random.choice(len(train_full), NUM_TRAIN, replace=False)
test_indices = np.random.choice(len(test_full), NUM_TEST, replace=False)

train_dataset = Subset(train_full, train_indices)
test_dataset = Subset(test_full, test_indices)

# ----------------------------------------------------------------------------
# Modelo
# ----------------------------------------------------------------------------
try:
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
except AttributeError:
    model = models.resnet18(pretrained=True)

model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

for p in model.parameters():
    p.requires_grad = False
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

# Como conv1..layer3 están congeladas, su salida no cambia durante el
# entrenamiento: la calculamos una sola vez y la guardamos. Así podemos tomar
# muchas fotografías de los embeddings sin volver a pasar todas las imágenes
# por la red completa. (Diferencia menor con la práctica: aquí las BatchNorm
# congeladas se mantienen en modo eval y no actualizan sus estadísticas.)
frozen = nn.Sequential(
    model.conv1, model.bn1, model.relu, model.maxpool,
    model.layer1, model.layer2, model.layer3,
).eval()


@torch.no_grad()
def cache_features(dataset):
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)
    feats, labels = [], []
    for x, y in loader:
        feats.append(frozen(x.to(device)).half().cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


train_feats, train_labels = cache_features(train_dataset)
test_feats, test_labels = cache_features(test_dataset)


def embed(feats):
    """layer4 → avgpool → flatten: el embedding de 512 dimensiones."""
    z = model.layer4(feats.to(device).float())
    return torch.flatten(model.avgpool(z), 1)


@torch.no_grad()
def snapshot():
    """Embeddings de las imágenes de UMAP + accuracy en todo el test."""
    model.eval()
    embs, correct = [], 0
    for i in range(0, len(test_feats), 256):
        e = embed(test_feats[i:i + 256])
        preds = model.fc(e).argmax(1).cpu()
        correct += (preds == test_labels[i:i + 256]).sum().item()
        embs.append(e.cpu())
    embs = torch.cat(embs)[:NUM_UMAP_SAMPLES].numpy()
    return embs, correct / len(test_feats)


# ----------------------------------------------------------------------------
# Fine-tuning con fotografías de los embeddings
# ----------------------------------------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)

steps_per_epoch = int(np.ceil(NUM_TRAIN / BATCH_SIZE))
total_steps = EPOCHS * steps_per_epoch

# La representación cambia mucho al principio y poco al final:
# tomamos más fotografías al inicio (espaciado cuadrático).
snap_steps = np.unique(np.round(np.linspace(0, 1, NUM_SNAPSHOTS) ** 2 * total_steps).astype(int))
snap_set = set(snap_steps.tolist())

snap_embs, snap_acc = [], []

def take_snapshot():
    e, a = snapshot()
    snap_embs.append(e)
    snap_acc.append(a)

step = 0
take_snapshot()  # paso 0 = ResNet preentrenada en ImageNet, antes del fine-tuning
gen = torch.Generator().manual_seed(SEED)
for epoch in range(EPOCHS):
    perm = torch.randperm(NUM_TRAIN, generator=gen)
    for i in range(0, NUM_TRAIN, BATCH_SIZE):
        idx = perm[i:i + BATCH_SIZE]
        x, y = train_feats[idx], train_labels[idx].to(device)

        model.layer4.train()
        model.fc.train()
        optimizer.zero_grad()
        loss = criterion(model.fc(embed(x)), y)
        loss.backward()
        optimizer.step()

        step += 1
        if step in snap_set:
            take_snapshot()

labels = test_labels[:NUM_UMAP_SAMPLES].numpy()
snap_acc = np.array(snap_acc)
K = len(snap_embs)

# ----------------------------------------------------------------------------
# UMAP encadenado + alineación de Procrustes
# ----------------------------------------------------------------------------
def normalize(Y):
    Y = Y - Y.mean(0)
    return Y / np.sqrt((Y ** 2).sum(1).mean())

layouts = []
prev = None
for k, E in enumerate(snap_embs):
    reducer = umap.UMAP(
        n_neighbors=20, min_dist=0.1, n_components=2, metric="cosine",
        random_state=SEED,
        init="spectral" if prev is None else prev,
    )
    Y = normalize(reducer.fit_transform(E))
    if prev is not None:
        R, _ = orthogonal_procrustes(Y, prev)
        Y = Y @ R
    layouts.append(Y)
    prev = Y
layouts = np.stack(layouts)  # (K, N, 2)

# Separación de cada clase en el espacio original de 512 dimensiones
# (silhouette con distancia coseno; -1 = mezclada, 1 = cluster aislado).
sil = np.zeros((K, 10))
for k, E in enumerate(snap_embs):
    s = silhouette_samples(E, labels, metric="cosine")
    sil[k] = [s[labels == c].mean() for c in range(10)]
sil_mean = sil.mean(1)

# ----------------------------------------------------------------------------
# Interpolación temporal suave
# ----------------------------------------------------------------------------
# Un suavizado ligero entre fotografías elimina el "temblor" propio de UMAP;
# el spline cúbico da posiciones y velocidades continuas.
layouts_s = gaussian_filter1d(layouts, sigma=0.8, axis=0, mode="nearest")
layouts_s[0], layouts_s[-1] = layouts[0], layouts[-1]
knots = np.arange(K)
pos_spline = CubicSpline(knots, layouts_s, axis=0, bc_type="clamped")
acc_spline = CubicSpline(knots, snap_acc, bc_type="clamped")
sil_spline = CubicSpline(knots, sil, axis=0, bc_type="clamped")
silm_spline = CubicSpline(knots, sil_mean, bc_type="clamped")
step_spline = CubicSpline(knots, snap_steps.astype(float), bc_type="clamped")

n_move = (K - 1) * FRAMES_PER_SNAPSHOT
n_hold0 = int(HOLD_START_S * FPS)
n_hold1 = int(HOLD_END_S * FPS)
n_frames = n_hold0 + n_move + n_hold1

def ease(t):
    # arranque y llegada suaves (smootherstep)
    return t * t * t * (t * (6 * t - 15) + 10)

def frame_to_u(f):
    t = np.clip((f - n_hold0) / max(n_move - 1, 1), 0, 1)
    return ease(t) * (K - 1)

# Encuadre que sigue a los puntos: a medida que los clusters se compactan la
# cámara se acerca suavemente (misma escala en x e y para no deformar).
# Formato vertical 9:16 (1080x1920) para redes sociales
FIG_W, FIG_H = 7.2, 12.8
AX_RECT = [0.04, 0.335, 0.92, 0.445]
ASPECT = AX_RECT[2] * FIG_W / (AX_RECT[3] * FIG_H)   # ancho/alto del panel
centers, half_h = [], []
for Y in layouts_s:
    lo, hi = np.percentile(Y, [0.5, 99.5], axis=0)
    c = (lo + hi) / 2
    h = max((hi[1] - lo[1]) / 2, (hi[0] - lo[0]) / 2 / ASPECT) * 1.12
    centers.append(c); half_h.append(h)
centers = gaussian_filter1d(np.array(centers), sigma=2.0, axis=0, mode="nearest")
half_h = gaussian_filter1d(np.array(half_h), sigma=2.0, mode="nearest")
cam_c = CubicSpline(knots, centers, axis=0, bc_type="clamped")
cam_h = CubicSpline(knots, half_h, bc_type="clamped")

# ----------------------------------------------------------------------------
# Figura
# ----------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#a3a29c"
GRID = "#e7e6e2"
ACCENT = "#2a78d6"
palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4",
           "#008300", "#6250d6", "#e34948", "#8a6a3c", "#5f6b7a"]
point_colors = np.array(palette)[labels]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": GRID,
})

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150, facecolor=SURFACE)
ax = fig.add_axes(AX_RECT)
ax.set_facecolor(SURFACE)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

LX = 0.06
fig.text(LX, 0.965, "Fine-tuning de ResNet-18\nen CIFAR-10", fontsize=30,
         weight="bold", color=INK, va="top", linespacing=1.1)
fig.text(LX, 0.872, "Cómo se separan las clases en el espacio\nde embeddings (UMAP, 1200 imágenes)",
         fontsize=16, color=INK2, va="top", linespacing=1.25)

scat = ax.scatter(layouts[0][:, 0], layouts[0][:, 1], c=point_colors,
                  s=16, alpha=0.8, linewidths=0)

halo = [pe.withStroke(linewidth=5, foreground=SURFACE)]
class_texts = [ax.text(0, 0, n, fontsize=17, weight="bold", color=INK,
                       ha="center", va="bottom", path_effects=halo)
               for n in class_names]

# Estado y progreso
status_main = fig.text(LX, 0.792, "", fontsize=23, weight="bold", color=INK)
status_sub = fig.text(0.94, 0.792, "", fontsize=23, weight="bold", color=ACCENT, ha="right")

pbar = fig.add_axes([LX, 0.778, 0.88, 0.007])
pbar.set_xlim(0, 1); pbar.set_ylim(0, 1); pbar.axis("off")
pbar.add_patch(plt.Rectangle((0, 0), 1, 1, color=GRID, lw=0))
prog = plt.Rectangle((0, 0), 0, 1, color=ACCENT, lw=0)
pbar.add_patch(prog)
pbar.axvline(steps_per_epoch / total_steps, color=SURFACE, lw=3)

# Leyenda: 2 filas x 5 columnas
for c, name in enumerate(class_names):
    row, col = c // 5, c % 5
    x0 = LX + col * 0.18
    y0 = 0.298 - row * 0.03
    fig.patches.append(matplotlib.patches.Ellipse(
        (x0 + 0.012, y0 + 0.0045), 0.022, 0.022 * FIG_W / FIG_H,
        transform=fig.transFigure, color=palette[c], figure=fig))
    fig.text(x0 + 0.03, y0, name, fontsize=15, color=INK)

def small_chart(rect, title, ylim_, fmt):
    a = fig.add_axes(rect)
    a.set_facecolor(SURFACE)
    a.set_xlim(0, total_steps); a.set_ylim(*ylim_)
    for s in ["top", "right"]:
        a.spines[s].set_visible(False)
    a.spines["left"].set_color(GRID); a.spines["bottom"].set_color(GRID)
    a.tick_params(labelsize=13, length=0, pad=5)
    a.grid(axis="y", color=GRID, lw=1)
    a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    a.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
    a.set_xticks([0, steps_per_epoch, total_steps])
    a.set_xticklabels(["0", "época 1", "2"])
    a.set_title(title, loc="left", fontsize=16, color=INK2, pad=10)
    line, = a.plot([], [], color=ACCENT, lw=3, solid_capstyle="round")
    dot, = a.plot([], [], "o", color=ACCENT, ms=9, mec=SURFACE, mew=2)
    return a, line, dot

acc_lo = np.floor(snap_acc.min() * 10) / 10
_, acc_line, acc_dot = small_chart([0.16, 0.085, 0.31, 0.115],
                                   "Accuracy", (acc_lo, 1.0),
                                   lambda v, _: f"{v:.0%}")
sil_lo = np.floor(sil_mean.min() * 20) / 20
sil_hi = np.ceil(sil_mean.max() * 20) / 20
_, sil_line, sil_dot = small_chart([0.64, 0.085, 0.31, 0.115],
                                   "Separación", (sil_lo, sil_hi),
                                   lambda v, _: f"{v:.1f}")

# curvas densas para dibujar progresivamente
u_dense = np.linspace(0, K - 1, 600)
step_dense = step_spline(u_dense)
acc_dense = acc_spline(u_dense)
silm_dense = silm_spline(u_dense)

sil_floor, sil_ceil = np.percentile(sil, 5), np.percentile(sil, 95)

def update(f):
    u = frame_to_u(f)
    P = pos_spline(u)
    scat.set_offsets(P)
    c, h = cam_c(u), float(cam_h(u))
    ax.set_xlim(c[0] - h * ASPECT, c[0] + h * ASPECT)
    ax.set_ylim(c[1] - h, c[1] + h)

    s_cls = sil_spline(u)
    for k_, t in enumerate(class_texts):
        pts = P[labels == k_]
        med = np.median(pts, axis=0)
        top = np.percentile(pts[:, 1], 80)
        t.set_position((med[0], top + 0.03 * h))
        a = np.clip((s_cls[k_] - sil_floor) / (sil_ceil - sil_floor + 1e-9), 0, 1)
        t.set_alpha(0.15 + 0.85 * a)

    st = float(step_spline(u))
    acc = float(acc_spline(u))
    if st < 0.5:
        status_main.set_text("Antes del fine-tuning")
        status_sub.set_text("")
    else:
        ep = min(int(st // steps_per_epoch) + 1, EPOCHS)
        status_main.set_text(f"Época {ep}/{EPOCHS} · paso {int(round(st))}")
        status_sub.set_text(f"{acc:.1%}")
    prog.set_width(st / total_steps)

    m = u_dense <= u
    acc_line.set_data(step_dense[m], acc_dense[m]); acc_dot.set_data([st], [acc])
    silm = float(silm_spline(u))
    sil_line.set_data(step_dense[m], silm_dense[m]); sil_dot.set_data([st], [silm])
    return []

anim = FuncAnimation(fig, update, frames=n_frames, blit=False)
anim.save(OUTPUT, writer=FFMpegWriter(fps=FPS, bitrate=6000,
                                      extra_args=["-pix_fmt", "yuv420p"]),
          savefig_kwargs={"facecolor": SURFACE})
plt.close(fig)
