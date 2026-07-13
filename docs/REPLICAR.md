# Replicar el proyecto en otro ambiente

El repositorio trae **todo el código, configs y docs**, pero dos carpetas grandes están
**ignoradas por git** (ver `.gitignore`) y por tanto NO viajan al clonar. Hay que
llevarlas aparte y acomodarlas en su sitio.

---

## 1. Qué NO está en el repo (y hay que traer aparte)

| Carpeta | Tamaño | ¿Para qué sirve? | ¿Cuándo la necesitas? |
|---|---|---|---|
| `models/` | ~626 MB | Especialistas PPO por layout (`best.zip` + marcador `enabled` + `terrain.key`), más seeds, `tb/` y `final/last.zip` | Solo si quieres desplegar los **especialistas PPO** o re-entrenar/analizar. El planner NO usa modelos. |
| `data/bc/` | ~13 MB | 9 datasets `.npz` de Behavioral Cloning (warm-start del planner) | Solo si vas a **re-entrenar** (BC + PPO). |

> **Convención del proyecto** (`.gitignore`): los modelos entrenados son grandes y NO se
> versionan; se versiona el *config* al lado, no el `.zip`. Por eso se transfieren como
> paquetes aparte, no dentro del repo.

Estas carpetas se distribuyen como dos zips:

- `overcooked_models.zip`  → contiene `models/…`
- `overcooked_bc_data.zip` → contiene `data/bc/…`

Los zips guardan las rutas relativas a la raíz del repo, así que al descomprimirlos en la
raíz del clon, cada carpeta cae exactamente en su sitio.

---

## 2. Cómo acomodarlas (paso a paso)

```bash
# 1) Clonar el repo
git clone git@github.com:Edgar100800/overcooked-py-game.git
cd overcooked-py-game

# 2) Recrear el entorno (NO se copia .venv; se reconstruye)
bash scripts/setup_venv.sh      # crea .venv, instala torch (cu118) + overcooked-ai + RL
source scripts/env.sh           # activa módulos + venv

# 3) Copiar los dos zips al nuevo ambiente (scp / Drive / USB) y descomprimir
#    en la RAÍZ del repo para que queden en su lugar:
unzip /ruta/a/overcooked_models.zip  -d .    # crea models/
unzip /ruta/a/overcooked_bc_data.zip -d .    # crea data/bc/

# 4) Verificar
python -m evaluation.run_gate --gate G8      # gate integral de la entrega
```

Estructura resultante esperada (extracto):

```
overcooked-py-game/
├── models/                         # ← del zip (ignorado por git)
│   ├── rehearsal_kitchen/
│   │   ├── best.zip                #   modelo desplegado
│   │   ├── enabled                 #   marcador: se despliega el PPO
│   │   └── terrain.key             #   hash del layout (fallback por terreno)
│   └── custom_zigzag_kitchen/      #   (los 2 layouts habilitados hoy)
│       └── ...
└── data/
    └── bc/                         # ← del zip (ignorado por git)
        ├── asymmetric_advantages.npz
        └── ...                     #   9 datasets .npz
```

> Como `models/` y `data/` siguen en `.gitignore`, tras descomprimir **no** se commitean
> por accidente.

---

## 3. Requisitos de entorno (los instala `scripts/setup_venv.sh`)

- **Python 3.10.2** (fallback `PYVER=3.8.0` si `overcooked-ai` no compila).
- **torch con CUDA 11.8** para GPU; el mismo wheel corre en CPU (para solo evaluar el
  planner basta CPU).
- Dependencias: `requirements.txt` (`overcooked-ai`, `numpy<2`, PyYAML, Pillow, imageio,
  scipy) y `requirements-rl.txt` (stable-baselines3, gymnasium, tensorboard, cloudpickle,
  imageio-ffmpeg).
- ⚠️ **`numpy<2` es restricción dura** de overcooked-ai — no subirla.

---

## 4. ¿Qué necesito según lo que quiera hacer?

- **Solo el planner** (lo que juega E1-E3): basta el repo + `.venv`. No hacen falta zips.
- **Especialistas PPO**: repo + `.venv` + `overcooked_models.zip`.
- **Re-entrenar**: todo lo anterior + `overcooked_bc_data.zip` + GPU.

> Versión ligera opcional: si solo quieres **desplegar** (no re-entrenar ni analizar), un
> zip de ~14 MB con `models/*/best.zip`, `models/*/enabled` y `models/*/terrain.key` de los
> layouts habilitados es suficiente.
