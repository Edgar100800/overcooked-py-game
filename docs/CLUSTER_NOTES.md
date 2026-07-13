# Notas del clúster khipu (SLURM) — trampas y recetas

Apuntes prácticos para correr entrenamientos/gates de este proyecto en khipu sin
tropezar. Verificado en 2026-07-11; patrón packed (§3.5) verificado 2026-07-13.

---

## 0. TL;DR (lo que hay que recordar)

- **Un solo job TUYO por nodo.** Dos jobs tuyos en el mismo nodo se matan entre sí
  (bug del epilog, §2). Aplica aunque sean de cuentas distintas (es por usuario).
- **El límite no son los cores** (~250 idle), es la **cuota de QOS**: `a-tesis` 32 cpu /
  3 jobs, `a-pregrado` 32 cpu / 2 jobs. Combinadas: **64 cpu, 5 jobs**.
- El `.venv` necesita el **módulo cargado** en runtime (`source scripts/env.sh`).
- Entrenamiento en **CPU** por defecto (Overcooked es CPU-bound; evita fragilidad GPU).

---

## 1. Cuentas, QOS y límites

Dos cuentas, con presupuesto SEPARADO cada una (`sacctmgr show qos`):

| Cuenta   | QOS         | MaxSubmit | MaxJobs (corriendo) | CPU | MaxWall |
|----------|-------------|-----------|---------------------|-----|---------|
| tesis    | a-tesis     | 5         | 3                   | 32  | 24 h    |
| pregrado | a-pregrado  | 3         | 2                   | 32  | **8 h** |

- Para GPU: `--account=tesis --qos=a-tesis --partition=gpu`.
- Para CPU: `--partition=standard` (nodos n[003-006]).
- **OJO pregrado:** wall ≤ 8 h → usar `--time=07:00:00` (el default de 12 h se queda
  `PENDING` con razón `QOSMaxWallDurationPerJobLimit`).
- Nodos: standard = n003(64c) n004(64c) n005(64c) n006(96c); GPU = ag001(128c, A100+MIG),
  ds001/g002 (RTX A6000), g001 (Tesla).

---

## 2. 🐞 EL BUG: el epilog mata tus jobs co-ubicados

### Síntoma
Varios jobs tuyos en el **mismo nodo**; cuando **uno** termina, los demás mueren con
`State=FAILED ExitCode 0:9` (**SIGKILL**), **sin traceback**. Pasa en CPU y en GPU.
No es OOM (RSS bajo) ni error del código.

### Causa raíz
`/etc/slurm/slurm.epilog.clean` (script de root que corre al terminar cada job) hace:

```sh
job_list=$(squeue --noheader --format=%A --user=$SLURM_UID --node=localhost)  # <-- BUG
for job_id in $job_list; do
    [ $job_id -ne $SLURM_JOB_ID ] && exit 0     # "tengo otro job aqui" -> no limpiar
done
pkill -KILL -U $SLURM_UID                        # si no -> mata TODO lo tuyo en el nodo
```

El guard consulta `squeue --node=localhost`, pero **`localhost` no es un nombre de nodo
válido** en SLURM (los nodos son `n003`, `ag001`, ...):

```
$ squeue --node=localhost
squeue: error: Invalid node name localhost      # devuelve VACIO
```

→ `job_list` vacío → el `for` nunca corre → el guard nunca hace `exit 0` → **siempre**
cae en `pkill -KILL -U tú`. Es decir: **cada fin de job tuyo hace un `kill -9` de todos
tus procesos en ese nodo**, incluidos tus otros jobs.

- Es `pkill -U` (por **usuario**, a nivel SO) → no tiene nada que ver con MPS/CUDA; por
  eso mata igual CPU y GPU, y por eso el SIGKILL no deja traceback.
- El fix "correcto" sería del admin: `--node=$SLURMD_NODENAME` en vez de `localhost`.

### La pista que lo delató
Correlación temporal: los jobs morían **1 segundo después** de que otro job mío terminaba
en el mismo nodo. + No-OOM (RSS 3 GB / 24 GB) + sin traceback = señal externa, no bug propio.

---

## 3. La receta segura (un job por nodo, ambas cuentas)

Para correr N entrenamientos en paralelo sin que se maten:

```bash
# tesis (3 jobs) en nodos distintos:
sbatch --array=0-0 --nodelist=n004 --export=ALL,JOBS=training/jobs_stepA.txt sbatch/train/run_train_ppo.sh
sbatch --array=1-1 --nodelist=n005 --export=ALL,JOBS=training/jobs_stepA.txt sbatch/train/run_train_ppo.sh
sbatch --array=2-2 --nodelist=n006 --export=ALL,JOBS=training/jobs_stepA.txt sbatch/train/run_train_ppo.sh

# pregrado (2 jobs) en nodos SIN jobs de tesis (n003, ag001), wall <=8h:
sbatch --account=pregrado --qos=a-pregrado --time=07:00:00 --array=0-0 --nodelist=n003 \
       --export=ALL,JOBS=training/jobs_stepA2.txt sbatch/train/run_train_ppo.sh
sbatch --account=pregrado --qos=a-pregrado --time=07:00:00 --partition=gpu --array=1-1 --nodelist=ag001 \
       --export=ALL,JOBS=training/jobs_stepA2.txt sbatch/train/run_train_ppo.sh
```

→ 5 trainings, 5 nodos, ~50 cpu. Máximo aprovechamiento seguro.

### ¿Tiene sentido meter MÁS de 5?
Como jobs SEPARADOS no: `MaxJobs` (3+2) lo capa a 5, y apilar más en un nodo = se matan
(§2). **Pero SÍ como entrenamientos empaquetados (ver §3.5)**: la cuota real es de CPUs
(32/cuenta = 64 total), no de entrenamientos — 6 entrenamientos de ~10 hilos caben en
2 jobs de 30 cpus.

## 3.5. Patrón "job empaquetado" (verificado 2026-07-13, jobs 46809/46810 COMPLETED)

**N entrenamientos `train_ppo` en PARALELO dentro de UN solo job** de 30 cpus
(`sbatch/train/run_train_packed.sh` + manifiesto `training/jobs_packed_*.txt`):

```bash
sbatch --export=ALL,JOBS=training/jobs_packed_tesis.txt sbatch/train/run_train_packed.sh
# variante pregrado (wall <=8h) y/o nodo fijo:
sbatch --account=pregrado --qos=a-pregrado --time=07:00:00 --partition=gpu --nodelist=g002 \
       --export=ALL,JOBS=training/jobs_packed_pregrado.txt sbatch/train/run_train_packed.sh
```

Por qué funciona donde los jobs separados no:
- **Inmune al bug del epilog (§2)**: sigue habiendo UN job mío por nodo; cuando termina,
  todos sus procesos terminan juntos — no hay job co-ubicado que matar.
- **Esquiva `MaxJobs`**: 1 slot de los 5 aloja 3 entrenamientos → hasta **6 concurrentes**
  (2 jobs × 3) con la cuota combinada de 64 cpus, vs 5 con jobs de 10 cpus.
- **Encadenable**: con `--dependency=afterany:<jobs>` la siguiente tanda arranca sola al
  liberarse la cuota (así corrió la tanda 2 del día de competencia sin intervención).

Trade-off: si el job muere, mueren los N entrenamientos (mitigado: `callbacks.py` guarda
`last.zip` en cada eval y cada training escribe su propio log `logs/ppo-pack-<jid>-*.log`).

---

## 4. Otras trampas ya resueltas (contexto del proyecto)

- **`.venv` + libpython**: el venv se creó con `module load python3/3.10.2`; su `libpython`
  vive en el módulo. Sin el módulo, `.venv/bin/python` no arranca (`libpython3.10.so.1.0`).
  Solución: `source scripts/env.sh` (carga módulo + activa venv). Los sbatch lo hacen solo.
- **torch cu118 vs cu130**: instalar torch con índice cu118 (compat driver A100). Un `torch`
  sin versión en requirements lo reinstala a cu130. Ver `scripts/setup_venv.sh`.
- **Lmod + `set -u`**: los scripts de Lmod referencian variables sin definir; sourcearlos
  bajo `set -u` aborta. Usar `set -eo pipefail` (sin `-u`).
- **numpy<2**: overcooked-ai lo exige; no actualizar.
- **/tmp es local al nodo**: un archivo en `/tmp` del login NO se ve en el nodo de cómputo.
  Para pasar archivos a un job, ponerlos en el repo (FS compartido /home), no en /tmp.
- **CPU vs GPU para entrenar**: Overcooked es CPU-bound y el CNN es diminuto → la GPU casi
  no acelera. Entrenar en CPU es igual de rápido y evita la fragilidad de compartir GPU.

---

## 5. Comandos útiles

```bash
squeue -u $USER -o "%.12i %.9T %.4C %.10a %.9P %.10R"   # mis jobs (id, estado, cpus, cuenta, part, nodo)
sacct -j <JID> --format=JobID,State,Elapsed,ExitCode,MaxRSS   # post-mortem (¿OOM? ¿SIGKILL?)
sinfo -N -n n003,n004,n005,n006 -o "%.8N %.5c %.15C %.10T"    # cores libres por nodo (A/I/O/T)
sacctmgr -n show qos a-tesis,a-pregrado format=Name,MaxJobsPerUser,MaxTRESPerUser,MaxWall
scontrol show node ag001 | grep -iE "Gres|CfgTRES"           # rebanadas MIG/shard del A100
```
