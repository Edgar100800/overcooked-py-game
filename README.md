# Overcooked AI Demonstration Collector

Proyecto para recolectar demostraciones humanas en Overcooked-AI. Incluye un menu interactivo para jugar partidas, ver progreso, cambiar niveles/agentes y preparar datasets para Imitation Learning.

## Que incluye

- Runner de juego y grabacion de demostraciones.
- Menu interactivo para usuarios no tecnicos.
- Seguimiento de progreso por escenario.
- Politicas automaticas: `stay`, `random_motion`, `greedy_full_task`.
- Control humano por teclado.
- Escenarios oficiales y custom en `configs/layouts/`.

## Instalacion

Recomendado usar entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Si estas en macOS y `python` no existe, usa `python3` para crear el entorno. Dentro del entorno activado, `python` deberia funcionar.

## Uso recomendado

Abrir el menu:

```bash
source .venv/bin/activate
python -m src.game_menu
```

Desde el menu puedes:

- Jugar la siguiente grabacion recomendada.
- Elegir nivel y agente manualmente.
- Ver progreso detallado.
- Leer controles e instrucciones.

## Controles

- Movimiento: flechas o `W/A/S/D`
- Interactuar, tomar, dejar o servir: `Space`, `E` o `Enter`
- Cancelar partida: `Escape` o `Q`

La ventana muestra el nivel y el paso actual, por ejemplo:

```text
Overcooked AI | Grabacion 4/36 | Nivel: cramped_room | Paso 120/250
```

## Comandos utiles

Probar una partida sin grabacion manual:

```bash
python -m src.run_game --config configs/play.yaml
```

Crear una grabacion directa con YAML:

```bash
python -m src.collect_demonstrations --config configs/collect_demonstrations.yaml
```

Ver progreso:

```bash
python -m src.dataset_progress
```

## Configuracion principal

Archivo:

```text
configs/collect_demonstrations.yaml
```

Puntos importantes:

- `environment.horizon: 250`: duracion de cada partida.
- `environment.layout_name`: escenario oficial.
- `environment.layout_file`: escenario custom `.layout`.
- `policies.agent_0.name`: agente automatico.
- `policies.agent_1.name: human_keyboard`: agente humano.
- `data_collection.output_dir: data/demonstrations`: salida de grabaciones.

## Escenarios custom

Los escenarios custom estan en:

```text
configs/layouts/
```

Si usas un escenario custom en una entrega, tambien debes adjuntar su archivo `.layout`.

## Entrega de datos

Las grabaciones generadas no se versionan en Git. Estan ignoradas por `.gitignore`:

- `data/`
- `outputs/`
- `resultados/`
- carpetas de entrega

Para entregar, prepara una carpeta aparte con:

- Grabaciones `.pkl` desde `data/demonstrations/`.
- `integrantes.txt`.
- Archivos `.layout` si usaste escenarios custom.

## Notas

El warning de Gym sobre mantenimiento puede aparecer al iniciar Overcooked-AI. No bloquea el juego.

Si las teclas no responden bien, usa el menu actualizado y mantén presionada la tecla de movimiento. El control humano tambien captura pulsaciones rapidas para acciones como `Space` o `E`.
