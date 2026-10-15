# devils-advocate

*[English](#english) · [Español](#español)*

## English

A command-line agent that looks for reasons why an idea or architecture would **not** be a good one: hidden premises, failure modes and edge cases. Claude critiques, Grok audits the critique, and a judge writes a prioritized report.

> **Status:** complete (stages 1–6). Critic (Claude), Auditor (Grok) and Judge (Claude) make real calls within a USD budget, the report is saved, and the same code is published as a notebook. `--dry-run` makes no API calls.

### Usage
```bash
pip install -r requirements.txt
python devils_advocate.py --dry-run "Move our database to a single large server"
python devils_advocate.py --dry-run -f idea.md --rounds 1
python devils_advocate.py --configure   # store both API keys in the keychain, create config.toml
python devils_advocate.py "Move our database to a single large server"
python devils_advocate.py --budget 0.05 -f idea.md
python devils_advocate.py -v "..."     # every objection per round and the full synthesis
python devils_advocate.py -vv "..."    # also masking, calls and the cost table
python -m pytest -q
```

By default the terminal shows only the verdict, the strongest objections and the open questions, plus any failure. The saved report always has everything. The notebook runs at `-vv`.

The Critic uses `claude-haiku-5-5` (at most 1000 output tokens per call), the Auditor `grok-4.20-0309-non-reasoning` (at most 1000) and the Judge `claude-sonnet-5-5` (at most 1500). Change them in `config.toml`:
```toml
[models]
critic = "claude-sonnet-5-5"
auditor = "grok-4.3"
judge = "claude-opus-5-5"
```

### Rounds
Each round the Critic raises objections and the Auditor rates each one (*valid*, *exaggerated* or *incorrect*) and adds up to 3 edge cases. Only the 3 strongest valid or new objections (the beam) go to the next round, where the Critic must find new ones. The run stops after `--rounds` (default 2) or earlier if the Critic finds nothing new. At most `2 × rounds + 1` API calls are made; the last one is always kept for the Judge.
If the Auditor fails (for example, a wrong xAI key), the run still shows the Critic's objections, marked *unaudited*.

### Report
The Judge reads every objection with its verdict and writes the synthesis: consensus, disagreements, a steelman (the strongest case *for* the idea), open questions and a global verdict (*proceed*, *adjust* or *drop*). The terminal shows a summary (`-v` the full synthesis) and the full report is saved as `~/devils-advocate/history/YYYY-MM-DD_HHMM_<slug>.md`, readable only by you. It contains the **masked** idea, never the original. If the Judge fails, the report is saved without the synthesis. If Claude declines the request for some safety categories, the API retries it on another Claude model and the report says so.

### Keys, prices and budget
`--configure` asks for the Anthropic and xAI keys without showing them and stores them in the macOS keychain (`ANTHROPIC_API_KEY` and `XAI_API_KEY` also work). It also creates `~/.config/devils-advocate/config.toml` if it does not exist.

Prices are **not** built in: copy them (USD per million tokens) from each console's pricing page into `config.toml`:
```toml
[prices]
"claude-haiku-5-5" = { input = ..., output = ... }
"grok-4.20-0309-non-reasoning" = { input = ..., output = ... }
"claude-sonnet-5-5" = { input = ..., output = ... }
```
Before each call the tool estimates its worst-case cost (estimated input plus the full output cap). If the call could take the run over `--budget` (default $0.10), the rounds stop and the report is written with what exists; a share of the budget is always kept for the Judge. If the budget cannot cover even the first Critic call plus the Judge's share, the run stops before any call and no report is written. The budget is a best-effort cap: the input size is estimated, a Judge fallback is priced as the requested model, and a call that times out is not counted. A very long idea with `--rounds 2` or `3` may need a larger `--budget`. Use a non-reasoning Auditor model. With `-vv` it prints a cost table per role and model (the saved report always has it). A model without a price is counted in tokens only, and the tool warns that the budget cannot cover it.

Also set a spend limit on console.anthropic.com and turn auto top-up **off** on console.x.ai.

### Notebook
`devils_advocate.ipynb` is generated from `devils_advocate.py`: the same code without comments, docstrings or the command-line-only parts (marked `# cli-only`), explained in Spanish by its markdown cells, with an example (simulated) run. It runs in Google Colab: the first code cell installs the dependencies there, and keys can come from Colab's Secrets (`ANTHROPIC_API_KEY`, `XAI_API_KEY`). Open it, edit the variables in the last cells (`IDEA`, `ROUNDS`, `BUDGET`, `DRY_RUN`) and run all cells. With `DRY_RUN = False` it asks for any missing API key (Colab Secrets first) without showing or storing it. To rebuild it after changing the code:
```bash
python tools/build_notebook.py --execute
```

### How objections are ranked
Every objection gets two scores from 1 to 5:

| Score | Severity: *how bad is it if it happens?* | Probability: *how likely is it to happen?* |
|---|---|---|
| 1 | Cosmetic, easy to ignore | Very unlikely, needs a rare combination |
| 2 | Minor annoyance, cheap to fix | Unlikely |
| 3 | Real cost or delay | Plausible, could happen |
| 4 | Serious: data loss, outage, large cost | Likely |
| 5 | Fatal: the idea fails or causes harm | Almost certain |

**Priority = severity × probability** (1 to 25). A fatal but very unlikely risk (5×1 = 5) ranks below an annoying problem that will almost certainly happen (2×5 = 10). After each round only the top 3 valid or new objections move on, like beam search.

### Privacy
Before any text is sent to a model it is masked: AWS keys, `sk-`/`xai-` tokens, IPv4/IPv6 addresses, ARNs, 12-digit account IDs, emails, and your own terms from `~/.config/devils-advocate/config.toml`:
```toml
[mask]
terms = ["Project Falcon", "internal-db"]
```
The tool says how many values it masked (with `-vv` also when it masked none).

**Known limits:**
- A secret glued to a preceding letter, digit or `_` (e.g. `x2001:db8::1`, `id_123456789012`) can leak in part or whole. Keep secrets separated by spaces or punctuation, or add them to `terms`.
- An email whose local part contains a character such as `!`, `#` or `/` (e.g. `a!b@example.com`) is masked only after that character. Add the address to `terms`.
- IPv6 in non-standard forms (e.g. a full address followed by `::port`) may not be masked.

## Español

Un agente de línea de comandos que busca razones por las que una idea o arquitectura **no** sería buena: premisas ocultas, modos de fallo y casos límite. Claude critica, Grok audita la crítica y un juez escribe un informe priorizado.

> **Estado:** completo (etapas 1–6). Crítico (Claude), Auditor (Grok) y Juez (Claude) hacen llamadas reales dentro de un presupuesto en USD, el informe se guarda y el mismo código se publica como notebook. `--dry-run` no hace llamadas a APIs.

### Uso
```bash
pip install -r requirements.txt
python devils_advocate.py --dry-run "Mover nuestra base de datos a un único servidor grande"
python devils_advocate.py --dry-run -f idea.md --rounds 1
python devils_advocate.py --configure   # guarda las dos claves en el llavero y crea config.toml
python devils_advocate.py "Mover nuestra base de datos a un único servidor grande"
python devils_advocate.py --budget 0.05 -f idea.md
python devils_advocate.py -v "..."     # todas las objeciones por ronda y la síntesis completa
python devils_advocate.py -vv "..."    # además enmascarado, llamadas y tabla de costes
python -m pytest -q
```

Por defecto la terminal muestra solo el veredicto, las objeciones más fuertes y las preguntas abiertas, además de cualquier fallo. El informe guardado siempre lo tiene todo. El notebook se ejecuta con `-vv`.

El Crítico usa `claude-haiku-5-5` (máximo 1000 tokens de salida por llamada), el Auditor `grok-4.20-0309-non-reasoning` (máximo 1000) y el Juez `claude-sonnet-5-5` (máximo 1500). Los modelos se cambian en `config.toml` (sección `[models]`, ver arriba).

### Rondas
En cada ronda el Crítico plantea objeciones y el Auditor califica cada una (*valid*, *exaggerated* o *incorrect*) y añade hasta 3 casos límite. Solo las 3 objeciones válidas o nuevas más fuertes (el *beam*) pasan a la siguiente ronda, donde el Crítico debe encontrar otras nuevas. La ejecución termina tras `--rounds` (2 por defecto) o antes si el Crítico no encuentra nada nuevo. Se hacen como máximo `2 × rondas + 1` llamadas; la última siempre se reserva para el Juez.
Si el Auditor falla (por ejemplo, una clave de xAI incorrecta), se muestran igualmente las objeciones del Crítico, marcadas como *unaudited*.

### Informe
El Juez lee todas las objeciones con su veredicto y escribe la síntesis: consenso, desacuerdos, un *steelman* (el mejor argumento *a favor* de la idea), preguntas abiertas y un veredicto global (*proceed*, *adjust* o *drop*). La terminal muestra un resumen (`-v` la síntesis completa) y el informe completo se guarda en `~/devils-advocate/history/AAAA-MM-DD_HHMM_<slug>.md`, legible solo por ti. Contiene la idea **enmascarada**, nunca la original. Si el Juez falla, el informe se guarda sin la síntesis. Si Claude rechaza la petición por ciertas categorías de seguridad, la API la reintenta en otro modelo de Claude y el informe lo indica.

### Claves, precios y presupuesto
`--configure` pide las claves de Anthropic y xAI sin mostrarlas y las guarda en el llavero de macOS (también sirven `ANTHROPIC_API_KEY` y `XAI_API_KEY`). Además crea `~/.config/devils-advocate/config.toml` si no existe.

Los precios **no** vienen incluidos: cópialos (USD por millón de tokens) de la página de precios de cada consola en `config.toml`, en la sección `[prices]` (ver el ejemplo de arriba). Antes de cada llamada se estima su coste máximo (entrada estimada más el límite de salida completo). Si la llamada pudiera superar `--budget` (0,10 USD por defecto), las rondas se detienen y el informe se escribe con lo que haya; siempre se reserva una parte del presupuesto para el Juez. Si el presupuesto no cubre ni la primera llamada del Crítico más la parte del Juez, se detiene antes de cualquier llamada y no se escribe informe. El presupuesto es un límite aproximado: el tamaño de la entrada se estima, un fallback del Juez se valora al precio del modelo pedido y una llamada que agota el tiempo no se cuenta. Una idea muy larga con `--rounds 2` o `3` puede necesitar un `--budget` mayor. Usa un modelo de Auditor sin razonamiento. Con `-vv` se muestra una tabla de costes por rol y modelo (el informe guardado siempre la incluye). Un modelo sin precio se cuenta solo en tokens, y la herramienta avisa de que el presupuesto no lo cubre.

Pon también un límite de gasto en console.anthropic.com y desactiva la recarga automática en console.x.ai.

### Notebook
`devils_advocate.ipynb` se genera a partir de `devils_advocate.py`: el mismo código sin comentarios, docstrings ni las partes exclusivas de la línea de comandos (marcadas `# cli-only`), explicado en español en sus celdas markdown, con una ejecución de ejemplo (simulada). Funciona en Google Colab: allí la primera celda de código instala las dependencias, y las claves pueden venir de los *Secrets* de Colab (`ANTHROPIC_API_KEY`, `XAI_API_KEY`). Ábrelo, cambia las variables de las últimas celdas (`IDEA`, `ROUNDS`, `BUDGET`, `DRY_RUN`) y ejecuta todas las celdas. Con `DRY_RUN = False` pide las claves que falten (primero en los Secrets de Colab) sin mostrarlas ni guardarlas. Para regenerarlo tras cambiar el código: `python tools/build_notebook.py --execute`.

### Cómo se ordenan las objeciones
Cada objeción recibe dos notas del 1 al 5:

| Nota | Severidad: *¿qué tan grave es si ocurre?* | Probabilidad: *¿qué tan probable es que ocurra?* |
|---|---|---|
| 1 | Cosmético, se puede ignorar | Muy improbable, requiere una combinación rara |
| 2 | Molestia menor, barata de arreglar | Improbable |
| 3 | Coste o retraso real | Plausible, podría pasar |
| 4 | Grave: pérdida de datos, caída, coste alto | Probable |
| 5 | Fatal: la idea fracasa o causa daño | Casi seguro |

**Prioridad = severidad × probabilidad** (de 1 a 25). Un riesgo fatal pero muy improbable (5×1 = 5) queda por debajo de un problema molesto que casi seguro ocurrirá (2×5 = 10). Tras cada ronda solo pasan las 3 objeciones válidas o nuevas con mayor prioridad, como en beam search.

### Privacidad
Antes de enviar texto a un modelo se enmascara: claves de AWS, tokens `sk-`/`xai-`, direcciones IPv4/IPv6, ARNs, IDs de cuenta de 12 dígitos, emails y tus propios términos de `~/.config/devils-advocate/config.toml` (ver el ejemplo de arriba). La herramienta dice cuántos valores enmascaró (con `-vv` también cuando no enmascaró ninguno).

**Límites conocidos:**
- Un secreto pegado a una letra, dígito o `_` anterior (p. ej. `x2001:db8::1`, `id_123456789012`) puede filtrarse en parte o entero. Separa los secretos con espacios o puntuación, o añádelos a `terms`.
- Un email cuya parte local contiene un carácter como `!`, `#` o `/` (p. ej. `a!b@example.com`) solo se enmascara después de ese carácter. Añade la dirección a `terms`.
- IPv6 en formas no estándar (p. ej. una dirección completa seguida de `::puerto`) puede no enmascararse.
