# devils-advocate

*[English](#english) · [Español](#español)*

## English

A command-line agent that looks for reasons why an idea or architecture would **not** be a good one: hidden premises, failure modes and edge cases. Claude critiques, Grok audits the critique, and a judge writes a prioritized report.

> **Status:** stage 4. Critic (Claude), Auditor (Grok) and Judge (Claude) make real calls, and the report is saved. Cost tracking and `--configure` come in stage 5. `--dry-run` makes no API calls.

### Usage
```bash
pip install -r requirements.txt
python devils_advocate.py --dry-run "Move our database to a single large server"
python devils_advocate.py --dry-run -f idea.md --rounds 1
export ANTHROPIC_API_KEY=... XAI_API_KEY=...   # live run: Claude (Critic) and Grok (Auditor)
python devils_advocate.py "Move our database to a single large server"
python -m pytest -q
```

The Critic uses `claude-haiku-5-5` (at most 600 output tokens per call), the Auditor `grok-4.20-0309-non-reasoning` (at most 500) and the Judge `claude-sonnet-5-5` (at most 700). Change them in `config.toml`:
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
The Judge reads every objection with its verdict and writes the synthesis: consensus, disagreements, a steelman (the strongest case *for* the idea), open questions and a global verdict (*proceed*, *adjust* or *drop*). The report is printed and saved as `~/devils-advocate/history/YYYY-MM-DD_HHMM_<slug>.md`, readable only by you. It contains the **masked** idea, never the original. If the Judge fails, the report is saved without the synthesis. If Claude declines the request for some safety categories, the API retries it on another Claude model and the report says so.

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
The tool prints how many replacements were made.

**Known limits:**
- A secret glued to a preceding letter, digit or `_` (e.g. `x2001:db8::1`, `id_123456789012`) can leak in part or whole. Keep secrets separated by spaces or punctuation, or add them to `terms`.
- An email whose local part contains a character such as `!`, `#` or `/` (e.g. `a!b@example.com`) is masked only after that character. Add the address to `terms`.
- IPv6 in non-standard forms (e.g. a full address followed by `::port`) may not be masked.

## Español

Un agente de línea de comandos que busca razones por las que una idea o arquitectura **no** sería buena: premisas ocultas, modos de fallo y casos límite. Claude critica, Grok audita la crítica y un juez escribe un informe priorizado.

> **Estado:** etapa 4. Crítico (Claude), Auditor (Grok) y Juez (Claude) hacen llamadas reales y el informe se guarda. El control de costes y `--configure` llegan en la etapa 5. `--dry-run` no hace llamadas a APIs.

### Uso
```bash
pip install -r requirements.txt
python devils_advocate.py --dry-run "Mover nuestra base de datos a un único servidor grande"
python devils_advocate.py --dry-run -f idea.md --rounds 1
export ANTHROPIC_API_KEY=... XAI_API_KEY=...   # ejecución real: Claude (Crítico) y Grok (Auditor)
python devils_advocate.py "Mover nuestra base de datos a un único servidor grande"
python -m pytest -q
```

El Crítico usa `claude-haiku-5-5` (máximo 600 tokens de salida por llamada), el Auditor `grok-4.20-0309-non-reasoning` (máximo 500) y el Juez `claude-sonnet-5-5` (máximo 700). Los modelos se cambian en `config.toml` (sección `[models]`, ver arriba).

### Rondas
En cada ronda el Crítico plantea objeciones y el Auditor califica cada una (*valid*, *exaggerated* o *incorrect*) y añade hasta 3 casos límite. Solo las 3 objeciones válidas o nuevas más fuertes (el *beam*) pasan a la siguiente ronda, donde el Crítico debe encontrar otras nuevas. La ejecución termina tras `--rounds` (2 por defecto) o antes si el Crítico no encuentra nada nuevo. Se hacen como máximo `2 × rondas + 1` llamadas; la última siempre se reserva para el Juez.
Si el Auditor falla (por ejemplo, una clave de xAI incorrecta), se muestran igualmente las objeciones del Crítico, marcadas como *unaudited*.

### Informe
El Juez lee todas las objeciones con su veredicto y escribe la síntesis: consenso, desacuerdos, un *steelman* (el mejor argumento *a favor* de la idea), preguntas abiertas y un veredicto global (*proceed*, *adjust* o *drop*). El informe se muestra y se guarda en `~/devils-advocate/history/AAAA-MM-DD_HHMM_<slug>.md`, legible solo por ti. Contiene la idea **enmascarada**, nunca la original. Si el Juez falla, el informe se guarda sin la síntesis. Si Claude rechaza la petición por ciertas categorías de seguridad, la API la reintenta en otro modelo de Claude y el informe lo indica.

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
Antes de enviar texto a un modelo se enmascara: claves de AWS, tokens `sk-`/`xai-`, direcciones IPv4/IPv6, ARNs, IDs de cuenta de 12 dígitos, emails y tus propios términos de `~/.config/devils-advocate/config.toml` (ver el ejemplo de arriba). La herramienta muestra cuántos reemplazos hizo.

**Límites conocidos:**
- Un secreto pegado a una letra, dígito o `_` anterior (p. ej. `x2001:db8::1`, `id_123456789012`) puede filtrarse en parte o entero. Separa los secretos con espacios o puntuación, o añádelos a `terms`.
- Un email cuya parte local contiene un carácter como `!`, `#` o `/` (p. ej. `a!b@example.com`) solo se enmascara después de ese carácter. Añade la dirección a `terms`.
- IPv6 en formas no estándar (p. ej. una dirección completa seguida de `::puerto`) puede no enmascararse.
