# Duckling — overdracht (stand 15 okt 2026, eind)

Upload in een nieuwe chat: een verse `duck_basis.zip` van de pod (zie onderaan) + dit bestand + de ONNX die je wilt laten meten.
Antwoorden graag kort, scherp en in het Nederlands. Eerst meten, dan claimen.

## Stand 15 okt — volg-taak, afgesloten
**Eindnetwerk: volg16c (volg16 → 16b → 16c, elk ~300 iteraties) + grid v7b + player uit `duck_player_final.zip`.**
Config = `duck_volg16c.zip` (compleet, werkt direct op duck_basis). Bewaar de 16c-ONNX én `model_*.pt` naast deze zip.
- 16b: `com_forward` rekent met het lijf-zwaartepunt ZONDER kop (kop+nek+snavel = 65 van ~190 g; volg16 hing zijn kop
  naar voren als contragewicht) + kop-reward grof 0.6 rad, gewicht 3 + `head_level` (yaw/roll) → roll 0°, yaw 4–10°.
- 16c: `head_level` op alle 4 kopgewrichten (pitch grof 0.6 rad), gewicht 4 → nek niet meer op zijn +30°-aanslag. Door Jairo goedgekeurd.
Taak: een "persoon" volgen op 50 cm met camera-achtige waarneming (bearing, afstand−0.5, zichtbaar; 60° FOV, ruis, 80 ms vertraging).
Lopen ~4.5 stap/s, rennen 7.1 stap/s (grid v7b, 1.5× cadans), terugstappen als de persoon < 36 cm komt.

**Controller (buiten het netwerk, = `Target.twist` in `duckling_follow.py`; dit gaat ook op de robot):**
- Zones met hysterese: rust (in < 8 cm & 14°, uit > 14 cm of 26°), terug (< 36 cm tot > 45 cm), lopen (≤ 7.5 cm/s), rennen (> 62 cm tot < 54 cm, vast 13 cm/s).
- Vastloop-waakhond: persoon staat, eend < 15 cm / 35° van het doel en komt 1 s niet vooruit (< 1.5 cm, < 6°) → rust.
- Rust = netwerk in 1.0 s uitfaden naar HOME (statisch stabiel).
- Persoon "loopt" pas boven 1.5 cm/s (stopt onder 0.8): sensor- of muisruis mag de rust nooit blokkeren (op de robot: camera-ruis!).
- Kop: optioneel door de controller (toets H), voor netwerken met een scheve kop.
- TODO toestandsmachine (later, Jairo 15 okt):
  - netwerk pas aan als hij ~1 s rechtop en stil staat (ook na aanzetten/oppakken);
  - **snel wakker worden in rust**: een zetje vangt hij met het netwerk uit niet op. Uitfaden mag traag (1 s),
    infaden moet ~0.1 s of direct, getriggerd door IMU (gyro-piek of kanteling > ~5° t.o.v. de rusthouding);
    het netwerk kent cmd = 0 met duwtjes uit de training, dus het kan het opvangen. Eerst in de sim testen.
  - opgepakt (sterke kanteling / verticale stoot) → netwerk uit, HOME; gevallen → later opstaan-policy.

Gemeten volg16 + deze controller (84 situaties): 82/83 eindigt in rust, 1 val, 2% trappelen op de plaats.

**volg17 (zones/stand_still in de training) = mislukt (opgelost in 16b/16c):** kop 20–35° links + 20° rol, draait niet meer naar rechts,
trappelt op ~43 cm / −16°, verder trainen werd erger. Niet gebruiken. Waarschijnlijke oorzaak: `head_pose_tracking` std 0.25
geeft bij > 20° fout ~0 reward (geen gradiënt) — bij een volgende training eerst die term twee-schalig maken (grof 0.5 rad).
Les: rust/zones horen in de controller, niet in de reward; het netwerk hoeft alleen te lopen, draaien en terugstappen.

## Robot
~15 cm, ~190 g, SLA. Heup: yaw + roll + pitch (6 mm coreless + encoders). Enkel/teen/nek: Emax ES9051 (stall ~80 mNm, onbelast 13.1 rad/s).
Hoofd 3 DOF (pitch, roll, yaw) + nek-pitch. Snavel (`passive_jaw`, 0–30°) wordt later een ES9051, buiten de policy om.
RP2350, policy ~57k params, 50 Hz. HOME is statisch in balans → rust-modus (policy uitfaden bij stilstand).

**Namen (vogelpoot):** in model/code ↔ anatomisch

| model | anatomisch |
|---|---|
| `hip_yaw` / `hip_roll` / `hip_pitch` | heup |
| `knee` | **enkel** |
| `ankle` | **teen** |

Code-namen blijven zo (CSV, symmetrietabel en checkpoints hangen eraan).

## Pijplijn (bron van de animatie)
1. **Jouw loopje:** `favorites/jairo_walk_v5c.npz` = parameters uit `gait_gen.py --view` (schuifjes, `S` = opslaan).
   Nieuw sinds v5: `com_ahead_m` (schuif `zwaartepunt_voor_mm`) en `push_deg` (`teen_afzet_gr`).
2. **Regels per snelheid/bocht/zijstap:** `walk_speed.py` → `rule_params()`.
   Zwaartepunt schaalt met |snelheid| (0 onder 20%, vol bij 100%, meer daarboven; achteruit = áchter de voeten).
   Teen-afzet alleen vooruit, begrensd op `PUSH_MAX = 20°` (30° vroeg ~15 rad/s van de teen).
3. **Generator (IK op het MuJoCo-model):** `gait_gen.py` → `generate()`, 15 snelheden × 5 bochten × 3 zijstappen.
4. **Grid:** `gait_grid.npz` (actief) + `grids/gait_grid_v6e.npz`. Grid maken:
   `python ../../../../tools/walk_speed.py --style favorites/jairo_walk_v5c.npz --grid grids/gait_grid_vX.npz`
5. **Training:** mjlab op RunPod (`duck.sh`) → ONNX → `tools/duckling_play_onnx.py`.

**De ONNX heeft de grid nodig** voor de loopklok (`cycle_s`, 75 getallen; op de RP2350 een tabelletje). Altijd met dezelfde grid afspelen/meten als waarmee getraind is.

## Taak (`Mjlab-Velocity-Flat-Duckling`, `MODE=walk`)
- Lopen-policy = **alleen vooruit + draaien + stoppen + hoofd**: `lin_vel_x=(0, 0.12)`, `lin_vel_y=0`, `ang_vel_z=±0.6`.
  Achteruit, zijwaarts, opstaan, koprol = losse policies later, gewisseld via HOME.
- Obs 63 (incl. gait-klok sin/cos). Actieschaal 1.0, gewone MuJoCo-actuators (geen BAM).
- Rewards (v6e): `track_linear_velocity` w4 en `track_angular_velocity` w4, **twee-schaal** (fijn 0.03 m/s + grof 0.10 m/s; fijn 0.25 + grof 0.6 rad/s);
  `walk_imitation` w2 (per gewricht: hip_yaw 1.0, hip_roll 1.0, hip_pitch 1.0, knee 1.0, ankle 0.1); `com_forward` w1.5;
  contact_schedule, swing_clearance, stance_grip, trunk_height, flat_feet, home_pose, stand_still, head_pose_with_style.
- `COM_AHEAD_100` in de config (None = uit de grid-stijl) om het zwaartepunt-doel snel te overrulen.
- Spiegel-symmetrie (mirror loss 0.5), tabel FK-geverifieerd.

## Versies (zips)
| zip | wat | status |
|---|---|---|
| lopen | alleen vooruit + draaien | basis |
| lopen2 | imitatie 4 → 2 | in alles hierna |
| lopen3 | draaien belonen 2 → 4 | in alles hierna |
| grid_v5 | zwaartepunt + teen-afzet op v4b | **vervangen door v6** |
| v6 | jouw nieuwe loopje (v5) als grid + `com_forward` | |
| v6b | zwaartepunt 10 mm bij 100% (v5b) + hip_yaw-imitatie 1.0 | |
| **v6c** | **twee-schaal tracking** → vooruit doorbraak | 1800 = beste tot nu toe |
| v6d | zwaartepunt symmetrisch + hip_roll-imitatie 1.0 | **niet gebruiken zonder v6e** |
| **v6e** | v6d + grid met `sway` 0.7 (heup-roll 16–20° → 10–14°) | nu actief, nog niet gemeten |

## Metingen (gemiddeld over 4 startfases, `meas2.py`)
| | vooruit 4/8/12 (mm/s) | draai L/R (±29 °/s) |
|---|---|---|
| v4b-tijdperk (2000–2850) | 5–21 / 5–14 / 5–38 | ±5–16, vaak de verkeerde kant op |
| v6b 2300 | −4 / −5 / 6 | −7 / −20 |
| **v6c 1800** | **21 / 53 / 77** | 8 / 4 (geen sturen) |

## Lessen (duur betaald)
- **Eén wijziging per run, en meten met een ONNX vóór je meer GPU uitgeeft.**
- **Reken een reward eerst door bij "niets doen / half / goed".** Tracking met std 0.03 m/s gaf 0.000 bij 3 cm/s op een commando van 12 → geen gradiënt → trappelen op de plek, 6 runs lang. Twee-schaal loste het binnen 150 iteraties op.
- Een laag imitatiegewicht per gewricht = dat gewricht is "vrij": de policy gebruikte de heup-yaw ±15° voor balans in plaats van om te sturen.
- Wat de policy fout doet, staat soms al in het voorbeeld: het zijwaarts optillen via heup-roll kwam uit de grid (`sway` 1.0 → heup-roll tot de limiet van 20°).
- `com_forward` alleen is te omzeilen: met mini-stapjes (voet onder het zwaartepunt) scoort hij zonder te lopen. Werkt pas samen met goede tracking.
- Meet over meerdere startfases van de klok: de drift hangt af van welke voet begint (mank looppatroon, het spiegelbeeld is net zo goed).
- **Enkel (`knee`-servo) is de fysieke zwakke plek:** 30–45% van de tijd op de krachtlimiet; statisch op één been al 45–60 mNm van de 80; ~11 rad/s bij 100%. De sim kent geen koppel-snelheidscurve → in het echt krapper. Opties: veer in de enkel, andere houding, sterkere servo.
- Curricula herstarten niet bij resume. `resume` pakt de nieuwst gewijzigde run-map → voor een oud checkpoint: nieuwe map + `model_X.pt` erin.
- `.gitignore` sluit `*.npz` uit → `git add -f`.

## Commando's
Pod:
```
export PATH="$HOME/.local/bin:$PATH"
cd /workspace && python3 -m zipfile -e duck_basis.zip .     # geen unzip nodig
cp microduck_rl/duck.sh . && chmod +x duck.sh && ./duck.sh setup
./duck.sh update /workspace/<wijziging>.zip
ENVS=8192 ./duck.sh resume 1000      # 1000 iteraties extra vanaf het nieuwste checkpoint
./duck.sh export [iter]              # → /workspace/duckling_walk_<iter>.onnx
./duck.sh status / stop
```
Vanaf een ouder checkpoint:
```
R=/workspace/microduck_rl/logs/rsl_rl/duckling_velocity
mkdir -p $R/nieuw && cp <pad>/model_1500.pt $R/nieuw/ && ENVS=8192 ./duck.sh resume 1000
```
Lokaal (vanuit `src/mjlab_microduck/robot/duckling`):
```
python ../../../../tools/duckling_play_onnx.py duckling_walk_XXXX.onnx --grid grids/gait_grid_v6e.npz   # --geen-rust, --mooi
PYTHONPATH=../../../../tools python ../../../../tools/meas2.py duckling_walk_XXXX.onnx grids/gait_grid_v6e.npz
python ../../../../tools/walk_speed.py --load grids/gait_grid_v6e.npz        # voorbeeld bekijken
python ../../../../tools/gait_gen.py --view                                 # loopje aanpassen
```
Verse basis-zip maken op de pod (zonder logs):
```
cd /workspace && python3 -c "import zipfile,os;z=zipfile.ZipFile('duck_basis.zip','w',zipfile.ZIP_DEFLATED);[z.write(os.path.join(r,f)) for r,d,fs in os.walk('microduck_rl') if '/logs' not in r and '.venv' not in r and '__pycache__' not in r for f in fs]"
```

## Plan
1. v6e ~250 iteraties → meten: heup-roll (moet ~10–14° worden), vooruit houden, sturen.
2. Beste checkpoint vastleggen (pt + onnx + config + grid + loopje).
3. **Robot (deels) bouwen.** Eerst de enkelservo testen op één been met vol gewicht; servo's meten (stapresponsie, vertraging, kracht) en in het model zetten.
4. Daarna: sturen (draai-patroon in de grid is zwak: heup-yaw ±8° bij draaien tegen ±3° rechtuit), toestandsmachine,
   losse policies (achteruit, opstaan, koprol), snavel als verstoring in de training, servo-veilige stijl.
