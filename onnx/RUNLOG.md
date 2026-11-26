# Runlogboek loop-netwerk

Per run: wat er veranderd is t.o.v. de vorige, en de keuring (`tools/keuring.py`, zie `KEURING.md`).
Regel: één verandering per run waar het kan.

## 2026-11-01 — de "1150"-run (oude referentie)
- Twee stijlen, gedragsmodel (50 %), kop kijkt rond, achteruit. Servo-variatie kracht 85-115 %, geen vertraging.
- Hervat bij 950 met heup-yaw-imitatie 1 -> 3 (tenen naar binnen).
- Keuring 1150: staat stil (-2 mm), loopt (91 % / 64 %), heup-yaw ok; draaien FOUT (-1 gr/s), sterke servo's FOUT (-10 %).
- Op de echte eend: spartelt direct en valt (18 nov). Vermoeden: servo's sterker/stijver dan in de simulatie.

## 2026-11-19 06-49 — "vereenvoudigd" (onbruikbaar)
- T.o.v. 1150: servo-variatie kracht 70-180 %, stijfheid 70-200 %, snelheid 80-150 %, vertraging 0-40 ms;
  één stijl, geen gedragsmodel, geen achteruit.
- Gevolg: X-poten (heup-yaw -11 gr bij 800 -> -20 gr bij 1500) en wegschuiven bij staan (15-28 cm).
  Oorzaken: feet_parallel te smal (geen gradient), toe_out uit; stilstaan maar 2-15 % van de envs.

## 2026-11-25 05-11 — oranje (8192 envs, vanaf 0)
- T.o.v. 19 nov: feet_parallel breed + toe_out (naar binnen) aan; stilsta-envs 25-30 %; accu op de echte plek.
- Keuring 300: voeten recht (-2 gr), maar staat niet stil (-128 mm), loopt niet (-6 %), draait niet.
- Tensorboard: imitatie piek 0.82 @110, daarna -> 0.47 @500; alive idem (valt vaker); stand_still ~1e-5 (nooit verdiend).

## 2026-11-25 06-04 — paars (hervat 450, strengere straffen) — gestopt
- target_rate -2 -> -3, target_jerk -1 -> -4, stand_action_rate -2: imitatie stortte in (0.28 -> 0.05). Teruggedraaid.

## 2026-11-25 ~07-20 — oranje hervat vanaf 450
- Enige verandering: stand_still half streng / half ruim (was alleen streng, verdiende ~0).
- Bij 600 (Jairo): veel te druk, te veel kop, te weinig heup. Gemeten (loop 0.5): nek 38 gr / kop 30 gr
  per pas (loopje 7 / 0), heup-pitch 21-23 gr (loopje 31).

## 2026-11-25 ~07-45 — hervat vanaf 600
- Twee veranderingen: head_calm (straf op nek/kop-pitch-snelheid, -0.05) en hip_pitch-imitatie 3 -> 5.
- Keuring 650: staat stil (-1 mm) maar draait rondjes in rust (-7 gr/s), loop 0.5 33 %, heup-yaw -7..-9 gr,
  nek knikt nog 1 gr (was 38), onrust 16 gr/stap (was 21-23); 1.6x servo's: achteruit. 750: kop vol omlaag.
- Op de eend (650, in de hand = geen grond): wild. Netwerkacties ~2-4 rad, net als in de simulatie.

## 2026-11-26 — RESIDU-ACTIES (nieuwe run vanaf 0)
- Servodoel = jouw loopje (gangrooster, opdracht + fase; kop = kijkopdracht + knik) + 0.30 rad x uitvoer
  (kop 0.20). Was: HOME + uitvoer, doelen tot +-4 rad = aan/uit-sturing tegen de grenzen.
- Het loopje alleen (uitvoer 0, open-loop in de simulatie): staat, 50 % -> op de plaats, 100 % -> +195 mm in 8 s.
- Player en keuring: --residu. Firmware: volgt als de training werkt.
- Fase "pasjes" (DUCK_FASE, standaard): 20-50 % vooruit (pas 4.5-11 mm, loopje ~4 passen/s), start vanuit HOME 70 %, duwtjes 0.1 m/s, niet draaien, opdracht elke 1.5-4 s, 40 % stilstaan.
  Daarna fase "lopen" (DUCK_FASE=lopen ... resume): 0-100 % + draaien.
- Firmware: `passen <n> [f]` voor losse pasjes op de eend.
- Pasjes-schema (vervangt willekeurige opdrachten in fase pasjes): staan 1-2.5 s -> wachten op halve gangcyclus ->
  precies n passen op f -> staan. Curriculum: it 0: 1 pas, <=25 %; 300: 1-2, <=30 %; 600: 1-4, <=40 %; 900: 1-8, <=50 %.
- Keuring 300: staan 0 mm, na stop 0 mm/1 gr, loop 0.3 57 %, 0.7x 39 %. Op de eend: viel niet, maar "meer geschuifel".
- Keuring 600: staan 0 mm, na stop 0 mm/0 gr, loop 0.3 59 % (net FOUT), 1.6x 52 %, 0.7x 49 %, onrust 6.6 gr/stap.
- Log 300 op de eend (`oefen 3`): elk pasje 0.25 s, gyro-piek 2.5-2.8 rad/s = de schok; geen gewichtsverplaatsing voor de stap.

## 2026-11-26 ~08-30 — RUSTIGE PASJES (hervat vanaf 600, nieuw rooster)
- Jairo: "gewicht op links, pas maken, gewicht weer verdelen, op een rustige manier".
- Nieuw rooster grids/gait_grid_pasje_v1.npz (walk_speed.py --pasje --style favorites/normaal_v3.json):
  tot 30 % quasi-statisch: romp eerst over de standvoet (80 %), dan voet op (6 mm), neerzetten, gewicht terug naar
  het midden. Fase 0 / 0.5 = gewicht verdeeld (midden dubbele steun). Cadans 2 s/cyclus (1 s per pasje) bij 6 %,
  1 s bij 30 %; dubbele steun 30 -> 15 %; hoogte en voetbreedte = HOME (geen sprong bij de start); geen waggel,
  geen afzet, kop rustiger. Snelheden 6/9/12/16/20/30/50/75/100 %; onder 5 % mengt naar stilstaan (was 15 %).
  Pasje 6 % = 5 mm, 9 % = 8 mm, 12 % = 10 mm, 20 % = 13 mm. Vanaf 50 % het oude loopje.
- Open-loop in de simulatie (uitvoer 0): pasje 6-12 % staat, gyro-piek 1.1-1.8 rad/s (oud rooster 25 %: 2.2-3.8).
- "Lopen"-drempel 0.02 -> 0.004 m/s overal (training _moving, firmware rust/kop, player): 0.02 was 22 % van v100,
  dus pasjes op 20 % telden als STILSTAAN (stand_still-beloning, geen contactschema, rust-modus op de eend).
- Pasjes-curriculum: it 600-900: 1 pas 6-9 %; 900: 1-2, <=12 %; 1200: 1-4, <=16 %; 1500: 1-8, <=20 %.
- Oud rooster terug: DUCK_ROOSTER=oud (training), --oud (player).
- v1 (romp 20 mm opzij) leek op "skien op een steile helling" (Jairo: eerder 5 mm). -> gait_grid_pasje_v2.npz: frac 0.25
  = ~6 mm opzij. Open-loop: 6-12 % staat, kanteling max 8.5 gr, gyro 1.3-2.0 rad/s. Training gebruikt v2.
- v2 (6 mm) is niet quasi-statisch: binnenrand zool zit 11 mm naast het midden. Zonder enkel-roll kantelt de zool
  als de romp alleen opzij schuift -> v3: romp ~11 mm opzij + licht leunen naar de standvoet (waddle 0.5).
  Open-loop: 6-12 % staat, kanteling max 6.3-7.2 gr (beste tot nu), gyro 1.3-1.8; vanaf 16 % valt hij zonder netwerk.
  Training gebruikt gait_grid_pasje_v3.npz.
- 650/750 (v3): verplaatst gewicht, maar tilt de voet niet op ("klein schuifje"). Open-loop bleek: met 11 mm opzij
  komt de voet maar ~3 mm los, ook met 10 mm opgetild in het rooster - het zwaartepunt staat nog op de rand van de zool.
- v4: romp ~16 mm opzij + 8 gr leunen, voet 10 mm op. Open-loop: voet 7 mm los, 11-16 mm vooruit per pasje, kanteling
  max 11 gr (incl. het leunen). swing_clearance 3 -> 6, doel 8 mm. Hervat vanaf 750.
- 850/950 (v4, hervat vanaf 750): voet blijft aan de grond (gemeten 850: voethoogteverschil 0.9 mm, rooster zonder
  netwerk 4-6 mm; |actie| 0.1). Het netwerk stuurt het optillen tegen.

## 2026-11-26 ~09-05 — PASJES V4 VANAF 0 (nieuwe run)
- Geen oude gewoontes (600 = snelle pasjes, 850 = voet vasthouden): residu begint op 0 = precies het pasje-rooster v4.
- residual_l2 -4 (benen dicht bij het rooster), 2-4 pasjes achter elkaar op 6-9 %, rust 0.5-1.5 s.
  Curriculum: 1250: <=12 %; 1500: 8 pasjes <=16 %; 1800: <=20 %.
- Player en firmware: netwerk standaard ook aan in rust.
