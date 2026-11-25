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
- Keuring: (volgt bij 650)
