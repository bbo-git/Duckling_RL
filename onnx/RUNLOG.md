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
- track_linear_velocity in pasjes: std 0.006/0.02 m/s (was 0.03/0.10: stilstaan bij 9 % kreeg 93 % van de beloning).
- 150 (gemeten): zwaaivoet gaat met heup-roll (+0.45) 6 mm naar buiten (breedte 49 -> 60 mm), nauwelijks vooruit; Jairo: "meer
  opzij dan naar voren, blijft haken". -> stance_width in pasjes: max 52 mm (HOME 49 + 3), straf -400 (was 56 mm, -200). Hervat vanaf 150.
- Pasjes om en om: een nieuwe reeks begint altijd met de andere voet dan de vorige eindigde (training; firmware/player volgen).
- 200 (v4): om en om ok, maar schuifel. Rooster v4 nagerekend: bij het leunen kantelen de zolen mee (geen enkel-roll),
  standvoet 2.7-2.9 mm met de rand "in de vloer", zwaaivoet maar ~0.25 s echt los. CONCLUSIE: quasi-statisch kan deze
  eend niet. Zool plat = heup recht boven de voet (HOME, 49 mm breed); gewicht kan dan maar ~5 mm opzij (romp/kop),
  zoolrand zit 11 mm naast het midden. Kort slingertje (~0.1 s): een pas moet dynamisch, ~0.25-0.3 s.

## 2026-11-26 ~09-40 — DYNAMISCHE PASJES v5 (nieuwe run vanaf 0)
- Jairo: "elke stap een element van de echte walk: gewicht opzij en naar voren, balans pakken door de voet naar voren te
  zetten, en weer stoppen". Rust zit TUSSEN de pasjes, niet erin.
- Rooster gait_grid_pasje_v5.npz (walk_speed.py --pasje-dyn): het loopje normaal_v3 (cadans 0.5 s, LIPM-zwaai), maar tot
  45 % op HOME-hoogte (kniesprong bij de start 22 -> 11 gr), voet 8 mm op, geen afzet, minder doorveren.
- DUCK_ROOSTER=dyn (standaard; pasje = v4, oud = normaal_v3). Pasjes 20-30 % (4.5-6.7 mm), 400: tot 45 %, 800: 8 passen tot 60 %.
  residual_l2 -1 (balans vraagt bijsturen), snelheid std 10/30 mm/s, swing_clearance 6 (doel 6.4 mm), om en om, rust 0.5-1.5 s.
- Opgelost sinds de eerste pasjes-run (schok op de eend): kniesprong bij de start, "lopen"-drempel 0.02 m/s (20 % telde als
  stilstaan), firmware startte midden in een halve cyclus.
- 100 (v5): reageert op duwtjes, maar zwaartepunt eerder naar achteren dan voren. com_forward in dyn: doel 3 mm voor de standvoeten bij lopen (was ~0 onder 20-30 %), voortgangsdrempel 0.02 -> 0.004 m/s. Hervat vanaf 100.
- 200 (v5): komt vooruit (38 mm/10 s), maar hupjes (beide voeten bijna tegelijk los), soms dubbel dezelfde voet, gyro 3.2 rad/s.
  Jairo: "het mag langzamer en rustiger".

## 2026-11-26 ~10-00 — DYN-PASJES v6 (nieuwe run vanaf 0)
- Rooster v6: als v5, maar cyclus 0.75 s tot 45 % (0.37 s per pas, was 0.25). Pas 6.7 mm bij 20 %, 10 mm bij 30 %, 13 mm bij 40 %.
  Knie max 3.5 rad/s, romp ~9 mm opzij.
- Nieuw in dyn: trunk_calm -0.5 (romp-roll/pitch-snelheid^2), both_feet_air -3 (geen hupjes); com_forward 3 mm voor de standvoet.
- Curriculum: 20-30 %, 400: tot 40 %, 800: 8 passen tot 50 %.
- 100 (v6): rustig (gyro 1.25, was 3.2), 31 mm vooruit, om en om binnen een reeks; bij het stoppen nog 2 extra stapjes.
  Jairo: "nog langzamer net iets beter".
- v7: cyclus 0.9 s (0.45 s per pas), pas 8 mm bij 20 %, 12 mm bij 30 %, knie max 2.8 rad/s, romp ~10 mm opzij. Hervat vanaf v6.
- Tempo is nu een parameter: walk_speed.py --pasje-dyn --pas-cyclus <s> --grid grids/<naam>.npz; training DUCK_GRID=<naam>.npz;
  player --rooster grids/<naam>.npz.
- 200 (v6->v7): Jairo: "hij moet echt een voorwaartse beweging maken met gewicht en teen-afzet". v8: teen-afzet 10 gr en zwaartepunt 4 mm voor de standvoet ook bij kleine pasjes (was 0 onder 20 %); com_forward-doel 4 mm. Enkel max 3 rad/s.
- com_forward 2 -> 5 in dyn.
- 300 (v8): zwaartepunt +2 mm voor (doel 4), 0 hupjes, voet 3.5x los per pasje (schuifel), 60 mm/10 s, romp-roll +-7.5 gr. Jairo: te veel zijwaartse heupzwaai. v9: waggel 30 % (romp-roll in het rooster 3.7 -> 1.3 gr).
- 500 (v9): zwaartepunt +5..6 mm voor (doel gehaald), 95-113 mm/10 s, stil na het pasje. Maar zwaai 0.12 s (rooster 0.4 s), voet 11 mm op, klap bij neerzetten (romp -107 mm/s) en dezelfde voet nog eens op ('double bounce'). -> soft_landing -20 (neerwaartse voetsnelheid^2 onder 4 mm), contact_schedule 4 -> 8. Hervat vanaf 500.
- Jairo: "met heup-roll balans zijwaarts, met de tenen voorwaarts: zichzelf een klein stukje naar voren laten kantelen" (500 kantelde bij elke pas achterover, ~1.2 rad/s). v10: romp 3 gr voorover, zwaartepunt 6 mm voor de standvoet; com_forward-doel 5 mm. Hervat.
- Run v9 150-250: tikt dezelfde voet 2-4x per pasje (voet 3 mm op), "nare dubbele voetsprong"; Jairo: 250 echt niet goed.

## 2026-11-26 ~11-05 — DYN-PASJES v10 + SCHONE ZWAAI (nieuwe run vanaf 0)
- Rooster v10: romp 3 gr voorover, zwaartepunt 6 mm voor de standvoet (Jairo: "met zijn tenen zichzelf naar voren laten
  hellen"), 0.45 s per pas, waggel 30 %, teen-afzet 10 gr, voet 8 mm op.
- swing_clearance -> swing_clean_lift (6): optillen telt alleen zolang de voet in deze zwaai nog niet opnieuw de grond raakte.
  swing_taps -5 per tikje. soft_landing -20, contact_schedule 8, both_feet_air -3, trunk_calm -0.5, com_forward 5 (doel 5 mm).
- Voetbreedte max = loopje + 2 mm (Jairo: "geen zijwaartse stap").
- 250 (sim en echt gelijk): rustig schuifelen, niet gevallen, nauwelijks vooruit, voet ~1 mm, zwaartepunt -1.3 mm. Te veel straffen -> niets doen is het veiligst. Versoepeld: soft_landing -20 -> -5, swing_taps -5 -> -2, trunk_calm -0.5 -> -0.2, residual_l2 -1 -> -0.5; beloningen swing_clean_lift 6 -> 10, track_linear_velocity 4 -> 6. Hervat vanaf 250.
- v10-run 450: geen echte pas (voet 3 mm, 1-2 mm vooruit). Patroon vandaag: 0.25 s/pas = echte (felle) stappen, 0.37 = redelijk,
  0.45 = schuifelen, ook met steeds meer beloningsaanpassingen.

## 2026-11-26 ~12-05 — DYN-PASJES v11 (nieuwe run vanaf 0)
- Rooster v11 = v10 met cyclus 0.6 s (0.3 s per pas): pas 5.4 mm bij 20 %, 8 mm bij 30 %; knie 3.4, enkel 4.5 rad/s; romp 8 mm opzij.
  Met com_ref (vallen in de pas), rechtop in rust (HOME).
- Beloningen dyn: com_track 5, swing_clean_lift 10, contact_schedule 8 (te vroeg neergezet mag blijven staan), both_feet_air -3,
  track_linear_velocity 6, residual_l2 -0.5, voetbreedte +2 mm. ERUIT: trunk_calm, soft_landing, swing_taps (maakten hem passief).
- v11-run gestopt op 150 (Jairo: jerky stap). Pod uit.
- Firmware (netwerk 250, v10): 'rondkijken aan' = kop kijkt rond (yaw 45, pitch 0.3), langzame gewichtsverplaatsing,
  zwaaien (camera, intensiteit >= 2) -> kijken + kopje scheef + snavel. Jairo: leuk; viel aan het eind wel om.

## VOLGENDE KEER: eerst een goede stap ontwerpen (Jairo, 26 nov)
1. Stap samen ontwerpen in de viewer (walk_speed --pasje-dyn --view / --load), Jairo keurt op het oog: gewicht opzij (heup-roll)
   en naar voren (tenen), voet op, neerzetten, opvangen, stoppen. Tempo ~0.3 s per pas (sneller = fel, trager = schuifelen).
2. Fysica-check zonder netwerk (lift.py / meet.py): voet echt los, zool plat, geen zoolrand in de vloer, zwaartepunt-verloop.
3. Pas dan EEN training, met een klein vast beloningspakket; meten bij 150/300 met dezelfde meetlat (voet los per pas = 1,
   voethoogte, vooruit per pas, zwaartepunt t.o.v. de standvoet, romp-gyro). Niet elke run iets anders.

## 2026-11-26 middag — STAP-ROOSTER (ontworpen met Jairo in de viewer)
- tools/stap_gen.py: pas uit (bijna) gebalanceerde standen met platte zolen: leunen via heup-roll (zwaartepunt ~10 mm opzij,
  net binnen de zoolrand), vloeiend afzetten (hiel op over de teen) en zwaaien (0.24 s, voet 20 mm in het ontwerp, ~10 echt),
  teen op en landen op de hak, afrollen, terug naar het midden, 1 s pauze. Om en om. Zonder netwerk (servo 1.6x): valt niet,
  geen bounce; na het landen komt de standvoet nog ~0.08 s los (gewicht schiet door) - aan het netwerk.
  Hiel-landing + stoppen kan alleen als het zwaartepunt bij het landen niet voor de voet staat; afrollen van de achterste voet
  na het stoppen valt open-loop om (opties stand_rol / stand_duw / breed staan uit).
- tools/stap_speler.py: open-loop afspelen met fysica (servo 1.6x standaard). tools/stap_grid.py: naar trainingsrooster
  grids/gait_grid_stap_v1.npz (alle vooruit-opdrachten = deze pas, 4.5 mm/s = 5 %; snelheidsknoop 0.02; walk_s0 0.02; com_ref).
- Training (DUCK_GRID standaard stap_v1): pasjes 1-2 per keer op 5 %, 400: tot 4, 800: tot 8. Rest van de beloningen = v11.

## 26 nov avond - stap-rooster v2
- Echte eend (netwerk 200, stap v1) viel bij de eerste pas naar rechts: heup-roll stond al op de grens van 20 gr (het ontwerp gebruikte de hele uitslag), geen bijsturen meer mogelijk.
- v2: zwaartepunt 4 mm opzij (was 10,5) -> heup-roll max 8 gr, romp-roll ~11 gr (was ~22, Jairo: "de helft of minder hellen"). Open lus staat bij 1.0x en 1.6x.
- Training: vertraging netwerk->servo 0/20/40/60 ms (25/35/25/15 %), was 0/20/40.
- CORRECTIE (zelfde avond): v2 stuiterde in de player (Jairo). Mijn test had een oude kopie van stap_gen gebruikt. Met de juiste: v2 valt bij 1.0x en heeft 16 voet-los i.p.v. 8. Alleen v1 (10,5 mm, heup-roll 20) is schoon. Kop-roll helpt niet voor het zwaartepunt (getest), heup-roll max 20 (Jairo). Terug naar v1 als standaard; vertraging tot 60 ms blijft.
- v3 (Jairo: statische balans vergeten, rustige stapjes in dynamische balans): com_y 3 mm, zwaai 0,15 s, lift 16 mm, fall_y 0.8. Open lus: staat bij 0.7-2.0x, schoon (2x voet-los per pas, zoals v1), romp-roll ~5 gr (v1 ~20), heup-roll 6 gr, ~9 mm vooruit per pas. Gezocht over 104 varianten.
- Run v3: 100 echt niet gevallen (romp +-5 gr, knie ~10). 250 echt niet gevallen maar wiegt: heup-roll +-12 (ontwerp 6), romp -10..+13, knie niet meer. Nieuwe run (Jairo: opnieuw) met hip_roll_residual -3 (alleen heup-roll-residu).

## VOLGENDE KEER (27 nov): doorlopend loopje
- Jairo: verder met een doorlopend loopje in een vast ritme (natuurlijker voor deze maat: valtijd ~80 ms, servo 40-60 ms achter).
- Uitgangspunt: stap v3 (3 mm opzij, zwaai 0,15 s, romp ~5 gr) zonder pauzes: hold_s 0, back_s kort; cadans zoeken in de sim (open lus, 0.7-2.0x), daarna rooster met snelheid als parameter.
- Training: hip_roll_residual -3 en vertraging tot 60 ms houden. Laatste UF2: netwerk 150 (stap v3 + heup-roll-straf), niet gevallen.
- Grip: dun rubber/antislip onder de zolen (geen dik foam); voet slipt in yaw bij afzetten.
- Echt, 150 (nieuwe run met heup-roll-straf): 2x oefen zonder val; heup-roll +-7 (ontwerp 6), romp -6..+7,5, knie tot +9 (ontwerp +13). 250 van de vorige run viel na een paar oefeningen (35 gr).

## 27 nov - loopje v1 (stap v3 doorlopend)
- Rooster gait_grid_stap_loop_v1.npz: hold 0, back 0,25, shift 0,35, zwaai 0,15 -> 0,72 s per pas, ~10 mm per pas. Open lus 12 passen schoon bij 1.0-2.0x (valt bij 0.7x).
- Schema: 1-6 passen, vanaf 300 tot 12, vanaf 600 tot 24, met staan ertussen. Heup-roll-straf -3 en vertraging tot 60 ms blijven.
- Kop-roll deels vrij (Jairo: trillingen opvangen): fout op kop-roll telt x0.25 in head_pose_tracking en head_level (DUCK_KOP_ROLL_VRIJ). Residu-bereik kop blijft 0.2 rad (~11 gr).
- Nek- en kop-pitch half vrij (Jairo): fout x0.5 in head_pose_tracking, head_calm -0.025 (DUCK_KOP_PITCH_VRIJ). Let op terugkeer van het knikken (run 25 nov: 30-38 gr per pas).
- DOEL LATER (Jairo: loop_v1/v2 is nog geen vloeiende loop): nieuw ontwerp i.p.v. losse fasen - doorlopende heupgolf (niet terug naar het midden), zwaai overlapt met gewichtsverplaatsing, cadans op de eigen wiegfrequentie (zoals de microduck).
- Loop-run (27 nov) 250 in de sim: heup-roll +-19 (ontwerp 6), romp +-20, maar 54 mm vooruit in 20 s. Straf -3 te zwak. Nu harde grens: residu-schaal heup-roll 0.12 (~7 gr) i.p.v. 0.3 (training, player, firmware). Oude netwerken: player DUCK_RES_HEUPROL=0.3, firmware --res-heuprol 0.3.
- Teen-afrol standbeen (stand_duw 8/15) en zwaartepunt verder voor in het ontwerp: open lus slechter of valt -> niet in het ontwerp; vooruit moet het netwerk leren.
- Vloeiend ontwerp (tools/loop_gen.py, EXPERIMENTEEL): doorlopende sinus opzij, constante snelheid vooruit, korte dubbele steun, aanloop vanuit stilstand. Open lus valt in ~50 varianten binnen 2-4 passen (voor- of achterover; zijwaarts loopt de eend uit de maat met het ontwerp). Nog niet bruikbaar als voorbeeld.
- Jairo kiest: trainen op het vloeiende ontwerp (met instelbaar aantal passen). Rooster gait_grid_stap_loop_vloei_v1.npz (loop_gen Th 0,4 s, ds 0,4, com_x 2 mm, A 3 mm, L 10 mm; ~25 mm/s). Open lus valt na ~2 s - het netwerk moet voor/achter bijsturen. Standaard in training en player. Kijken: stap_speler.py tools/loop_frames_vloei.npz (met aanloop).
- vloei_v2 (Jairo: v1 viel direct, niet te beoordelen): kleinere teen/hiel-hoeken (afzet 6, hiel 0, teen 10), com_x 1,5 mm. Open lus 24 passen staan bij 0.85-2.0x, romp +-6, pitch +-5, ~10 cm. Standaard in training en player.
- Rondkijken in rust (Jairo: pasjes + stilstaan + kijken in een netwerk): kijkopdrachten nu ook bij stilstand in observatie en beloning (DUCK_KIJK_RUST=1), kop-curriculum 15/40/70/100 % op 0/150/300/450. Firmware: rondkijken gaat als kijkopdracht naar het netwerk (KIJK_RUST, maak_firmware_data --kijk-rust 0 voor oude netwerken). Player idem (DUCK_KIJK_RUST=0 voor oude).
- LATER (Jairo): telefoonbediening loop/kijk/stop. Plan: webpagina met knoppen op de ESP32-CAM (wifi staat al aan) + Siri via Opdrachten (HTTP); commando via CamPacket naar de RP2350. Geen BLE (iPhone: geen Web Bluetooth).
- FOUT gevonden (Jairo: 200 van de vloei-run loopt achteruit): de snelheidsopdracht in de pasjes-training stond vast op 5 % (= stap v3), maar loop_v1 gaat 15,5 %, loop_v2 16,4 %, vloei_v2 27,7 %. track_linear_velocity beloonde dus langzamer lopen dan het voorbeeld -> achteruit/niet vooruit. Nu: opdracht = stap_v/v100 uit het rooster. Op de eend/player voor vloei_v2: snelheid 0.28 (oefen 3 4 0.28 1.5).
- Gebibber (Jairo, echte eend, 200 van de loop-run): been-acties springen ~0.12 per 20 ms (enkels/knieen); in de sim zelfs 0.3 -> eigenschap van het netwerk, geen sim-to-real. Nieuw: walk_action_rate -0.3 (DUCK_GLAD) op de been-acties tijdens het lopen.
- Dubbele stap zit in het vloeiende VOORBEELD (open lus ~3x voet los per pas bij 0,4-0,5 s per pas; de eend wiegt van nature ~0,28 s per pas en wiegt in de lange dubbele steun terug). 250 met opdracht 0.05: geen dubbele stap maar niet vooruit; met 0.28: vooruit maar dubbele stap -> het netwerk kan het. Nu: snelheidsopdracht 0.25-1.0 x het rooster (DUCK_SNELHEID_MIN), zelfde ritme = snelheidsknop.

## KEURING (27 nov) - tools/keuring.py, vaste maat voor elk voorbeeld en netwerk
Voorbeelden (open lus, servo 1.0x/1.6x; "buiten zwaai" = voet los terwijl hij moet staan = dubbele stap):
- stap_v3 (losse pasjes): staat, 2.0 los/pas waarvan 1.0 buiten zwaai, 94/86 % van ontwerpsnelheid
- stap_loop_v2: staat, 2.0 / 1.0 buiten, 52/49 %
- vloei_v2: staat, 3.0 / 2.0 buiten, 40/29 %
- vloei_v3: staat, 2.9-3.5 / 1.9-2.5 buiten, 49/46 %
=> GEEN enkel voorbeeld is schoon; alle hebben minstens 1 extra voet-los per pas.
Netwerk 250 (vloei-run, op vloei_v2, 40 ms vertraging): staat overal; hoog (0.277): 8 mm/s van 25, 1.5 buiten zwaai/pas, heup-roll tot +13, romp +-15, trillen 0.15; laag (0.069): achteruit, rust-schuiven 16-29 mm.
