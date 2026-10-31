# ijk.py - servo's van de Duckling ijken, voorzichtig en per gewricht (MicroPython, RP2350).
#
# In Thonny: open dit bestand, Run (F5). Daarna typ je commando's in de Shell onderaan.
# De ijking wordt bewaard in calib.json op het bordje (en elke opslag ook in calib_log.txt).
#
# VEILIGHEID
#  * Bij de start staan ALLE servo's UIT (geen signaal = slap). Je zet ze een voor een aan.
#  * Elke beweging gaat geleidelijk (max ~60 gr/s servo-as), stappen standaard 1 gr.
#  * Offsets en hoeken zijn begrensd (zie MAX_OFFSET en het bereik per gewricht).
#  * 'uit' of 'q' maakt alles weer slap.
#
# BEGRIPPEN
#  offset  = servohoek (gr) waarbij het gewricht in HOME staat (de CAD-nulstand).
#  teken   = +1 of -1: welke kant de servo op draait bij een positieve gewrichtshoek.
#  differentieel (heup, kop): servo A = 1.5*(uit + drager), servo B = 1.5*(uit - drager)
#                             heup: uit = pitch, drager = roll; kop: uit = yaw, drager = roll.
#
# COMMANDO'S (Enter na elk commando)
#  lijst                 alle gewrichten en servo's met hun ijking
#  kies <naam>           gewricht kiezen, bv. kies L_heup_pitch  (of een losse servo, bv. L_diffA)
#  aan [naam]            servo('s) van het gekozen (of genoemde) gewricht aan -> HOME; 'aan alles'
#  uit [naam]            slap (zonder naam: alles)
#  + / -                 gekozen gewricht een stap bijstellen (= IJKEN: HOME schuift mee!)
#  naar <gr>             gekozen gewricht naar <gr> t.o.v. HOME, ijking blijft ('thuis' = terug)
#  herstel               niet-bewaarde bijstellingen weggooien (terug naar calib.json)
#  ++ / --               5 stappen
#  stap <gr>             stapgrootte (standaard 1, bv. stap 0.5)
#  test <gr> [s] [n]     gekozen gewricht langzaam heen en weer rond HOME: +gr en -gr, n keer
#                        in s seconden (standaard 5 s, 3 keer). Bv. test 5  of  test 10 8 2
#  test alles <gr> [s] [n]  ALLE aangezette gewrichten tegelijk, zelfde beweging; bij een
#                           differentieel worden pitch en roll per servo opgeteld (been gaat schuin)
#  test soort <gr> [s] [n]  per soort tegelijk: eerst alle pitch, dan roll, dan yaw, dan de rest
#  test elk <gr> [s] [n]    idem, een voor een
#  draai                 richting van het gekozen gewricht omdraaien (als 'test' verkeerd ging)
#  spiegel [R|L]         yaw en roll van die kant (standaard R) omdraaien, in een keer
#  thuis [naam|alles]    naar HOME: alles wat aan staat (met naam/alles: die ook aanzetten)
#  bewaar                calib.json opslaan (+ regel in calib_log.txt)
#  toon                  ijking als JSON (voor de repo: firmware/calib.json)
#  q                     alles slap en stoppen
#
# Differentiëlen (heup pitch/roll, kop yaw/roll) ijk je per RICHTING: L_heup_pitch, L_heup_roll,
# R_heup_pitch, R_heup_roll, kop_yaw, kop_roll. Het script rekent het om naar servo A en B:
#   A = 1.5*(sp*uit + sr*drager),  B = 1.5*(sp*uit - sr*drager)
# 'draai' bij pitch/yaw keert sp om, bij roll sr (dat is hetzelfde als A en B verwisselen).
import json, math, time
from machine import Pin, PWM

US_MID = 1500.0        # us bij servohoek 0
US_PER_DEG = 10.0      # us per graad servo-as (CONTROLEREN: zet 'test 45' en meet de echte hoek)
US_MIN, US_MAX = 600, 2400
MAX_OFFSET = 30.0      # gr: groter is vrijwel zeker een verkeerd gemonteerde hoorn/tandwiel
MAX_SPEED = 60.0       # gr/s servo-as bij bewegen
RATIO = 1.5            # differentieel 24:16

# GPIO-indeling uit OVERDRACHT.md (2026-10-30). Pas aan als je anders hebt gesoldeerd.
SERVOS = {             # naam: gpio
    "L_diffA": 0, "L_diffB": 1, "R_diffA": 2, "R_diffB": 3,
    "kop_diffA": 4, "kop_diffB": 5, "L_yaw": 6, "R_yaw": 7,
    "L_knie": 8, "R_knie": 9, "L_enkel": 10, "R_enkel": 11,
    "nek": 12, "kop_pitch": 13, "kaak": 14,
}
DIFFS = {   # naam: (servo A, servo B)
    "L_heup": ("L_diffA", "L_diffB"), "R_heup": ("R_diffA", "R_diffB"), "kop": ("kop_diffA", "kop_diffB"),
}
# gewrichten: naam -> ("enkel", servo, bereik) of ("diff", differentieel, "p"|"r", bereik)
JOINTS = {
    "L_heup_pitch": ("diff", "L_heup", "p", (-60, 60)), "L_heup_roll": ("diff", "L_heup", "r", (-20, 20)),
    "R_heup_pitch": ("diff", "R_heup", "p", (-60, 60)), "R_heup_roll": ("diff", "R_heup", "r", (-20, 20)),
    "kop_yaw": ("diff", "kop", "p", (-90, 90)), "kop_roll": ("diff", "kop", "r", (-20, 20)),
    "L_heup_yaw": ("enkel", "L_yaw", (-20, 20)), "R_heup_yaw": ("enkel", "R_yaw", (-20, 20)),
    "L_knie": ("enkel", "L_knie", (-50, 90)), "R_knie": ("enkel", "R_knie", (-50, 90)),
    "L_enkel": ("enkel", "L_enkel", (-60, 90)), "R_enkel": ("enkel", "R_enkel", (-60, 90)),
    "nek": ("enkel", "nek", (-30, 30)), "kop_pitch": ("enkel", "kop_pitch", (-30, 30)),
    "kaak": ("enkel", "kaak", (0, 30)),
}
CALIB = "calib.json"


def load():
    c = {n: {"offset": 0.0, "teken": 1} for n in SERVOS}
    for dn in DIFFS:
        c[dn] = {"sp": 1, "sr": 1}
    try:
        with open(CALIB) as f:
            for n, v in json.load(f).items():
                if n in c:
                    c[n].update(v)
        print("ijking geladen uit", CALIB)
    except OSError:
        print("nog geen", CALIB, "- alles op offset 0")
    return c


cal = load()
pwm = {}               # naam -> PWM (alleen servo's die aan staan)
pos = {}               # naam -> huidige servohoek (gr, t.o.v. offset)
sel = None
step = 1.0
actief = []            # gewrichten die met 'aan'/'thuis' zijn aangezet (voor 'test alles')


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def write(n, servo_deg):
    us = clamp(US_MID + cal[n]["teken"] * (cal[n]["offset"] + servo_deg) * US_PER_DEG, US_MIN, US_MAX)
    pwm[n].duty_ns(int(us * 1000))


def move(targets):
    """targets: {servo: servohoek t.o.v. offset}; geleidelijk."""
    targets = {n: t for n, t in targets.items() if n in pwm}
    if not targets:
        print("  (die servo('s) staan uit - eerst 'aan <naam>')")
        return
    dmax = max(abs(t - pos.get(n, 0.0)) for n, t in targets.items())
    k = max(1, int(dmax / MAX_SPEED / 0.02))
    start = {n: pos.get(n, 0.0) for n in targets}
    for i in range(1, k + 1):
        for n, t in targets.items():
            write(n, start[n] + (t - start[n]) * i / k)
        time.sleep_ms(20)
    pos.update(targets)


def on(n):
    if n in pwm:
        return
    p = PWM(Pin(SERVOS[n]))
    p.freq(50)
    pwm[n] = p
    pos[n] = 0.0
    write(n, 0.0)
    print("  aan:", n, "(gpio", SERVOS[n], ") -> HOME")


def off(n):
    if n in pwm:
        pwm[n].duty_ns(0)
        pwm[n].deinit()
        del pwm[n]
        pos.pop(n, None)


def servos_of(name):
    if name in SERVOS:
        return [name]
    if name in DIFFS:
        return list(DIFFS[name])
    j = JOINTS.get(name)
    if j is None:
        return []
    return list(DIFFS[j[1]]) if j[0] == "diff" else [j[1]]


def diff_servo_deltas(j, d):
    """gewrichtshoek d (pitch/yaw of roll) van een differentieel -> {servoA: x, servoB: y} (servo-graden)"""
    a, b = DIFFS[j[1]]
    c = cal[j[1]]
    if j[2] == "p":
        return {a: RATIO * c["sp"] * d, b: RATIO * c["sp"] * d}
    return {a: RATIO * c["sr"] * d, b: -RATIO * c["sr"] * d}


def trim(n, d):
    """offset van servo n met d graden verschuiven (de servo beweegt mee, HOME blijft HOME)."""
    o = clamp(cal[n]["offset"] + d, -MAX_OFFSET, MAX_OFFSET)
    if o != cal[n]["offset"] + d:
        print("  ! offset begrensd op", MAX_OFFSET, "gr - klopt de montage van de hoorn/het tandwiel?")
    cal[n]["offset"] = o
    if n in pwm:
        write(n, pos[n])


def show_sel():
    if sel is None:
        return
    j = JOINTS.get(sel)
    if j is not None and j[0] == "diff":
        print("  %s: richting %+d (%s)" % (sel, cal[j[1]]["sp" if j[2] == "p" else "sr"], "sp" if j[2] == "p" else "sr"))
    for n in servos_of(sel):
        print("  %-10s offset %+6.1f gr  teken %+d  %s" % (n, cal[n]["offset"], cal[n]["teken"], "AAN" if n in pwm else "uit"))


def test(deg, dur=5.0, cycles=3, name=None):
    """Langzaam en continu heen en weer rond HOME: +deg en -deg, 'cycles' keer in 'dur' s.
    Zachte start en stop (amplitude zwelt aan en af), begint en eindigt in HOME."""
    name = name or sel
    j = JOINTS.get(name)
    if j is None:
        rng = (-90, 90)
        targets = lambda x: {name: x}
    elif j[0] == "enkel":
        rng = j[2]
        targets = lambda x: {j[1]: x}
    else:
        rng = j[3]
        targets = lambda x: diff_servo_deltas(j, x)
    a = abs(deg)
    servos = list(targets(0.0))
    if any(n not in pwm for n in servos):
        print("  eerst 'aan'")
        return
    move({n: 0.0 for n in servos})                       # vanuit HOME
    print("  ", name, "+-%.1f gr, %d keer heen en weer in %.0f s (Ctrl+C = stoppen)" % (a, cycles, dur))
    n = int(dur / 0.02)
    try:
        for i in range(n + 1):
            t = i / n                                     # 0..1
            env = math.sin(math.pi * t) ** 2              # zacht aan en uit
            x = clamp(a * env * math.sin(2 * math.pi * cycles * t), rng[0], rng[1])   # bv. kaak: niet onder 0
            for sv, v in targets(x).items():
                write(sv, v)
            time.sleep_ms(20)
    except KeyboardInterrupt:
        print("  gestopt")
        move({sv: 0.0 for sv in servos})
        return False
    move({sv: 0.0 for sv in servos})
    pos.update({sv: 0.0 for sv in servos})
    return True


def joint_targets(name, x):
    """gewrichtshoek x (gr) van gewricht/servo 'name' -> {servo: servohoek}, geclipt op het bereik"""
    j = JOINTS.get(name)
    if j is None:
        return {name: x}
    rng = j[2] if j[0] == "enkel" else j[3]
    x = clamp(x, rng[0], rng[1])
    return {j[1]: x} if j[0] == "enkel" else diff_servo_deltas(j, x)


def fase(jn):
    """faseverschuiving per soort, zodat pitch en roll van een differentieel elkaar niet opheffen
    (het been draait dan een rondje i.p.v. schuin); links en rechts zelfde fase = spiegelbeeld"""
    return 0.0 if jn.endswith("pitch") else 0.25 if jn.endswith("roll") else 0.5 if jn.endswith("yaw") else 0.0


def test_samen(names, deg, dur=5.0, cycles=3, met_fase=False):
    """Alle gegeven gewrichten TEGELIJK langzaam heen en weer (zelfde sinus, zachte start/stop).
    Bij een differentieel tellen pitch en roll op."""
    servos = sorted({n for jn in names for n in servos_of(jn)})
    move({n: 0.0 for n in servos})
    print("   samen:", ", ".join(names), "+-%.1f gr, %d keer in %.0f s (Ctrl+C = stoppen)" % (abs(deg), cycles, dur))
    n = int(dur / 0.02)
    try:
        for i in range(n + 1):
            t = i / n
            env = abs(deg) * math.sin(math.pi * t) ** 2
            tot = {}
            for jn in names:
                ph = fase(jn) if met_fase else 0.0
                x = env * math.sin(2 * math.pi * (cycles * t + ph))
                for sv, v in joint_targets(jn, x).items():
                    tot[sv] = tot.get(sv, 0.0) + v
            for sv, v in tot.items():
                write(sv, v)
            time.sleep_ms(20)
    except KeyboardInterrupt:
        print("  gestopt")
    move({sv: 0.0 for sv in servos})
    pos.update({sv: 0.0 for sv in servos})


def save():
    with open(CALIB, "w") as f:
        json.dump(cal, f)
    t = time.localtime()
    with open("calib_log.txt", "a") as f:
        f.write("%04d-%02d-%02d %02d:%02d %s\n" % (t[0], t[1], t[2], t[3], t[4], json.dumps(cal)))
    print("  opgeslagen in", CALIB, "(+ calib_log.txt)")


def listing():
    for jn, j in JOINTS.items():
        ss = servos_of(jn)
        extra = ("  richting %+d" % cal[j[1]]["sp" if j[2] == "p" else "sr"]) if j[0] == "diff" else ""
        print("%-13s" % jn + "  ".join("%s(gpio%d) %+5.1f %+d %s" % (n, SERVOS[n], cal[n]["offset"], cal[n]["teken"],
                                                                    "AAN" if n in pwm else "uit") for n in ss) + extra)


print(__doc__ if False else "ijk.py - typ 'lijst' of zie de commando's bovenaan het bestand. Alles staat UIT.")
listing()
while True:
    try:
        cmd = input("ijk> ").strip().split()
    except (KeyboardInterrupt, EOFError):
        cmd = ["q"]
    if not cmd:
        continue
    c, args = cmd[0], cmd[1:]
    try:
        if c == "q":
            for n in list(pwm):
                off(n)
            print("alles slap. Niet vergeten: 'bewaar' als je wilde opslaan.")
            break
        elif c == "lijst":
            listing()
        elif c == "aan":
            tgt = args[0] if args else (sel or "")
            names = list(SERVOS) if tgt == "alles" else servos_of(tgt)
            for jn in (list(JOINTS) if tgt == "alles" else [tgt] if tgt in JOINTS else
                       [k for k, v in JOINTS.items() if v[0] == "diff" and v[1] == tgt]):
                if jn not in actief:
                    actief.append(jn)
            for n in names:
                on(n)
        elif c == "uit":
            for n in (servos_of(args[0]) if args else list(pwm)):
                off(n)
            # gewrichten waarvan een servo nu uit staat tellen niet meer mee
            actief[:] = [jn for jn in actief if all(n in pwm for n in servos_of(jn))]
            print("  slap")
        elif c == "kies":
            if args and (args[0] in JOINTS or args[0] in SERVOS):
                sel = args[0]
                show_sel()
                if any(n not in pwm for n in servos_of(sel)):
                    print("  (tip: 'aan", sel + "')")
            else:
                print("  onbekend; kies uit:", ", ".join(list(JOINTS) + list(SERVOS)))
        elif c in ("+", "-", "++", "--"):
            if sel is None:
                print("  eerst 'kies <naam>'")
                continue
            d = step * (5 if len(c) == 2 else 1) * (1 if c[0] == "+" else -1)
            j = JOINTS.get(sel)
            if j is not None and j[0] == "diff":
                for n, x in diff_servo_deltas(j, d).items():
                    trim(n, x)
            else:
                for n in servos_of(sel):
                    trim(n, d)
            show_sel()
        elif c == "stap":
            step = clamp(float(args[0]), 0.1, 5.0)
            print("  stap", step, "gr")
        elif c == "test":
            alles = bool(args) and args[0] in ("alles", "elk", "soort")
            samen = bool(args) and args[0] in ("alles", "soort")
            per_soort = bool(args) and args[0] == "soort"
            if alles:
                args = args[1:]
            amp = float(args[0]) if args else 10.0
            dur = float(args[1]) if len(args) > 1 else 5.0
            cyc = int(args[2]) if len(args) > 2 else 3
            if alles:   # de gewrichten die je met 'aan' / 'thuis' hebt aangezet
                todo = [jn for jn in JOINTS if jn in actief and all(n in pwm for n in servos_of(jn))]
                if not todo:
                    print("  er staat geen gewricht aan - eerst 'aan alles' of 'thuis alles'")
                elif samen and not per_soort:
                    test_samen(todo, amp, dur, cyc)     # ALLES tegelijk, hoeken per servo opgeteld
                elif samen:
                    # per soort tegelijk (anders lopen pitch en roll van een differentieel door elkaar):
                    # eerst alle heup-pitch, dan alle roll, dan alle yaw, dan de rest
                    groepen = [[j for j in todo if j.endswith("heup_pitch")],
                               [j for j in todo if j.endswith("_roll")],
                               [j for j in todo if j.endswith("_yaw")]]
                    rest = [j for j in todo if not any(j in g for g in groepen)]
                    for g in groepen + [rest]:
                        if g:
                            test_samen(g, amp, dur, cyc)
                            time.sleep_ms(500)
                for jn in ([] if samen else todo):
                    if test(amp, dur, cyc, jn) is False:     # Ctrl+C stopt de hele reeks
                        break
                    time.sleep_ms(500)
            elif sel is None:
                print("  eerst 'kies <naam>' (of 'test alles ...')")
            else:
                test(amp, dur, cyc)
        elif c == "draai":
            j = JOINTS.get(sel) if sel else None
            if j is not None and j[0] == "diff":
                k = "sp" if j[2] == "p" else "sr"
                cal[j[1]][k] = -cal[j[1]][k]
                print("  richting", sel, "omgedraaid (HOME blijft staan)")
            else:
                for n in servos_of(sel) if sel else []:
                    cal[n]["teken"] = -cal[n]["teken"]
                    cal[n]["offset"] = -cal[n]["offset"]      # zelfde fysieke HOME houden
                    pos[n] = -pos[n] if n in pos else 0.0
            show_sel()
        elif c == "naar":
            # gewricht naar een hoek t.o.v. HOME, ZONDER de ijking te veranderen ('thuis' = terug)
            if sel is None or not args:
                print("  gebruik: kies <naam>, dan: naar <gr>   (bv. naar 10, naar -5)")
            else:
                tg = joint_targets(sel, float(args[0]))
                if any(n not in pwm for n in tg):
                    print("  eerst 'aan'")
                else:
                    move(tg)
                    print("  ", sel, "op", args[0], "gr t.o.v. HOME (ijking ongewijzigd; 'thuis' = terug)")
        elif c == "herstel":
            # niet-bewaarde bijstellingen weggooien: terug naar calib.json
            for k, v in load().items():
                cal[k] = v
            for n in pwm:
                write(n, pos[n])
            show_sel()
        elif c == "spiegel":
            # 2026-10-31: rechts yaw en roll omdraaien (links en rechts bewogen dezelfde kant op)
            kant = args[0] if args else "R"
            pre = "R" if kant.upper().startswith("R") else "L"
            ys = pre + "_yaw"
            cal[ys]["teken"] = -cal[ys]["teken"]
            cal[ys]["offset"] = -cal[ys]["offset"]
            if ys in pos:
                pos[ys] = -pos[ys]
            dn = pre + "_heup"
            cal[dn]["sr"] = -cal[dn]["sr"]
            for n in pwm:
                write(n, pos[n])
            print("  %s_heup_yaw en %s_heup_roll omgedraaid (HOME blijft staan). Check met 'test alles 5 5 3', dan 'bewaar'." % (pre, pre))
        elif c == "thuis":
            # 'thuis' = alles wat aan staat; 'thuis <naam>' / 'thuis alles' zet die ook aan
            if args:
                for n in (list(SERVOS) if args[0] == "alles" else servos_of(args[0])):
                    on(n)
                for jn in (list(JOINTS) if args[0] == "alles" else [args[0]] if args[0] in JOINTS else []):
                    if jn not in actief:
                        actief.append(jn)
            if not pwm:
                print("  er staat geen servo aan - 'thuis alles' of 'thuis <naam>' zet ze aan")
            else:
                move({n: 0.0 for n in pwm})
                for n in pwm:
                    write(n, 0.0)                     # nog een keer expliciet de HOME-puls
                print("  naar HOME:", ", ".join(sorted(pwm)))
        elif c == "bewaar":
            save()
        elif c == "toon":
            print(json.dumps(cal))
        else:
            print("  onbekend commando")
    except Exception as e:
        print("  fout:", e)
