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
#  lijst                 alle servo's en gewrichten met hun ijking
#  aan <naam|alles>      servo('s) aanzetten (gaan naar HOME = hun offset)
#  uit [naam]            servo('s) slap (zonder naam: alles)
#  kies <naam>           gewricht of servo kiezen om te ijken (bv. kies L_heup)
#  + / -                 kozen servo/gewricht een stap verder (offset aanpassen = ijken)
#  ++ / --               5 stappen
#  stap <gr>             stapgrootte (standaard 1, bv. stap 0.5)
#  p+ p- r+ r-           (differentieel) trim in pitch/yaw (p) of roll (r): verschuift A en B samen
#  test <gr>             kozen gewricht naar <gr> (t.o.v. HOME) en terug: kijk of de kant klopt
#  draai                 teken omdraaien als 'test' de verkeerde kant op ging
#  thuis                 alle aan-staande servo's naar HOME
#  bewaar                calib.json opslaan (+ regel in calib_log.txt)
#  toon                  de ijking als JSON printen (plakken in de repo: firmware/calib.json)
#  q                     alles slap en stoppen
import json, time
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
# gewrichten: naam -> ("enkel", servo, bereik) of ("diff", servoA, servoB, bereik_uit, bereik_drager)
JOINTS = {
    "L_heup": ("diff", "L_diffA", "L_diffB", (-60, 60), (-20, 20)),
    "R_heup": ("diff", "R_diffA", "R_diffB", (-60, 60), (-20, 20)),
    "kop": ("diff", "kop_diffA", "kop_diffB", (-90, 90), (-20, 20)),
    "L_heup_yaw": ("enkel", "L_yaw", (-20, 20)), "R_heup_yaw": ("enkel", "R_yaw", (-20, 20)),
    "L_knie": ("enkel", "L_knie", (-50, 90)), "R_knie": ("enkel", "R_knie", (-50, 90)),
    "L_enkel": ("enkel", "L_enkel", (-60, 90)), "R_enkel": ("enkel", "R_enkel", (-60, 90)),
    "nek": ("enkel", "nek", (-30, 30)), "kop_pitch": ("enkel", "kop_pitch", (-30, 30)),
    "kaak": ("enkel", "kaak", (0, 30)),
}
CALIB = "calib.json"


def load():
    c = {n: {"offset": 0.0, "teken": 1} for n in SERVOS}
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
    j = JOINTS.get(name)
    if j is None:
        return []
    return [j[1], j[2]] if j[0] == "diff" else [j[1]]


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
    for n in servos_of(sel):
        print("  %-10s offset %+6.1f gr  teken %+d  %s" % (n, cal[n]["offset"], cal[n]["teken"], "AAN" if n in pwm else "uit"))


def test(deg):
    j = JOINTS.get(sel)
    if j is None:
        servo = sel
        print("  servo", servo, "naar", deg, "gr en terug")
        move({servo: deg}); time.sleep_ms(800); move({servo: 0.0})
        return
    if j[0] == "enkel":
        lo, hi = j[2]
        d = clamp(deg, lo, hi)
        print("  ", sel, "naar", d, "gr (gewricht) en terug")
        move({j[1]: d}); time.sleep_ms(800); move({j[1]: 0.0})
    else:
        lo, hi = j[3]
        d = clamp(deg, lo, hi)
        a, b = j[1], j[2]
        print("  ", sel, "pitch/yaw naar", d, "gr en terug; daarna roll", clamp(deg, *j[4]), "gr en terug")
        move({a: RATIO * d, b: RATIO * d}); time.sleep_ms(800); move({a: 0.0, b: 0.0}); time.sleep_ms(400)
        r = clamp(deg, *j[4])
        move({a: RATIO * r, b: -RATIO * r}); time.sleep_ms(800); move({a: 0.0, b: 0.0})


def save():
    with open(CALIB, "w") as f:
        json.dump(cal, f)
    t = time.localtime()
    with open("calib_log.txt", "a") as f:
        f.write("%04d-%02d-%02d %02d:%02d %s\n" % (t[0], t[1], t[2], t[3], t[4], json.dumps(cal)))
    print("  opgeslagen in", CALIB, "(+ calib_log.txt)")


def listing():
    for jn, j in JOINTS.items():
        ss = [j[1], j[2]] if j[0] == "diff" else [j[1]]
        print("%-11s " % jn + "  ".join("%s(gpio%d) %+5.1f %+d %s" % (s, SERVOS[s], cal[s]["offset"], cal[s]["teken"],
                                                                       "AAN" if s in pwm else "uit") for s in ss))


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
            names = list(SERVOS) if args and args[0] == "alles" else servos_of(args[0]) if args else []
            for n in names:
                on(n)
        elif c == "uit":
            for n in (servos_of(args[0]) if args else list(pwm)):
                off(n)
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
                print("  differentieel: gebruik p+/p- (pitch/yaw) en r+/r- (roll), of kies een losse servo (bv.", j[1] + ")")
                continue
            for n in servos_of(sel):
                trim(n, d)
            show_sel()
        elif c in ("p+", "p-", "r+", "r-"):
            j = JOINTS.get(sel)
            if j is None or j[0] != "diff":
                print("  alleen voor L_heup, R_heup en kop")
                continue
            d = step * (1 if c[1] == "+" else -1)
            a, b = j[1], j[2]
            # gewrichtstrim d -> servo's: A +1.5d, B +/-1.5d (teken van de servo zit in write)
            trim(a, RATIO * d)
            trim(b, RATIO * d if c[0] == "p" else -RATIO * d)
            show_sel()
        elif c == "stap":
            step = clamp(float(args[0]), 0.1, 5.0)
            print("  stap", step, "gr")
        elif c == "test":
            if sel is None:
                print("  eerst 'kies <naam>'")
            else:
                test(float(args[0]) if args else 10.0)
        elif c == "draai":
            for n in servos_of(sel) if sel else []:
                cal[n]["teken"] = -cal[n]["teken"]
                cal[n]["offset"] = -cal[n]["offset"]      # zelfde fysieke HOME houden
                pos[n] = -pos[n] if n in pos else 0.0
                if n in pwm:
                    write(n, pos[n])
            show_sel()
            print("  let op: bij een differentieel kan het ook zijn dat A en B verwisseld zijn")
        elif c == "thuis":
            move({n: 0.0 for n in pwm})
        elif c == "bewaar":
            save()
        elif c == "toon":
            print(json.dumps(cal))
        else:
            print("  onbekend commando")
    except Exception as e:
        print("  fout:", e)
