"""DucklingSTLExport - export the SELECTED bodies (e.g. a browser group) as separate STLs into one folder.

Fusion's API cannot read browser Groups, but clicking a group in the browser selects everything in it.
So: click the group "been_links_export" -> run this script -> it exports the selection into a folder
with the name you give (default: the last one used). Then the same for "been_rechts_export".

- One binary STL per body, named after the body (spaces -> _). Selected components/occurrences
  are exported body by body (incl. sub-components).
- Tessellation HIGH (for printing).
- Units are checked and fixed by the script (Fusion's STL unit is not trusted):
    Printen -> millimeters (slicer)      MuJoCo -> meters (robot_walk.xml, scale="1 1 1")
- Assembly coordinates are kept (same frame as the existing MuJoCo meshes).
- Remembers the export base folder and mode between runs (settings.json next to this script).

Install: Fusion > Utilities > Add-Ins > Scripts and Add-Ins > "+" > Create script (Python),
name it DucklingSTLExport, replace the .py content with this file, and Run.
"""
import adsk.core, adsk.fusion, traceback, os, re, struct, json

REFINE = 0.5        # x Fusion's "High" deviations: 0.5 = twice as fine (smaller = finer, bigger files)
PART_MAX_M = 0.5    # a leg is ~0.1 m: a batch larger than this was not written in meters
SETTINGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'settings.json')


def _safe(name):
    return re.sub(r'[^A-Za-z0-9_\-]+', '_', name).strip('_')


def _read_stl(path):
    with open(path, 'rb') as f:
        data = f.read()
    n = struct.unpack_from('<I', data, 80)[0]
    return data[:80], [struct.unpack_from('<12fH', data, 84 + i * 50) for i in range(n)]


def _write_stl(path, header, tris, s):
    with open(path, 'wb') as f:
        f.write(header)
        f.write(struct.pack('<I', len(tris)))
        for t in tris:
            f.write(struct.pack('<12fH', *t[0:3], *[x * s for x in t[3:12]], t[12]))


def _occ_bodies(occ):
    out = [b for b in occ.bRepBodies if b.isVisible]
    for child in occ.childOccurrences:
        if child.isVisible:
            out += _occ_bodies(child)
    return out


def _selected_bodies(ui):
    bodies = []
    for i in range(ui.activeSelections.count):
        e = ui.activeSelections.item(i).entity
        b = adsk.fusion.BRepBody.cast(e)
        o = adsk.fusion.Occurrence.cast(e)
        if b:
            bodies.append(b)
        elif o:
            bodies += _occ_bodies(o)
    uniq, keys = [], set()                 # a body selected twice (body + its component) only once
    for b in bodies:
        k = b.entityToken
        if k not in keys:
            keys.add(k)
            uniq.append(b)
    return uniq


def _load():
    try:
        with open(SETTINGS) as f:
            return json.load(f)
    except Exception:
        return {}


def _save(s):
    try:
        with open(SETTINGS, 'w') as f:
            json.dump(s, f)
    except Exception:
        pass


def run(context):
    app = adsk.core.Application.get()
    ui = app.userInterface
    try:
        design = adsk.fusion.Design.cast(app.activeProduct)
        if not design:
            ui.messageBox('Open eerst het Duckling-ontwerp (Design workspace).')
            return
        bodies = _selected_bodies(ui)
        if not bodies:
            ui.messageBox('Niets geselecteerd.\n\nKlik eerst in de browser op de group '
                          '(bijv. been_links_export), dan dit script starten.')
            return

        st = _load()
        name, cancelled = ui.inputBox(f'{len(bodies)} bodies geselecteerd.\nNaam van het export-mapje:',
                                      'Duckling STL export', st.get('last_name', 'been_links_export'))
        if cancelled or not name.strip():
            return
        name = _safe(name)

        r = ui.messageBox('Waarvoor exporteer je?\n\nJa = PRINTEN (millimeters)\nNee = MUJOCO (meters)',
                          'Duckling STL export', adsk.core.MessageBoxButtonTypes.YesNoCancelButtonType)
        if r == adsk.core.DialogResults.DialogCancel:
            return
        target_unit, target_m = ('mm', 0.001) if r == adsk.core.DialogResults.DialogYes else ('m', 1.0)

        base = st.get('base')
        if not base or not os.path.isdir(base):
            dlg = ui.createFolderDialog()
            dlg.title = 'Kies de map waarin de export-mapjes komen (wordt onthouden)'
            if dlg.showDialog() != adsk.core.DialogResults.DialogOK:
                return
            base = dlg.folder
        folder = os.path.join(base, name)
        os.makedirs(folder, exist_ok=True)

        em = design.exportManager
        done, seen = [], set()
        for b in bodies:
            fname = _safe(b.name)
            if fname in seen:
                fname = _safe(f'{b.parentComponent.name}_{b.name}')
            seen.add(fname)
            path = os.path.join(folder, fname + '.stl')
            opts = em.createSTLExportOptions(b, path)
            opts.isBinaryFormat = True
            opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
            try:   # twice as fine as Fusion's "High": half the surface and normal deviation
                sd, nd = opts.surfaceDeviation, opts.normalDeviation
                opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementCustom
                opts.surfaceDeviation = sd * REFINE
                opts.normalDeviation = nd * REFINE
            except Exception:
                opts.meshRefinement = adsk.fusion.MeshRefinementSettings.MeshRefinementHigh
            opts.sendToPrintUtility = False
            em.execute(opts)
            done.append(path)

        # which unit did Fusion write? decide on the whole batch (one leg is ~0.1 m)
        stl = {p: _read_stl(p) for p in done}
        mx = max(max(abs(x) for x in t[3:12]) for _, tris in stl.values() for t in tris)
        if mx > PART_MAX_M * 100:
            src_unit, src_m = 'mm', 0.001
        elif mx > PART_MAX_M:
            src_unit, src_m = 'cm', 0.01
        else:
            src_unit, src_m = 'm', 1.0
        s = src_m / target_m
        if abs(s - 1.0) > 1e-9:
            for p, (hdr, tris) in stl.items():
                _write_stl(p, hdr, tris, s)

        st.update(base=base, last_name=('been_rechts_export' if name == 'been_links_export' else name))
        _save(st)
        ui.messageBox(f'{len(done)} STLs in\n{folder}\n\nFusion schreef in {src_unit}; nu in {target_unit} '
                      f'(x{s:g}). Tessellation: {1 / REFINE:g}x zo fijn als Hoog.\n\nAndere basismap kiezen: verwijder settings.json '
                      f'naast dit script.')
    except Exception:
        ui.messageBox('Fout:\n' + traceback.format_exc())
