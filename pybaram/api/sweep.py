# -*- coding: utf-8 -*-
import csv
import glob
import os
import shutil

from pybaram.inifile import INIFile


def run_aoa_sweep(meshf, inif, aoas, outdir='sweep-aoa', ui='tui',
                  comm='none', overwrite=False):
    from pybaram.api.simulation import run
    from pybaram.api.sweep_progress import make_sweep_progress
    from pybaram.readers.native import NativeReader
    from pybaram.utils.mpi import mpi_init

    if comm == 'none':
        comm = mpi_init()

    meshf = os.path.abspath(meshf)
    inif = os.path.abspath(inif)
    outdir = os.path.abspath(outdir)
    root = os.getcwd()

    if comm.rank == 0:
        os.makedirs(outdir, exist_ok=True)
    comm.Barrier()

    sweep_progress = make_sweep_progress(aoas, comm, ui)
    summary_rows = []

    try:
        sweep_progress.start()

        for i, aoa in enumerate(aoas):
            sweep_progress.start_case(aoa, i)
            _run_aoa_case(
                meshf, inif, outdir, root, aoa, comm, overwrite,
                summary_rows, run, NativeReader
            )
            sweep_progress.complete_case(aoa, i)
    finally:
        sweep_progress.stop()

    if comm.rank == 0:
        write_sweep_summary(os.path.join(outdir, 'sweep.csv'), summary_rows)


def _run_aoa_case(meshf, inif, outdir, root, aoa, comm, overwrite,
                  summary_rows, run, NativeReader):
    case_name = aoa_case_name(aoa)
    case_dir = os.path.join(outdir, case_name)

    error = None
    if comm.rank == 0:
        try:
            prepare_case_dir(case_dir, overwrite)
        except RuntimeError as exc:
            error = str(exc)

    error = comm.bcast(error, root=0)
    if error:
        raise RuntimeError(error)

    comm.Barrier()

    cfg = INIFile(inif)
    cfg.set('constants', 'aoa', format_sweep_value(aoa))

    if comm.rank == 0:
        with open(os.path.join(case_dir, 'config.ini'), 'w') as outf:
            outf.write(cfg.tostr())

    os.chdir(case_dir)
    mesh = NativeReader(meshf)
    try:
        run(mesh, cfg, comm=comm, ui='none')
    finally:
        mesh.close()
        os.chdir(root)

    if comm.rank == 0:
        rows = collect_force_summary(case_dir, aoa)
        if rows:
            summary_rows.extend(rows)
        else:
            summary_rows.append({
                'aoa': format_sweep_value(aoa),
                'case': case_name,
                'force_file': ''
            })


def parse_sweep_values(values):
    aoas = []
    for value in values.split(','):
        value = value.strip()
        if value:
            aoas.append(float(value))

    if not aoas:
        raise ValueError('No AOA values were provided')

    return aoas


def parse_sweep_range(start, stop, step):
    start = float(start)
    stop = float(stop)
    step = float(step)

    if step == 0:
        raise ValueError('AOA range step must be non-zero')
    if stop > start and step < 0:
        raise ValueError('AOA range step must be positive')
    if stop < start and step > 0:
        raise ValueError('AOA range step must be negative')

    values = []
    current = start
    eps = abs(step)*1e-12

    if step > 0:
        while current <= stop + eps:
            values.append(current)
            current += step
    else:
        while current >= stop - eps:
            values.append(current)
            current += step

    return values


def aoa_case_name(aoa):
    value = format_sweep_value(aoa)
    value = value.replace('-', 'n').replace('+', '')
    value = value.replace('.', 'p')
    return 'aoa{}'.format(value)


def prepare_case_dir(case_dir, overwrite=False):
    if os.path.isdir(case_dir) and os.listdir(case_dir):
        if not overwrite:
            raise RuntimeError(
                "Sweep case directory '{}' already exists and is not empty; "
                "use --overwrite to replace it".format(case_dir)
            )

        shutil.rmtree(case_dir)

    os.makedirs(case_dir, exist_ok=True)


def format_sweep_value(value):
    return '{:.12g}'.format(float(value))


def collect_force_summary(case_dir, aoa):
    rows = []
    case_name = os.path.basename(case_dir)

    for fname in sorted(glob.glob(os.path.join(case_dir, 'force_*.csv'))):
        with open(fname, newline='') as inf:
            reader = csv.DictReader(inf)
            last = None
            for row in reader:
                last = row

        if last is None:
            continue

        out = {
            'aoa': format_sweep_value(aoa),
            'case': case_name,
            'force_file': os.path.basename(fname)
        }
        out.update(last)
        rows.append(out)

    return rows


def write_sweep_summary(fname, rows):
    fields = ['aoa', 'case', 'force_file']
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)

    with open(fname, 'w', newline='') as outf:
        writer = csv.DictWriter(outf, fields)
        writer.writeheader()
        writer.writerows(rows)
