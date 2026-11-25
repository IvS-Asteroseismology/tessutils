#!/usr/bin/env python3
"""
Written by Claude Sonnet 5.5
Cutouts of TESS FFIs straight from the public AWS bucket (no MAST).
Output is a TESScut-like FITS file.

Pipeline
--------
1. tess-point      RA/Dec -> (sector, camera, CCD) for every sector that sees the target
2. s3fs (anon)     list the calibrated FFIs (*_ffic.fits) of that sector/camera/CCD
3. one FFI header  WCS -> exact pixel position of the target (tess-point is only
                   accurate to ~1-2 px, the SPOC WCS is much better)
4. every FFI       byte-range read of just the cutout rows
                   (HDU 1 = calibrated flux, HDU 2 = flux uncertainty)

Output: one FITS file per (sector, camera, CCD), laid out like an astrocut/TESScut file so
it loads with lightkurve.TessTargetPixelFile and works in TESScut-based pipelines:
    PRIMARY    header only (TICID, SECTOR, CAMERA, CCD, RA_OBJ, DEC_OBJ, ...)
    PIXELS     one row per FFI: TIME (mid-exposure, BTJD), TIMECORR (barycentric
               correction), CADENCENO (FFI index), RAW_CNTS (-1), FLUX, FLUX_ERR
               (SPOC uncertainty image), FLUX_BKG / FLUX_BKG_ERR (0, FFIs are not
               background-subtracted separately), QUALITY (DQUALITY), POS_CORR1/2 (0),
               FFI_FILE. The header holds the cutout WCS and 1-based FFI box corner
               (1CRV5P / 2CRV5P) exactly as TESScut does.
    APERTURE   int32 image of the cutout size carrying the cutout WCS (reference frame's
               SPOC WCS incl. SIP distortion, restricted to the box); no pipeline aperture

Local cache (default ~/.cache/tess_s3, see --cache-dir / --no-cache / --refresh-cache):
    S3 file listing and the reference-frame header per (sector, camera, CCD).
    A few kB per CCD-sector.

Examples
--------
    # What would be downloaded? (only lists S3 keys, reads nothing else)
    python tess_s3_cutout.py --tic 261136679 --ra 84.291188 --dec -80.469119 --dry-run

    # All sectors, 11x11 pixel cutouts
    python tess_s3_cutout.py --tic 261136679 --ra 84.291188 --dec -80.469119 --size 11

    # Only a few sectors
    python tess_s3_cutout.py --tic 261136679 --ra 84.291188 --dec -80.469119 --sectors 1 2 13
"""
import argparse
import json
import logging
import os
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np

log = logging.getLogger("tess_s3_cutout")

S3_PREFIX = "stpubdata/tess/public/ffi"
DEFAULT_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "tess_s3")
# Small readahead block: rows of a cutout lie ~8.5 kB apart in the file, so one
# block covers many rows while never pulling much more than the cutout itself.
FSSPEC_KWARGS = {
    "anon": True,
    "default_block_size": 2**17,
    "default_cache_type": "readahead",
}


# --------------------------------------------------------------------------
# pure geometry helpers (numpy only)
# --------------------------------------------------------------------------
def box_corner(x, y, nx, ny):
    """0-based lower-left corner (col0, row0) of an nx*ny box centred on (x, y)."""
    return int(np.floor(x + 0.5)) - nx // 2, int(np.floor(y + 0.5)) - ny // 2


def clip_box(col0, row0, nx, ny, width, height):
    """Part of the box that lies on the image, as array slices bounds."""
    return (max(col0, 0), min(col0 + nx, width),
            max(row0, 0), min(row0 + ny, height))


def place(out, sub, col0, row0, xa, ya):
    """Paste `sub` (read at image position xa, ya) into the NaN-padded `out`."""
    h, w = sub.shape
    out[ya - row0: ya - row0 + h, xa - col0: xa - col0 + w] = sub
    return out


# --------------------------------------------------------------------------
# step 1: locate
# --------------------------------------------------------------------------
def locate(tic, ra, dec, sectors=None):
    """Return a list of (sector, camera, ccd, col, row) from tess-point."""
    from tess_stars2px import tess_stars2px_function_entry

    # returns (ID, eclLong, eclLat, sector, camera, ccd, col, row, scinfo)
    res = tess_stars2px_function_entry(tic, ra, dec)
    secs, cams, ccds, cols, rows = res[3], res[4], res[5], res[6], res[7]
    hits = [(int(s), int(c), int(d), float(x), float(y))
            for s, c, d, x, y in zip(secs, cams, ccds, cols, rows)]
    if sectors:
        hits = [h for h in hits if h[0] in set(sectors)]
    return sorted(hits)


# --------------------------------------------------------------------------
# step 2: list files on S3 (cached)
# --------------------------------------------------------------------------
def get_fs():
    import s3fs
    return s3fs.S3FileSystem(anon=True)


def list_ffi(fs, sector, camera, ccd, cache_dir=None, refresh=False):
    """Time-ordered S3 keys of the calibrated FFIs of one sector/camera/CCD."""
    cache_file = None
    if cache_dir:
        cache_file = os.path.join(cache_dir, f"s{sector:04d}-{camera}-{ccd}.keys.json")
        if os.path.exists(cache_file) and not refresh:
            with open(cache_file) as f:
                return json.load(f)
    pattern = f"{S3_PREFIX}/s{sector:04d}/*/*/{camera}-{ccd}/*_ffic.fits"
    keys = sorted(fs.glob(pattern))  # names start with a timestamp -> chronological
    if cache_file and keys:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(keys, f)
    return keys


# --------------------------------------------------------------------------
# steps 3 + 4: header WCS and byte-range reads
# --------------------------------------------------------------------------
def open_fits(key):
    """Open an FFI lazily. `key` is an S3 key or a local path (handy for tests)."""
    from astropy.io import fits

    if os.path.exists(key):
        return fits.open(key, lazy_load_hdus=True, memmap=False)
    return fits.open(f"s3://{key}", use_fsspec=True, fsspec_kwargs=FSSPEC_KWARGS,
                     lazy_load_hdus=True, memmap=False)


def reference_header(key, cache_dir=None, refresh=False):
    """Image header (HDU 1) of the reference FFI, cached on disk."""
    from astropy.io import fits

    cache_file = None
    if cache_dir:
        cache_file = os.path.join(cache_dir, os.path.basename(key) + ".hdr")
        if os.path.exists(cache_file) and not refresh:
            with open(cache_file) as f:
                return fits.Header.fromstring(f.read(), sep="\n")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with open_fits(key) as hdul:
            hdr = hdul[1].header.copy()
    if cache_file:
        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_file, "w") as f:
            f.write(hdr.tostring(sep="\n"))
    return hdr


def reference_position(key, ra, dec, cache_dir=None, refresh=False):
    """Target pixel position (0-based x=col, y=row) from one FFI's header WCS."""
    from astropy.wcs import WCS, FITSFixedWarning

    hdr = reference_header(key, cache_dir, refresh)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FITSFixedWarning)
        x, y = WCS(hdr).all_world2pix(ra, dec, 0)
    return float(x), float(y), hdr


def read_frame(key, box, retries=4):
    """Read one cutout. `box` = (col0, row0, nx, ny).

    Returns dict with flux, flux_err (ny, nx float32), tstart, tstop, quality,
    barycorr, cadenceno.
    """
    col0, row0, nx, ny = box
    last = None
    for attempt in range(retries):
        try:
            with open_fits(key) as hdul:
                sci, err = hdul[1], hdul[2]
                hdr = sci.header
                width, height = hdr["NAXIS1"], hdr["NAXIS2"]
                xa, xb, ya, yb = clip_box(col0, row0, nx, ny, width, height)
                flux = np.full((ny, nx), np.nan, dtype=np.float32)
                flux_err = np.full((ny, nx), np.nan, dtype=np.float32)
                if xb > xa and yb > ya:
                    place(flux, np.asarray(sci.section[ya:yb, xa:xb], dtype=np.float32),
                          col0, row0, xa, ya)
                    place(flux_err, np.asarray(err.section[ya:yb, xa:xb], dtype=np.float32),
                          col0, row0, xa, ya)
                prim = hdul[0].header
                return {
                    "flux": flux,
                    "flux_err": flux_err,
                    "tstart": float(hdr.get("TSTART", prim.get("TSTART", np.nan))),
                    "tstop": float(hdr.get("TSTOP", prim.get("TSTOP", np.nan))),
                    "quality": int(hdr.get("DQUALITY", 0)),
                    "barycorr": float(hdr.get("BARYCORR", np.nan)),
                    "cadenceno": int(prim.get("FFIINDEX", 0)),
                }
        except Exception as exc:  # network hiccups etc.
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"giving up on {key}: {last!r}")


# --------------------------------------------------------------------------
# FITS output (same layout as astrocut/TESScut files)
# --------------------------------------------------------------------------
# header keywords copied from the reference FFI into the PIXELS extension
TIMING_KEYS = ("DEADC", "EXPOSURE", "FRAMETIM", "INT_TIME", "LIVETIME", "NUM_FRM",
               "TIERRELA", "TIMEDEL", "TIMEPIXR", "TIMEREF", "TASSIGN", "TIMESYS",
               "TIMEUNIT", "BJDREFI", "BJDREFF", "TELESCOP")


def cutout_wcs(ref_hdr, col0, row0, nx, ny):
    """Reference-frame WCS (incl. SIP) restricted to the cutout box."""
    from astropy.wcs import WCS

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return WCS(ref_hdr)[row0:row0 + ny, col0:col0 + nx]


def linear_wcs_cards(wcs, col0, row0, first_col, last_col):
    """TESScut-style table-column WCS keywords (plain TAN, no SIP) for columns
    first_col..last_col. lightkurve reads the FLUX column's (5) keywords."""
    pc = wcs.wcs.get_pc()
    cdelt = wcs.wcs.get_cdelt()
    cards = {}
    for n in range(first_col, last_col + 1):
        cards.update({
            f"1CTYP{n}": "RA---TAN", f"2CTYP{n}": "DEC--TAN",
            f"1CRPX{n}": wcs.wcs.crpix[0], f"2CRPX{n}": wcs.wcs.crpix[1],
            f"1CRVL{n}": wcs.wcs.crval[0], f"2CRVL{n}": wcs.wcs.crval[1],
            f"1CUNI{n}": "deg", f"2CUNI{n}": "deg",
            f"1CDLT{n}": float(cdelt[0]), f"2CDLT{n}": float(cdelt[1]),
            f"11PC{n}": pc[0, 0], f"12PC{n}": pc[0, 1],
            f"21PC{n}": pc[1, 0], f"22PC{n}": pc[1, 1],
            f"WCSN{n}P": "PHYSICAL",
            f"1CTY{n}P": "RAWX", f"2CTY{n}P": "RAWY",
            f"1CUN{n}P": "PIXEL", f"2CUN{n}P": "PIXEL",
            f"1CRV{n}P": col0 + 1, f"2CRV{n}P": row0 + 1,  # 1-based FFI position
            f"1CDL{n}P": 1.0, f"2CDL{n}P": 1.0,
            f"1CRP{n}P": 1, f"2CRP{n}P": 1,
        })
    return cards


def write_fits(path, results, keys, ref_hdr, meta, col0, row0, nx, ny):
    """Write a TESScut-compatible FITS file with flux and flux errors.

    Extensions: PRIMARY, PIXELS (TIME, TIMECORR, CADENCENO, RAW_CNTS, FLUX, FLUX_ERR,
    FLUX_BKG, FLUX_BKG_ERR, QUALITY, POS_CORR1, POS_CORR2, FFI_FILE), APERTURE.
    Columns the FFIs do not provide are filled like TESScut does (RAW_CNTS=-1,
    background and POS_CORR = 0).
    """
    from astropy.io import fits

    n = len(results)
    stack = lambda name: np.stack([r[name] for r in results])  # noqa: E731
    flux, flux_err = stack("flux"), stack("flux_err")
    tstart = np.array([r["tstart"] for r in results])
    tstop = np.array([r["tstop"] for r in results])
    names = [os.path.basename(k).replace("_ffic.fits", "_ffi") for k in keys]
    wcs = cutout_wcs(ref_hdr, col0, row0, nx, ny)

    prim = fits.PrimaryHDU()
    h = prim.header
    # lightkurve recognises TESScut-style files by "STScI" / "MAST" in ORIGIN
    h["ORIGIN"] = ("STScI/MAST (AWS cutout)", "TESScut layout, data from public AWS bucket")
    h["CREATOR"] = ("tess_s3_cutout.py", "software used to produce this file")
    h["TSTART"] = (float(np.nanmin(tstart)), "observation start time in TJD of first FFI")
    h["TSTOP"] = (float(np.nanmax(tstop)), "observation stop time in TJD of last FFI")
    h["TELESCOP"] = "TESS"
    h["INSTRUME"] = "TESS Photometer"
    h["CAMERA"] = (meta["camera"], "Camera number")
    h["CCD"] = (meta["ccd"], "CCD chip number")
    h["SECTOR"] = (meta["sector"], "Observing sector")
    h["FFI_TYPE"] = ("SPOC", "the FFI type used to make the cutouts")
    h["RA_OBJ"] = (meta["ra"], "[deg] right ascension")
    h["DEC_OBJ"] = (meta["dec"], "[deg] declination")
    h["TICID"] = (meta["tic"], "unique TESS target identifier")
    h["TIMESYS"] = ("TDB", "time system is Barycentric Dynamical Time (TDB)")
    h["BJDREFI"] = (2457000, "integer part of BTJD reference date")
    h["BJDREFF"] = (0.0, "fraction of the day in BTJD reference date")
    h["TIMEUNIT"] = ("d", "time unit for TIME, TSTART, and TSTOP")
    h["TARGX"] = (meta["x"], "0-based FFI array column of target (reference WCS)")
    h["TARGY"] = (meta["y"], "0-based FFI array row of target (reference WCS)")

    cell = f"({nx},{ny})"
    npix = nx * ny
    zeros = np.zeros((n, npix), dtype=np.float32)
    cols = [
        fits.Column("TIME", "D", unit="BJD - 2457000, days", disp="D14.7",
                    array=(tstart + tstop) / 2),
        fits.Column("TIMECORR", "E", unit="d", disp="E14.7", array=stack("barycorr")),
        fits.Column("CADENCENO", "J", disp="I10", array=stack("cadenceno")),
        fits.Column("RAW_CNTS", f"{npix}J", unit="count", null=-1, dim=cell,
                    array=np.full((n, npix), -1, dtype=np.int32)),
        fits.Column("FLUX", f"{npix}E", unit="e-/s", dim=cell, array=flux.reshape(n, -1)),
        fits.Column("FLUX_ERR", f"{npix}E", unit="e-/s", dim=cell,
                    array=flux_err.reshape(n, -1)),
        fits.Column("FLUX_BKG", f"{npix}E", unit="e-/s", dim=cell, array=zeros),
        fits.Column("FLUX_BKG_ERR", f"{npix}E", unit="e-/s", dim=cell, array=zeros),
        fits.Column("QUALITY", "J", disp="B16.16", array=stack("quality")),
        fits.Column("POS_CORR1", "E", unit="pixel", array=np.zeros(n, dtype=np.float32)),
        fits.Column("POS_CORR2", "E", unit="pixel", array=np.zeros(n, dtype=np.float32)),
        fits.Column("FFI_FILE", f"{max(len(s) for s in names)}A", unit="pixel",
                    array=np.array(names)),
    ]
    pix = fits.BinTableHDU.from_columns(cols, name="PIXELS")
    for k in TIMING_KEYS:
        if k in ref_hdr:
            pix.header[k] = ref_hdr[k]
    # flux-like columns 4-8 carry the WCS and the 1-based FFI position of the box
    for k, v in linear_wcs_cards(wcs, col0, row0, 4, 8).items():
        pix.header[k] = v
    pix.header["WCS_FFI"] = os.path.basename(keys[len(keys) // 2])
    pix.header["TSTART"] = h["TSTART"]
    pix.header["TSTOP"] = h["TSTOP"]
    pix.header["CAMERA"], pix.header["CCD"] = meta["camera"], meta["ccd"]
    pix.header["RA_OBJ"], pix.header["DEC_OBJ"] = meta["ra"], meta["dec"]
    pix.header["INHERIT"] = True

    aper = fits.ImageHDU(np.zeros((ny, nx), dtype=np.int32), name="APERTURE")
    aper.header.update(wcs.to_header(relax=True))
    aper.header["INHERIT"] = True

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fits.HDUList([prim, pix, aper]).writeto(path, overwrite=True)


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def cutout_one(fs, tic, ra, dec, sector, camera, ccd, size, outdir, workers,
               overwrite=False, dry_run=False, tp_col=None, tp_row=None,
               cache_dir=None, refresh=False, max_frames=None, out_path=None):
    tag = f"s{sector:04d}-{camera}-{ccd}"
    if out_path is None:
        out_path = os.path.join(outdir, f"tic{tic}_{tag}_{size}x{size}.fits")
    if os.path.exists(out_path) and not overwrite and not dry_run:
        log.info("%s: already done (%s), skipping", tag, out_path)
        return out_path

    keys = list_ffi(fs, sector, camera, ccd, cache_dir, refresh)
    if not keys:
        log.warning("%s: no FFIs on S3 (not released yet, or different layout)", tag)
        return None
    if max_frames:
        keys = keys[:max_frames]
    if dry_run:
        log.info("%s: %d FFIs  (tess-point predicts col=%.1f row=%.1f)",
                 tag, len(keys), tp_col, tp_row)
        return None

    # WCS from the middle frame halves the effect of velocity-aberration drift
    ref_key = keys[len(keys) // 2]
    x, y, hdr = reference_position(ref_key, ra, dec, cache_dir, refresh)
    if tp_col is not None:
        log.info("%s: WCS pixel (x=%.1f, y=%.1f); tess-point said (col=%.1f, row=%.1f)",
                 tag, x + 1, y + 1, tp_col, tp_row)  # +1 -> FITS 1-based, like tess-point
    col0, row0 = box_corner(x, y, size, size)
    box = (col0, row0, size, size)
    log.info("%s: reading %d frames, box corner col0=%d row0=%d", tag, len(keys), col0, row0)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda k: read_frame(k, box), keys))

    meta = dict(tic=tic, sector=sector, camera=camera, ccd=ccd, ra=ra, dec=dec, x=x, y=y)
    write_fits(out_path, results, keys, hdr, meta, col0, row0, size, size)
    log.info("%s: wrote %s  (%.1f s)", tag, out_path, time.time() - t0)
    return out_path


def tic_coordinates(tic, retries=3):
    """ICRS (ra, dec) in degrees of a TIC star from the MAST TIC catalog."""
    from astroquery.mast import Catalogs

    last = None
    for attempt in range(retries):
        try:
            tab = Catalogs.query_criteria(catalog="TIC", ID=int(tic))
            if len(tab) == 0:
                raise LookupError(f"TIC {tic} not found in the TIC catalog")
            return float(tab["ra"][0]), float(tab["dec"][0])
        except LookupError:
            raise
        except Exception as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"TIC catalog query failed for TIC {tic}: {last!r}")


def _edge_distance(col, row):
    """Distance (pix) of a tess-point position to the nearest edge of the 2048x2048
    science area (columns 45-2092, rows 1-2048)."""
    return min(col - 45, 2092 - col, row - 1, 2048 - row)


def cutout_tic(tic, size=20, outdir="./tpfs", pattern="tic{TIC}_sec{SECTOR}.fits",
               sectors=None, overwrite=False, workers=16, ra=None, dec=None,
               cache_dir=DEFAULT_CACHE, max_frames=None):
    """TESScut replacement: write one FITS file per sector for a TIC star.

    If the star falls on two CCDs in a sector, the one where it is further from the
    edge is used (one file per sector, like the `pattern` implies). `ra`/`dec` default
    to the TIC catalog position. Returns the list of files written or already present.
    """
    if ra is None or dec is None:
        ra, dec = tic_coordinates(tic)
    if isinstance(sectors, int):
        sectors = [sectors]
    best = {}
    for hit in locate(tic, ra, dec, sectors):
        if hit[0] not in best or _edge_distance(*hit[3:]) > _edge_distance(*best[hit[0]][3:]):
            best[hit[0]] = hit
    fs = get_fs()
    paths = []
    for sector, (_, camera, ccd, col, row) in sorted(best.items()):
        out_path = os.path.join(outdir, pattern.format(TIC=tic, SECTOR=sector))
        path = cutout_one(fs, tic, ra, dec, sector, camera, ccd, size, outdir, workers,
                          overwrite=overwrite, tp_col=col, tp_row=row,
                          cache_dir=cache_dir, max_frames=max_frames, out_path=out_path)
        if path:
            paths.append(path)
    return paths


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tic", type=int, required=True, help="TIC ID (used for file names)")
    p.add_argument("--ra", type=float, required=True, help="RA in degrees (ICRS)")
    p.add_argument("--dec", type=float, required=True, help="Dec in degrees (ICRS)")
    p.add_argument("--size", type=int, default=11, help="cutout is size x size pixels")
    p.add_argument("--sectors", type=int, nargs="*", help="restrict to these sectors")
    p.add_argument("--outdir", default="tess_cutouts")
    p.add_argument("--workers", type=int, default=16, help="parallel S3 reads")
    p.add_argument("--max-frames", type=int, help="only the first N frames (testing)")
    p.add_argument("--cache-dir", default=DEFAULT_CACHE, help="listing/header cache")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--refresh-cache", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="only locate the target and count FFIs, read no pixels")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")

    hits = locate(args.tic, args.ra, args.dec, args.sectors)
    if not hits:
        log.error("tess-point finds no sector covering this position")
        return 1
    log.info("tess-point: target falls in %d sector/camera/CCD combinations", len(hits))

    cache_dir = None if args.no_cache else args.cache_dir
    fs = get_fs()
    for sector, camera, ccd, col, row in hits:
        try:
            cutout_one(fs, args.tic, args.ra, args.dec, sector, camera, ccd,
                       args.size, args.outdir, args.workers, args.overwrite,
                       args.dry_run, col, row, cache_dir, args.refresh_cache,
                       args.max_frames)
        except Exception:
            log.exception("sector %d failed, continuing", sector)
    return 0


if __name__ == "__main__":
    sys.exit(main())
