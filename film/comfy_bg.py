"""Generate the looping hero background on the local 4090, then build the loop.

Different job from comfy_hero.py. That made a scroll-scrubbed film where the frame
WAS the content. This is a background: it sits under a headline, so it has to be dark,
slow, and empty through the middle where the type goes. Bright centred subjects fight
the copy, which is exactly what the first set did.

Four shots are generated, crossfaded into each other, and the last is crossfaded back
into the first so the loop has no visible seam.

Usage:
    python comfy_bg.py            # generate + assemble
    python comfy_bg.py --assemble # re-assemble from frames already on disk
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:8188"
CKPT = "ltx-video-2b-v0.9.5.safetensors"
T5 = "t5xxl_fp8_e4m3fn.safetensors"
OUT_ROOT = r"C:\Users\arjun\ComfyUI_windows_portable\ComfyUI\output"

WIDTH, HEIGHT = 960, 544
LENGTH = 121                # 8n+1; 121 frames at 24fps = ~5.0s per shot
FPS = 24
STEPS = 30
CFG = 3.0
XFADE = 1.2                 # seconds of crossfade between shots and around the loop

NEG = ("text, letters, words, numbers, watermark, logo, caption, "
       "people, person, face, hands, fingers, limbs, animals, "
       "centred subject, hard edges, solid object, "
       "blurry, jpeg artifacts, oversaturated, neon, rainbow, "
       "fast motion, camera shake, jitter, warping, ground, floor, horizon, fire, flame")

# Edge-weighted and slow, with the middle of frame kept clear for the headline.
# Exposure is deliberately NORMAL here: the first pass baked darkness into the prompt
# and produced frames with a mean brightness of 0.7/255. Legibility is a CSS scrim's
# job, where it can be tuned without destroying detail that cannot be recovered.
SHOTS = [
    {"name": "bg1", "seed": 41010, "prompt": (
        "Extreme macro of luminous amber ink drifting slowly along the left edge of a deep "
        "black frame, the centre of frame kept clear, soft diffuse tendrils catching light, "
        "shallow depth of field, volumetric haze, extremely slow drift, cinematic, "
        "film grain, anamorphic lens, 35mm")},
    {"name": "bg2", "seed": 41020, "prompt": (
        "Extreme macro of glowing teal ink diffusing slowly across the lower third of a black "
        "frame, upper two thirds clear, soft billowing illuminated edges, shallow depth of "
        "field, volumetric haze, extremely slow motion, cinematic, film grain, "
        "anamorphic lens, 35mm")},
    {"name": "bg3", "seed": 41030, "prompt": (
        "Extreme macro of glowing embers of amber light rising slowly through deep black "
        "water, gathered toward the right edge, centre of frame clear, bright soft points of "
        "light, heavy bokeh, volumetric haze, cinematic, film grain, anamorphic lens, 35mm")},
    {"name": "bg4", "seed": 41040, "prompt": (
        "Extreme macro of a luminous teal cloud of ink turning over on itself in black water "
        "at the top edge of frame, lower half clear, soft illuminated folds, shallow depth of "
        "field, volumetric haze, extremely slow rotation, cinematic, film grain, "
        "anamorphic lens, 35mm")},
]


def build(shot, run_id):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": T5, "type": "ltxv"}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": shot["prompt"], "clip": ["9", 0]}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": NEG, "clip": ["9", 0]}},
        "4": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["2", 0], "negative": ["3", 0], "frame_rate": FPS}},
        "5": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": WIDTH, "height": HEIGHT, "length": LENGTH, "batch_size": 1}},
        "10": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "11": {"class_type": "LTXVScheduler",
               "inputs": {"steps": STEPS, "max_shift": 2.05, "base_shift": 0.95,
                          "stretch": True, "terminal": 0.1, "latent": ["5", 0]}},
        "6": {"class_type": "SamplerCustom",
              "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": shot["seed"],
                         "cfg": CFG, "positive": ["4", 0], "negative": ["4", 1],
                         "sampler": ["10", 0], "sigmas": ["11", 0], "latent_image": ["5", 0]}},
        # output 1 is the denoised estimate; the schedule stops at terminal 0.1 and never
        # reaches sigma 0, so output 0 still carries noise.
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 1], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": f"etaxbg/{run_id}/{shot['name']}"}},
    }


def post(path, payload):
    req = urllib.request.Request(f"{HOST}{path}", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"    HTTP {e.code}: {body[:1500]}")
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    cannot reach ComfyUI: {e}")
        return None


def get(path):
    try:
        with urllib.request.urlopen(f"{HOST}{path}", timeout=60) as r:
            return json.loads(r.read())
    except Exception:
        return None


def run(shot, run_id):
    print(f"\n=== {shot['name']} seed {shot['seed']} ===")
    res = post("/prompt", {"prompt": build(shot, run_id)})
    if not res or "prompt_id" not in res:
        return None
    pid = res["prompt_id"]
    t0 = time.time()
    while time.time() - t0 < 1800:
        time.sleep(5)
        hist = get(f"/history/{pid}")
        if not hist or pid not in hist:
            continue
        st = hist[pid].get("status", {})
        if st.get("status_str") == "error":
            for m in st.get("messages", []):
                if m and m[0] == "execution_error":
                    print(f"    FAILED: {m[1].get('exception_message','').strip()[:200]}")
            return None
        if st.get("status_str") == "success" and st.get("completed") is True:
            imgs = []
            for o in hist[pid].get("outputs", {}).values():
                imgs += [i["filename"] for i in o.get("images", [])]
            if len(imgs) != LENGTH:
                print(f"    WRONG COUNT: {len(imgs)} != {LENGTH}")
                return None
            print(f"    done in {time.time()-t0:.0f}s -> {len(imgs)} frames")
            return imgs
    print("    timed out")
    return None


def ff(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-1200:])
        raise SystemExit(f"ffmpeg failed: {' '.join(args[:6])}")


def assemble(run_id, out_dir):
    """Crossfade the shots into each other, then wrap the loop back to the start."""
    src = os.path.join(OUT_ROOT, "etaxbg", run_id)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(out_dir, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    # Each shot -> its own clip.
    clips = []
    for s in SHOTS:
        clip = os.path.join(tmp, f"{s['name']}.mp4")
        ff(["-framerate", str(FPS), "-i", os.path.join(src, f"{s['name']}_%05d_.png"),
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", clip])
        clips.append(clip)
    print(f"  {len(clips)} clips")

    # Chain crossfades: 1x2, then that x3, then that x4.
    dur = LENGTH / FPS
    cur = clips[0]
    acc = dur
    for i in range(1, len(clips)):
        nxt = os.path.join(tmp, f"chain{i}.mp4")
        off = acc - XFADE
        ff(["-i", cur, "-i", clips[i],
            "-filter_complex", f"[0][1]xfade=transition=fade:duration={XFADE}:offset={off:.3f}",
            "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", nxt])
        acc = acc + dur - XFADE
        cur = nxt
    print(f"  chained: {acc:.1f}s")

    # Close the loop: crossfade the tail back into the head so playback has no seam.
    head = os.path.join(tmp, "head.mp4")
    ff(["-i", cur, "-t", f"{XFADE}", "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", head])
    looped = os.path.join(tmp, "looped.mp4")
    ff(["-i", cur, "-i", head,
        "-filter_complex", f"[0][1]xfade=transition=fade:duration={XFADE}:offset={acc - XFADE:.3f}",
        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", looped])

    mp4 = os.path.join(out_dir, "hero.mp4")
    webm = os.path.join(out_dir, "hero.webm")
    poster = os.path.join(out_dir, "hero-poster.jpg")
    # NVENC on the 4090 for the deliverables.
    ff(["-i", looped, "-c:v", "h264_nvenc", "-preset", "p7", "-tune", "hq", "-rc", "vbr",
        "-cq", "26", "-b:v", "0", "-maxrate", "6M", "-bufsize", "12M",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", mp4])
    ff(["-i", looped, "-c:v", "libvpx-vp9", "-crf", "38", "-b:v", "0",
        "-row-mt", "1", "-pix_fmt", "yuv420p", "-an", webm])
    ff(["-i", looped, "-ss", "1", "-frames:v", "1", "-q:v", "5", poster])

    for f in (mp4, webm, poster):
        print(f"  {os.path.basename(f)}: {os.path.getsize(f)/1e6:.2f} MB")
    return mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", default="v1")
    ap.add_argument("--assemble", action="store_true", help="skip generation")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "site", "assets", "video")
    out_dir = os.path.abspath(out_dir)

    if not args.assemble:
        for s in SHOTS:
            if run(s, args.run_id) is None:
                return 1
    print("\nassembling the loop")
    assemble(args.run_id, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
