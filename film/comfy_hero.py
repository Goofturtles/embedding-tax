"""Generate the hero film clips on the local 4090 via ComfyUI + LTX-Video.

The landing page hand-off follows Owed's: a dark scroll-scrubbed film that the white
page rides up over. These clips are ATMOSPHERE, not information -- the exact parameter
argument is made further down the page by a deterministic canvas, where it can be
precise. Here we only need something dark, slow and abstract.

Deliberately non-figurative: no people, no objects, no text. Generated video falls
apart on hands and faces, and none of that is needed for light in dark space.

PRECONDITIONS
  * ComfyUI running on 127.0.0.1:8188 (run_nvidia_gpu.bat; needs 60-120s to boot).
  * models/checkpoints/ltx-video-2b-v0.9.5.safetensors  (diffusion model + VAE)
  * models/clip/t5xxl_fp8_e4m3fn.safetensors            (text encoder -- the
    checkpoint does NOT contain one, so CheckpointLoaderSimple's CLIP output is None)

Usage:
    python comfy_hero.py --list
    python comfy_hero.py --shot scatter
    python comfy_hero.py --shot all
Exits nonzero if any shot fails, so a calling harness cannot report a green run
over three silent failures.
"""
import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

HOST = "http://127.0.0.1:8188"
CKPT = "ltx-video-2b-v0.9.5.safetensors"
T5 = "t5xxl_fp8_e4m3fn.safetensors"

WIDTH, HEIGHT = 960, 544        # this framing yields suspended tendrils;
                                # 1280x704 grounds the subject into a lit mass
LENGTH = 121                    # must be 8n+1; 121 frames at 24fps = ~5.0s
FPS = 24
STEPS = 30
CFG = 3.0

POLL_SECONDS = 6
JOB_TIMEOUT = 1800
HTTP_TIMEOUT = 60

assert (LENGTH - 1) % 8 == 0, f"LTX needs length 8n+1, got {LENGTH}"
assert WIDTH % 32 == 0 and HEIGHT % 32 == 0, "LTX needs dimensions that are multiples of 32"

NEG = ("text, letters, words, numbers, watermark, logo, caption, subtitles, "
       "people, person, face, hands, fingers, limbs, animals, "
       "blurry, out of focus, jpeg artifacts, compression, noise, "
       "oversaturated, neon, rainbow, fast motion, camera shake, jitter, warping, "
       "ground, floor, surface, table, horizon line, fire, flame, candle")

# Smooth, low-frequency, organic motion. A 2B model smears fine geometric detail, so
# asking for a crisp lattice of particles buys mush; ink, smoke and refracted light
# are what this size actually renders cleanly.
SHOTS = [
    {
        "name": "bloom",
        "seed": 20260908,
        "prompt": (
            "Extreme macro of a single drop of luminous amber ink blooming slowly through "
            "still black water, soft tendrils unfurling, shallow depth of field, "
            "volumetric backlight, very slow graceful motion, cinematic, film grain, "
            "anamorphic lens, 35mm, dark background"
        ),
    },
    {
        "name": "drift",
        "seed": 20260909,
        "prompt": (
            "Extreme macro of deep teal ink diffusing through still black water, fine "
            "thread-like tendrils spreading and interweaving into a dense structure, "
            "high contrast, shallow depth of field, volumetric backlight, very slow "
            "deliberate motion, cinematic, film grain, anamorphic lens, 35mm"
        ),
    },
    {
        "name": "settle",
        "seed": 20260910,
        "prompt": (
            "Extreme macro of a dense billowing cloud of teal ink suspended in black "
            "water, thick luminous plumes slowly unfurling and folding into each other, "
            "bright core fading to dark edges, high contrast, shallow depth of field, "
            "volumetric backlight, slow graceful motion, cinematic, film grain, "
            "anamorphic lens, 35mm"
        ),
    },
]


def build(shot, run_id):
    """The LTX graph in ComfyUI API format, matching the reference t2v workflow.

    Node 1 stays even though its CLIP output is unused: it supplies MODEL and VAE.
    LTXVConditioning returns (positive, negative), hence ["4", 0] and ["4", 1].
    SamplerCustom + LTXVScheduler is the reference stack; plain KSampler runs but
    produces a washed-out result because the sigma schedule is wrong for LTX.
    """
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "9": {"class_type": "CLIPLoader", "inputs": {"clip_name": T5, "type": "ltxv"}},
        "2": {"class_type": "CLIPTextEncode",
              "inputs": {"text": shot["prompt"], "clip": ["9", 0]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": NEG, "clip": ["9", 0]}},
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
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 1], "vae": ["1", 2]}},
        # Run-scoped prefix: reruns land in their own folder instead of interleaving
        # frame numbers with the previous take inside one directory.
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0],
                         "filename_prefix": f"etax/{run_id}/{shot['name']}"}},
    }


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(f"{HOST}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # ComfyUI returns 400 with a JSON node_errors body naming the offending node.
        # Swallowing it is what made the missing text encoder expensive to find.
        body = e.read().decode("utf-8", "replace")
        print(f"    HTTP {e.code} from {path}:")
        try:
            print(json.dumps(json.loads(body), indent=2)[:2500])
        except json.JSONDecodeError:
            print(f"      {body[:1200]}")
        return None
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"    cannot reach ComfyUI at {HOST}: {e}")
        print("    is it running? python_embeded\\python.exe -s ComfyUI\\main.py")
        return None


def get(path):
    """Returns None on any transport hiccup so a live render is not aborted by one
    dropped poll."""
    try:
        with urllib.request.urlopen(f"{HOST}{path}", timeout=HTTP_TIMEOUT) as r:
            return json.loads(r.read())
    except (urllib.error.URLError, urllib.error.HTTPError,
            TimeoutError, json.JSONDecodeError) as e:
        print(f"    poll hiccup ({type(e).__name__}), retrying")
        return None


def run(shot, run_id):
    print(f"\n=== {shot['name']}  seed {shot['seed']} ===")
    print(f"    {WIDTH}x{HEIGHT}, {LENGTH} frames @ {FPS}fps, {STEPS} steps, cfg {CFG}")

    res = post("/prompt", {"prompt": build(shot, run_id)})
    if not res or "prompt_id" not in res:
        print("    could not queue")
        return None
    pid = res["prompt_id"]
    print(f"    queued: {pid}")

    t0 = time.time()
    while time.time() - t0 < JOB_TIMEOUT:
        time.sleep(POLL_SECONDS)
        hist = get(f"/history/{pid}")
        if not hist or pid not in hist:
            continue

        entry = hist[pid]
        status = entry.get("status", {})
        state = status.get("status_str")

        if state == "error":
            print(f"    FAILED after {time.time()-t0:.0f}s")
            for m in status.get("messages", []):
                if m and m[0] == "execution_error":
                    d = m[1]
                    print(f"      node {d.get('node_id')} ({d.get('node_type')}): "
                          f"{d.get('exception_message', '').strip()}")
            return None

        # Anything that is not an explicit success means keep waiting. Treating an
        # unknown or empty status as success is how a cancelled job reports green.
        if not (state == "success" and status.get("completed") is True):
            continue

        imgs = []
        for out in entry.get("outputs", {}).values():
            for i in out.get("images", []):
                imgs.append("/".join(p for p in (i.get("subfolder", ""), i.get("filename", "")) if p))
        if len(imgs) != LENGTH:
            print(f"    WRONG FRAME COUNT after {time.time()-t0:.0f}s: "
                  f"got {len(imgs)}, expected {LENGTH}")
            return None
        print(f"    done in {time.time()-t0:.0f}s -> {len(imgs)} frames")
        return imgs

    print(f"    timed out after {JOB_TIMEOUT}s (job may still be on the GPU)")
    return None


def main():
    names = [s["name"] for s in SHOTS]
    ap = argparse.ArgumentParser()
    ap.add_argument("--shot", default="all", choices=["all", *names])
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"),
                    help="output subfolder under etax/, keeps reruns separate")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in SHOTS:
            print(f"{s['name']:>8}: {s['prompt'][:76]}...")
        return 0

    shots = SHOTS if args.shot == "all" else [s for s in SHOTS if s["name"] == args.shot]
    manifest, failed = {}, []
    for s in shots:
        frames = run(s, args.run_id)
        if frames is None:
            failed.append(s["name"])
        else:
            manifest[s["name"]] = frames

    if manifest:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"hero_manifest_{args.run_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"run_id": args.run_id, "width": WIDTH, "height": HEIGHT,
                       "fps": FPS, "length": LENGTH, "shots": manifest}, f, indent=2)
        print(f"\nmanifest -> {path}")

    if failed:
        print(f"\nFAILED: {', '.join(failed)}")
        return 1
    print(f"\nall {len(manifest)} shot(s) ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
