import cv2
import time
import threading
import os
import glob
import shutil
import logging
import subprocess
import urllib.request
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template_string, send_from_directory

"""
Pet Cam - Motion Activated Recorder + Toggelable Live Stream
-------------------------------------------------------------
Turns a Raspberry Pi 2B (dietpi) + one or more USB webcams into a pet monitor.

- Always-on motion detection (via frame differencing). records .mp4 clips.
- Recording stops 3s after motion ends.
- Optional live MJPEG stream, off by default to save cpu resources.
- Web dashboard: start/stop stream, view live feeds, play/delete/download clips.
- Old clips are deleted automatically (see RETENTION_DAYS).

Run:      python3 pet_cam.py
Open:     http://<host-ip>:5000/

If using two cameras, set CAMERA_INDICES = [0, 1] below.
"""

# --- Configuration ---
CAMERA_INDICES = [0]        
# e.g. [0, 1] for two cameras, depending on where your camera is. 
# In my setup, /dev/video0 is camera 1 and /dev/video1 is metadata for that camera (i think), 
# so 2 is my second camera. Run `v4l2-ctl --list-devices` to verify, or just try different notes in 
# /dev/video* until you find the right camera.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 20.0
MOTION_TIMEOUT = 3          # seconds of stillness before recording stops
MOTION_THRESHOLD = 5000     # changed pixels needed to count as motion, tunable from the dashboard
PREBUFFER_SECONDS = 2       # footage kept before motion so a clip leads into the action
RETENTION_DAYS = 2          # delete clips older than this
RECORDINGS_DIR = "."
NTFY_TOPIC = ""             # ntfy.sh topic for a phone push on motion, "" disables it

app = Flask(__name__)
stream_enabled = False       # global: streaming on/off for all cameras

# opencv on the pi can't encode h.264, so it falls back to mp4v (mpeg-4 part 2)
# which no browser plays.  when ffmpeg is on PATH we pipe frames to it and
# get real h.264 instead. None means it's missing, we warn and fall back to opencv.
FFMPEG = shutil.which("ffmpeg")

# the dashboard polls /status twice a second per tab, so werkzeug's per-request
# logs flood the console and bury our own prints. drop them, real errors still show.
logging.getLogger("werkzeug").setLevel(logging.WARNING)


def notify_motion(index):
    """send a phone push via ntfy.sh when a recording starts. does nothing if NTFY_TOPIC is unset."""
    if not NTFY_TOPIC:
        return

    def send():
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=f"Motion detected on cam {index}".encode(),
                headers={"Title": "Pet Cam", "Tags": "cat"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            # a network hiccup should never stall the capture loop.
            print("Notify error:", e)

    # send off-thread so the push request can't block frame capture.
    threading.Thread(target=send, daemon=True).start()


class FfmpegWriter:
    """Pipe raw BGR frames to ffmpeg and get back browser-playable h.264.

    drop-in for cv2.VideoWriter, same write()/release() the loop already calls.
    frames have to match (width, height) or the raw pipe desyncs.
    """

    def __init__(self, path, size, fps):
        w, h = size
        self.proc = subprocess.Popen(
            [
                FFMPEG, "-hide_banner", "-loglevel", "error", "-y",
                # raw BGR frames come in on stdin at our capture size and rate.
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
                "-an",                              # no audio
                "-c:v", "libx264",
                # ultrafast/zerolatency since the pi encodes in software. if it
                # still can't keep up, drop FPS or FRAME_* up top.
                "-preset", "ultrafast", "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",              # 4:2:0, browsers need this to play it
                "-movflags", "+faststart",          # moov atom up front so it can start and seek
                path,
            ],
            stdin=subprocess.PIPE,
        )

    def write(self, frame):
        try:
            self.proc.stdin.write(frame.tobytes())
        except (BrokenPipeError, OSError):
            # ffmpeg died mid-clip, nothing we can do from in here but push forward.
            pass

    def release(self):
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=10)   # give it a sec to flush and write the moov atom
        except subprocess.TimeoutExpired:
            self.proc.kill()


class Camera:
    """One webcam: captures frames, detects motion, records clips. One of these per camera."""

    def __init__(self, index):
        self.index = index
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        # use the size the camera actually hands back, otherwise recordings can corrupt.
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or FRAME_HEIGHT

        self.frame = None
        self.motion = False
        self.recording = False
        self.writer = None
        self.last_motion_time = 0
        self.last_changed = 0        # most recent changed-pixel count, shown on the dashboard
        self.buffer = deque(maxlen=int(FPS * PREBUFFER_SECONDS))
        self.lock = threading.Lock()

        ok, first = self.cap.read()
        self.prev_gray = self._blur_gray(first) if ok else None

        # say what actually opened. a usb webcam often has a second /dev/video
        # node that opens fine but never hands back a frame, so a wrong index
        # just shows a blank feed with no error. flag it.
        if not self.cap.isOpened() or not ok:
            print(f"[cam{self.index}] WARNING: opened but got no frame, wrong index? "
                  f"try `v4l2-ctl --list-devices` to find the right one.")
        else:
            print(f"[cam{self.index}] ready at {self.width}x{self.height}")

    @staticmethod
    def _blur_gray(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # lighter blur than 21x21 so a small/distant cat isn't smeared away
        # before we diff. stilll enough to kill sensor noise.
        return cv2.GaussianBlur(gray, (11, 11), 0)

    def _open_writer(self):
        name = datetime.now().strftime(f"motion_cam{self.index}_%Y%m%d_%H%M%S.mp4")
        path = os.path.join(RECORDINGS_DIR, name)
        size = (self.width, self.height)
        print(f"[cam{self.index}] recording: {name}")

        # ffmpeg gives real h.264 that browsers can play, use it if we have it.
        if FFMPEG:
            return FfmpegWriter(path, size, FPS)

        # no ffmpeg, fall back to opencv. 'avc1' is h.264 if the build has it,
        # else 'mp4v', which records fine but won't play in a browser.
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), FPS, size)
        if not writer.isOpened():
            print(f"[cam{self.index}] WARNING: no ffmpeg and opencv has no h.264 encoder, "
                  "clip will be mp4v and probably won't play in the browser. "
                  "install ffmpeg: sudo apt install ffmpeg")
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        return writer

    def _detect_motion(self, frame):
        gray = self._blur_gray(frame)
        if self.prev_gray is None:
            self.prev_gray = gray
            return False
        diff = cv2.absdiff(self.prev_gray, gray)
        thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
        self.prev_gray = gray
        # frame-to-frame diff scales with speed, so a sprint spikes this far above a
        # walk. the dashboard shows this live so I can park the threshold between them.
        self.last_changed = cv2.countNonZero(thresh)
        return self.last_changed > MOTION_THRESHOLD

    def _stamp(self, frame, when):
        # burn the capture time into a copy, the raw frame is shared with the stream.
        stamped = frame.copy()
        label = datetime.fromtimestamp(when).strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(stamped, label, (10, self.height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return stamped

    def loop(self):
        """Capture, detect motion, and record. Runs until process is killed in a thread."""
        while True:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.1)
                continue

            now = time.time()
            with self.lock:
                self.frame = frame
            self.buffer.append((now, frame))   # rolling pre-motion window

            self.motion = self._detect_motion(frame)

            if self.motion:
                self.last_motion_time = now
                if not self.recording:
                    self.writer = self._open_writer()
                    self.recording = True
                    notify_motion(self.index)
                    # flush the buffer first so the clip leads into the action.
                    for t, f in list(self.buffer):
                        self.writer.write(self._stamp(f, t))
                    continue

            if self.recording:
                self.writer.write(self._stamp(frame, now))
                if now - self.last_motion_time > MOTION_TIMEOUT:
                    self.writer.release()
                    self.recording = False
                    print(f"[cam{self.index}] stopped recording")

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


cameras = {i: Camera(i) for i in CAMERA_INDICES}


def mjpeg_stream(cam):
    """Yield JPEG frames for one camera while streaming is enabled."""
    while True:
        if not stream_enabled:
            time.sleep(0.1)
            continue

        frame = cam.get_frame()
        if frame is None:
            time.sleep(0.05)
            continue

        if cam.motion:
            cv2.putText(frame, "MOTION DETECTED", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")


def cleanup_old_videos():
    """Delete clips older than RETENTION_DAYS. Run every hour in a thread."""
    max_age = RETENTION_DAYS * 86400
    while True:
        now = time.time()
        for path in glob.glob(os.path.join(RECORDINGS_DIR, "*.mp4")):
            try:
                if now - os.path.getmtime(path) > max_age:
                    print("Deleting old video:", path)
                    os.remove(path)
            except OSError as e:
                print("Cleanup error:", e)
        time.sleep(3600)


# ----------------------------------
# Dashboard
# ---------------------------------
DASHBOARD = """
<!DOCTYPE html>
<html>
<head>
    <title>Pet Cam</title>
    <style>
        body { font-family: Arial; background: #222; color: #eee; text-align: center; }
        button {
            padding: 15px 30px; margin: 20px; font-size: 20px;
            border: none; border-radius: 8px; cursor: pointer; color: #fff;
        }
        #start { background: #4CAF50; }
        #stop { background: #E53935; }
        .feed { margin: 10px; max-width: 90%; cursor: pointer; }
        .feed.fs {
            position: fixed; top: 0; left: 0;
            width: 100vw; height: 100vh;
            margin: 0; max-width: none;
            object-fit: contain; background: #000; z-index: 1000;
        }
        a { color: #4FC3F7; }
        /* the grey help text. centered block, but left-aligned inside so the
           longer lines don't go ragged. */
        .hint {
            color: #888; font-size: 14px; line-height: 1.5;
            max-width: 620px; margin: 8px auto 0; text-align: left;
        }
        /* slider end labels, same width as the slider so they line up. */
        .ends {
            width: 60%; margin: 4px auto 8px; display: flex;
            justify-content: space-between; color: #888; font-size: 13px;
        }
    </style>
</head>
<body>
    <h1>Pet Cam Dashboard</h1>
    <button id="start" onclick="toggleStream('enable')">Start Stream</button>
    <button id="stop" onclick="toggleStream('disable')">Stop Stream</button>

    <h2>Motion Sensitivity</h2>
    <div id="tuning">
        <div class="hint" style="margin-bottom:6px;">
            How much has to change between two frames before a clip starts recording.
        </div>
        <input type="range" id="threshold" min="500" max="50000" step="250"
               value="{{ threshold }}" oninput="setThreshold(this.value)" style="width:60%;">
        <div class="ends">
            <span>&#9664; more sensitive</span><span>less sensitive &#9654;</span>
        </div>
        <div>trigger above <span id="threshVal">{{ threshold }}</span> changed pixels</div>

        <div class="hint">
            <strong>Decrease</strong> to catch slower movement: more clips, but
            shadows and light changes can set it off.<br>
            <strong>Increase</strong> to catch only fast movement like zoomies:
            fewer clips, but a slow wander may be missed.
        </div>

        <div id="live" style="margin-top:12px; color:#4FC3F7; font-size:18px;"></div>
        <div class="hint">
            That live count is how much changed since the last frame, so it tracks
            <em>speed</em>, not size. A cat sitting still reads near zero.
            Watch it with the room quiet, then while the cat runs, and park the slider
            between the two.<br>
            Changes apply instantly but reset when the script restarts. To keep a value,
            set MOTION_THRESHOLD near the top of pet_cam.py.
        </div>
    </div>

    <h2>Live Stream</h2>
    <div id="feeds">
        {% for i in indices %}
            <img class="feed" id="feed{{ i }}" src="" width="640" alt="Camera {{ i }}"
                 onclick="toggleFullscreen(this)">
        {% endfor %}
    </div>
    <p class="hint" style="text-align:center;">Tap a live feed to toggle fullscreen.</p>

    <h2>Recordings</h2>
    <div id="recordings"></div>

<script>
const CAMERAS = {{ indices | tojson }};

function toggleStream(action) {
    fetch('/' + action).then(() => {
        CAMERAS.forEach(i => {
            document.getElementById('feed' + i).src =
                (action === 'enable') ? '/stream/' + i : '';
        });
    });
}

// iphone safari won't fullscreen an <img>, so fake it with a full-viewport css class.
function toggleFullscreen(el) {
    el.classList.toggle('fs');
}

function loadRecordings() {
    fetch('/recordings').then(r => r.text())
        .then(html => { document.getElementById('recordings').innerHTML = html; });
}

function deleteRecording(name) {
    if (!confirm("Delete " + name + "?")) return;
    fetch('/delete/' + name).then(() => loadRecordings());
}

function setThreshold(v) {
    document.getElementById('threshVal').textContent = v;
    fetch('/threshold/' + v);
}

function pollStatus() {
    fetch('/status').then(r => r.json()).then(s => {
        document.getElementById('live').innerHTML =
            CAMERAS.map(i => 'cam ' + i + ': ' + s.cameras[i] + ' changed pixels').join('<br>');
    });
}
setInterval(pollStatus, 500);
pollStatus();

loadRecordings();
</script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    return render_template_string(DASHBOARD, indices=list(cameras), threshold=MOTION_THRESHOLD)


@app.route("/status")
def status():
    # live changed-pixel counts, so the dashboard can show motion while I tune settings.
    return {"cameras": {i: cam.last_changed for i, cam in cameras.items()}}


@app.route("/threshold/<int:value>")
def set_threshold(value):
    global MOTION_THRESHOLD
    MOTION_THRESHOLD = value
    return "OK"


@app.route("/enable")
def enable_stream():
    global stream_enabled
    stream_enabled = True
    return "Streaming enabled"


@app.route("/disable")
def disable_stream():
    global stream_enabled
    stream_enabled = False
    return "Streaming disabled"


@app.route("/stream/<int:cam_id>")
def stream(cam_id):
    cam = cameras.get(cam_id)
    if cam is None:
        return "Unknown camera", 404
    return Response(mjpeg_stream(cam),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/recordings")
def recordings():
    files = sorted(
        (f for f in os.listdir(RECORDINGS_DIR) if f.endswith(".mp4")),
        reverse=True,
    )
    items = []
    for f in files:
        size_mb = os.path.getsize(os.path.join(RECORDINGS_DIR, f)) / (1024 * 1024)
        items.append(f"""
        <li style='margin-bottom:20px; list-style:none;'>
            <strong>{f}</strong> ({size_mb:.2f} MB)<br>
            <video width="320" controls preload="metadata">
                <source src="/media/{f}" type="video/mp4">
            </video><br>
            <button onclick="deleteRecording('{f}')"
                    style="background:#E53935;color:#fff;padding:8px 15px;border:none;border-radius:5px;margin-top:5px;">
                Delete
            </button>
            <a href="/download/{f}" download
               style="display:inline-block;background:#4FC3F7;color:#000;padding:8px 15px;border-radius:5px;margin-top:5px;text-decoration:none;">
                Download
            </a>
        </li>""")
    return "<ul style='padding:0;'>" + "".join(items) + "</ul>"


def _recording_path(filename):
    """Turn a clip name into a safe path inside RECORDINGS_DIR, or None if it isn't one.

    RECORDINGS_DIR is the app's own folder, so this has to block more than ../
    traversal. without the .mp4 check, /delete/pet_cam.py would wipe this file.
    """
    if not filename.endswith(".mp4"):
        return None
    root = os.path.realpath(RECORDINGS_DIR)
    path = os.path.realpath(os.path.join(root, filename))
    # clips all sit in one flat folder, so the resolved parent has to be exactly
    # root. realpath means a symlink pointing outside fails this too.
    if os.path.dirname(path) != root:
        return None
    return path


@app.route("/media/<filename>")
def media_file(filename):
    if _recording_path(filename) is None:
        return "Not a recording", 404
    # serve inline with range support so <video> can play and seek.
    return send_from_directory(RECORDINGS_DIR, filename, conditional=True)


@app.route("/download/<filename>")
def download_file(filename):
    if _recording_path(filename) is None:
        return "Not a recording", 404
    return send_from_directory(RECORDINGS_DIR, filename, as_attachment=True)


@app.route("/delete/<filename>")
def delete_file(filename):
    path = _recording_path(filename)
    if path is None:
        return "Not a recording", 404
    try:
        os.remove(path)
        return "OK"
    except OSError as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    for cam in cameras.values():
        threading.Thread(target=cam.loop, daemon=True).start()
    threading.Thread(target=cleanup_old_videos, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
