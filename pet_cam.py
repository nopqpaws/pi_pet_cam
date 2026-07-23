import cv2
import time
import threading
import os
import glob
import urllib.request
from collections import deque
from datetime import datetime
from flask import Flask, Response, render_template_string, send_from_directory

"""
Waffle Cam - Motion Activated Recorder + Toggelable Live Stream
-------------------------------------------------------------
Turns a Raspberry Pi 2B (dietpi) + one or more USB webcams into a pet monitor.

- Always-on motion detection (via frame differencing). records .mp4 clips.
- Recording stops 3s after motion ends.
- Optional live MJPEG stream, off by default to save cpu resources.
- Web dashboard: start/stop stream, view live feeds, play/delete/download clips.
- Old clips are deleted automatically (see RETENTION_DAYS).

Run:      python3 pet_cam.py
Open:     http://<raspberry-pi-ip>:5000/

If using two cameras, set CAMERA_INDICES = [0, 1] below.
"""

# --- Configuration ---
CAMERA_INDICES = [0]        # e.g. [0, 1] for two cameras
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


def notify_motion(index):
    """Run a phone push via ntfy.sh when a recording starts. no-op if NTFY_TOPIC is unset."""
    if not NTFY_TOPIC:
        return

    def send():
        try:
            req = urllib.request.Request(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=f"Motion detected on cam {index}".encode(),
                headers={"Title": "Waffle Cam", "Tags": "cat"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            # a network hiccup should never stall the capture loop.
            print("Notify error:", e)

    # send off-thread so the push request can't block frame capture.
    threading.Thread(target=send, daemon=True).start()


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

        # 'avc1' (h.264) plays in browsers, fall back to 'mp4v' if it's not available.
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"avc1"), FPS, size)
        if not writer.isOpened():
            writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, size)
        print(f"[cam{self.index}] recording: {name}")
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
    <title>Waffle Cam</title>
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
    </style>
</head>
<body>
    <h1>Waffle Cam Dashboard</h1>
    <button id="start" onclick="toggleStream('enable')">Start Stream</button>
    <button id="stop" onclick="toggleStream('disable')">Stop Stream</button>

    <h2>Motion Sensitivity</h2>
    <div id="tuning">
        <input type="range" id="threshold" min="500" max="50000" step="250"
               value="{{ threshold }}" oninput="setThreshold(this.value)" style="width:60%;">
        <div>trigger above <span id="threshVal">{{ threshold }}</span> changed pixels</div>
        <div id="live" style="margin-top:8px; color:#4FC3F7;"></div>
    </div>

    <h2>Live Stream</h2>
    <div id="feeds">
        {% for i in indices %}
            <img class="feed" id="feed{{ i }}" src="" width="640" alt="Camera {{ i }}"
                 onclick="toggleFullscreen(this)">
        {% endfor %}
    </div>
    <p style="color:#888; font-size:14px;">Tap a live feed to toggle fullscreen.</p>

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
    # live changed-pixel counts so the dashboard can show motion while I tune the threshold.
    return {
        "threshold": MOTION_THRESHOLD,
        "cameras": {i: cam.last_changed for i, cam in cameras.items()},
    }


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


@app.route("/media/<path:filename>")
def media_file(filename):
    # serve inline with range support so <video> can play and seek.
    return send_from_directory(RECORDINGS_DIR, filename, conditional=True)


@app.route("/download/<path:filename>")
def download_file(filename):
    return send_from_directory(RECORDINGS_DIR, filename, as_attachment=True)


@app.route("/delete/<path:filename>")
def delete_file(filename):
    try:
        os.remove(os.path.join(RECORDINGS_DIR, filename))
        return "OK"
    except OSError as e:
        return f"ERROR: {e}"


if __name__ == "__main__":
    for cam in cameras.values():
        threading.Thread(target=cam.loop, daemon=True).start()
    threading.Thread(target=cleanup_old_videos, daemon=True).start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
