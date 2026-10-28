# Vibe Check - SBHacks 1st Place: Entertainment Track

![Image not loading](VibeCheckDashboard.jpg)

## **What it does**
My goal is to accurately provide the DJ/manager with real-time data about energy levels in the crowd, where, in the frame of reference, the crowd is more "hype", and to sync different processes based on crowd engagement and preference. One of these features includes accessing attendees' Spotify data with their permission to extract crowd favorite genres, which are ranked in real-time by popularity. Another important feature I chose to integrate was synchronizing lights in a venue to match the energy levels of the crowd.

## **How I built it**

I built Vibe Check by starting from the core of this project, which was **optical flow tracking with OpenCV**. I ran this and tested its accuracy in low-light settings to assess its feasibility with my vision. Then, I worked to extract the intensity of movements from the **Farneback optical flow** method so that I could vectorize each pixel in successive frames and **measure movement**. After this, I added weights so that I could calculate a real score for **hype and energy levels**. Then, I split the frame into a grid and figured out where in the grid energy movement was coming from. This allowed me to create a **heatmap of energy** in the frame. 

To display this, I built a **React user interface** and used **WebSocket** to host it seamlessly on my local machine. I then connected this data to an **InfluxDB** database in which I could store energy levels by timestamp and per frame for later use. From here, I brainstormed additional features to add. For example, I thought to add a feature that allows attendees to scan a QR code and then give permission to access their Spotify listening data in exchange for a voucher of sorts. I have shown this as a proof of concept on the Vibe Check dashboard by simulating a demo stream of check-ins and **popular genre rankings** for the DJ to adapt to. Another feature I thought to demonstrate was integrating the energy level with venue lighting systems so that they can adapt accordingly.

## Build and test the motion engine

The `vibe_core` extension requires a C++17 compiler, CMake 3.16 or newer,
Python development headers, pybind11, and the OpenCV C++ development libraries
(`core`, `imgproc`, and `video`). The Python `opencv-python` package alone does
not provide the C++ build dependencies. If CMake cannot locate OpenCV, add
`-DOpenCV_DIR=/path/to/the/directory/containing/OpenCVConfig.cmake` to the
configure command.

Activate a Python environment with `numpy`, `pybind11`, and `aiohttp` installed,
then run these commands from the repository root. The status-handler tests use
`aiohttp`; they do not start a server. The extension is built for that
environment's Python interpreter; rebuild it when changing Python versions.

```sh
repo_dir="$PWD"
python_bin="$(python -c 'import sys; print(sys.executable)')"
build_dir="$(mktemp -d)"

cmake -S "$repo_dir" -B "$build_dir" \
  -DPYBIND11_FINDPYTHON=ON \
  -DPython_EXECUTABLE="$python_bin" \
  -Dpybind11_DIR="$("$python_bin" -m pybind11 --cmakedir)"
cmake --build "$build_dir" --parallel 2

(
  cd "$build_dir" &&
  PYTHONPATH="$build_dir" "$python_bin" -B -c \
    'import vibe_core; print(vibe_core.__file__)' &&
  PYTHONPATH="$build_dir" "$python_bin" -B -m unittest discover \
    -s "$repo_dir/tests" -v
)
```

The printed module path should be inside `$build_dir`. Running tests there
avoids importing an older, checked-in `vibe_core` binary from the repository
root. These tests exercise the motion engine with synthetic frames; they do
not require a camera or start the backend or dashboard.

## Camera configuration

Camera settings are constants in `main.py`; they are not command-line options.
`CAM_SOURCE` defaults to `1`. Set it to `0` to try the first camera, or to
`"auto"` on macOS to probe indices 0 through 9 using AVFoundation and select
the first camera that returns a frame. If none is readable, auto-detection
raises `Could not find an available camera`.

For an explicit camera index, the backend tries AVFoundation first, then
OpenCV's default backend if opening fails. Close other apps using the camera
and check camera permission for the application running Python if capture
cannot start.

`PREFERRED_RESOLUTION` defaults to `(1280, 720)` and `PREFERRED_FPS` to `30`.
These are requests to the capture device; unsupported settings may be ignored.
The startup `Resolution:` line reports the dimensions read back from the device.
