# =============================================================================
# DRONE DETECTION & TRACKING SYSTEM - JETSON NANO ORIN
# Electrical Engineering Senior Project
# Dual CSI Camera + YOLOv11 + TensorRT + Kalman Filter Tracking
# =============================================================================

# =============================================================================
# SECTION 1: IMPORTS & DEPENDENCIES
# =============================================================================

    # Standard Library Imports
    # Libraries: os, sys, time, datetime, threading, logging,
    #            argparse, json, signal, re, csv

    # Computer Vision Imports
    # Libraries: cv2 (OpenCV), numpy

    # Deep Learning Imports
    # Libraries: ultralytics (YOLO), torch

    # Scientific Computing (for Tracker)
    # Libraries: scipy (linear_sum_assignment, spatial.distance),
    #            numpy (matrix operations)

    # Utilities
    # Libraries: pathlib, collections (deque, OrderedDict),
    #            dataclasses, math


# =============================================================================
# SECTION 2: CONFIGURATION & CONSTANTS
# =============================================================================

    # ── 2.1 Camera Configuration ─────────────────────────────────────────────
    # Define sensor IDs (0 and 1), capture resolution (width, height),
    # target framerate, and GStreamer flip method
    # Libraries: dataclasses

    # ── 2.2 Model Configuration ──────────────────────────────────────────────
    # Define TensorRT engine path, confidence threshold,
    # IoU threshold, input image size, and class names list
    # Libraries: dataclasses, os

    # ── 2.3 Tracker Configuration ────────────────────────────────────────────
    # Define tracking parameters:
    #   max_disappeared  → frames before a track is dropped
    #   max_distance     → max IoU cost to accept an assignment
    #   min_hits         → minimum detections before track is confirmed
    #   history_length   → number of past positions to store per track
    #                      (used for trajectory trail visualization)
    # Libraries: dataclasses

    # ── 2.4 Kalman Filter Configuration ──────────────────────────────────────
    # Define Kalman filter matrix dimensions and tuning parameters:
    #   state_dim        → size of state vector [x,y,w,h,dx,dy] = 6
    #   measurement_dim  → size of measurement vector [x,y,w,h] = 4
    #   process_noise    → how much we trust the motion model (Q matrix)
    #   measurement_noise→ how much we trust the YOLO detections (R matrix)
    # Libraries: dataclasses, numpy

    # ── 2.5 Detection Configuration ──────────────────────────────────────────
    # Define post-processing parameters:
    #   max_detections   → maximum detections to process per frame
    #   min_box_area     → minimum bounding box area in pixels
    #                      (filters out noise detections)
    # Libraries: dataclasses

    # ── 2.6 Position Estimation Configuration ────────────────────────────────
    # Define parameters for estimating drone position in the scene:
    #   frame_width, frame_height → camera resolution
    #   fov_horizontal, fov_vertical → camera field of view in degrees
    #   reference_drone_size → known real-world drone size in meters
    #                          (used for distance estimation)
    # Libraries: dataclasses, math

    # ── 2.7 Display Configuration ────────────────────────────────────────────
    # Define display resolution, font scale, bounding box colors
    # keyed by track state (confirmed, tentative, lost),
    # trajectory line color and thickness, HUD layout settings
    # Libraries: dataclasses, cv2

    # ── 2.8 Logging & Output Configuration ───────────────────────────────────
    # Define log directory path, detection CSV file path,
    # image save directory, session report path,
    # and whether to save detection images to disk
    # Libraries: dataclasses, os, pathlib


# =============================================================================
# SECTION 3: LOGGER SETUP
# =============================================================================

    # ── 3.1 System Logger Initialization ─────────────────────────────────────
    # Configure root logger with format, log level,
    # and dual output handlers (console + rotating log file)
    # Libraries: logging, datetime, os

    # ── 3.2 Detection Event Logger ────────────────────────────────────────────
    # Separate dedicated logger for detection and tracking events only.
    # Logs: timestamp, camera_id, track_id, confidence,
    #       bounding box, estimated position, track state
    # Libraries: logging, json, datetime


# =============================================================================
# SECTION 4: GSTREAMER PIPELINE BUILDER
# =============================================================================

    # ── 4.1 CSI Camera Pipeline Builder ──────────────────────────────────────
    # Function that constructs and returns a GStreamer pipeline string
    # for a given sensor_id, resolution, framerate, and flip method.
    # Pipeline: nvarguscamerasrc → NVMM memory → nvvidconv → BGR → appsink
    # Libraries: (string formatting only)

    # ── 4.2 Pipeline Validator ────────────────────────────────────────────────
    # Function that tests a GStreamer pipeline string by attempting
    # to open it briefly with OpenCV and verifying a frame is returned.
    # Returns True/False with a descriptive message.
    # Libraries: cv2, subprocess, logging


# =============================================================================
# SECTION 5: CAMERA THREAD CLASS
# =============================================================================

    # class CSICamera
    # Manages one CSI camera in a dedicated background thread.
    # Continuously captures frames into a buffer so the latest
    # frame is always instantly available to the main pipeline
    # without blocking on camera I/O.
    # Libraries: cv2 (CAP_GSTREAMER), threading, time, numpy, logging

        # ── 5.1 __init__ ─────────────────────────────────────────────────────
        # Initialize sensor_id, resolution, fps target,
        # frame buffer (latest frame + timestamp),
        # thread lock, running flag, and performance counters
        # (total_frames, dropped_frames, current_fps)
        # Libraries: threading, numpy, time

        # ── 5.2 _build_pipeline ──────────────────────────────────────────────
        # Call Section 4.1 to construct this camera's GStreamer pipeline string
        # Libraries: (calls Section 4.1)

        # ── 5.3 _open_camera ─────────────────────────────────────────────────
        # Open VideoCapture with the GStreamer pipeline.
        # Verify it opened and is returning frames.
        # Raise RuntimeError with helpful message if it fails.
        # Libraries: cv2

        # ── 5.4 start ────────────────────────────────────────────────────────
        # Start the background daemon capture thread.
        # Return self to allow method chaining.
        # Libraries: threading

        # ── 5.5 _capture_loop ────────────────────────────────────────────────
        # Background thread loop: continuously call cap.read(),
        # store frame and capture timestamp in buffer under lock,
        # increment frame counter, track dropped frames,
        # compute rolling FPS estimate
        # Libraries: cv2, threading, time, collections (deque)

        # ── 5.6 read ─────────────────────────────────────────────────────────
        # Thread-safe read of the latest frame.
        # Returns: (success: bool, frame: numpy array, timestamp: float)
        # Libraries: threading, numpy, time

        # ── 5.7 get_stats ────────────────────────────────────────────────────
        # Return dict of camera performance stats:
        # current_fps, total_frames, dropped_frames, drop_rate, uptime
        # Libraries: time

        # ── 5.8 stop ─────────────────────────────────────────────────────────
        # Set running flag to False, join background thread,
        # release VideoCapture, log shutdown message
        # Libraries: threading, cv2, logging


# =============================================================================
# SECTION 6: DRONE DETECTOR CLASS
# =============================================================================

    # class DroneDetector
    # Loads and manages the YOLOv11 TensorRT model.
    # Accepts raw frames and returns structured detection results.
    # Stateless: each call to detect() is independent,
    # with no memory of previous frames.
    # Libraries: ultralytics (YOLO), torch, numpy, logging, time

        # ── 6.1 __init__ ─────────────────────────────────────────────────────
        # Load model from .engine path using YOLO().
        # Verify CUDA is available.
        # Initialize inference stats counters.
        # Run warmup pass to initialize TensorRT engine.
        # Libraries: ultralytics, torch, numpy, logging

        # ── 6.2 _warmup ──────────────────────────────────────────────────────
        # Run several dummy inference passes on a blank frame
        # to fully initialize the TensorRT engine before
        # live frames arrive (avoids latency spike on first frame)
        # Libraries: numpy, ultralytics

        # ── 6.3 detect ───────────────────────────────────────────────────────
        # Accept a raw BGR numpy frame.
        # Call model.predict() with configured conf and IoU thresholds.
        # Parse results into a list of RawDetection objects.
        # Record inference time.
        # Return: (list of RawDetection, inference_time_ms)
        # Libraries: ultralytics, numpy, time

        # ── 6.4 _parse_results ───────────────────────────────────────────────
        # Convert raw Ultralytics result object into a clean list
        # of RawDetection dataclass instances.
        # Filter out detections below min_box_area threshold.
        # Extract: xyxy coordinates, confidence, class_id, class_name
        # Libraries: ultralytics, numpy

        # ── 6.5 get_inference_stats ──────────────────────────────────────────
        # Return dict of inference performance stats:
        # avg_latency_ms, current_fps, total_inferences,
        # min_latency_ms, max_latency_ms
        # Libraries: collections (deque), numpy


# =============================================================================
# SECTION 7: KALMAN FILTER CLASS
# =============================================================================

    # class KalmanFilter
    # Implements a standard linear Kalman filter for tracking
    # a single drone's bounding box across frames.
    #
    # State vector (6D):
    #   [x_center, y_center, width, height, dx, dy]
    #    position (4)                      velocity (2)
    #
    # Measurement vector (4D):
    #   [x_center, y_center, width, height]
    #   (what YOLO gives us each frame)
    #
    # The filter maintains uncertainty estimates and balances
    # trust between the motion prediction and YOLO measurement.
    # Libraries: numpy

        # ── 7.1 __init__ ─────────────────────────────────────────────────────
        # Initialize all Kalman filter matrices:
        #
        #   F → State Transition Matrix (6×6)
        #       Models constant velocity motion.
        #       Predicts next state from current state.
        #
        #   H → Measurement Matrix (4×6)
        #       Maps state space to measurement space.
        #       Extracts [x,y,w,h] from full state vector.
        #
        #   Q → Process Noise Covariance (6×6)
        #       Uncertainty in the motion model.
        #       Higher Q = trust measurements more.
        #
        #   R → Measurement Noise Covariance (4×4)
        #       Uncertainty in YOLO detections.
        #       Higher R = trust predictions more.
        #
        #   P → State Covariance Matrix (6×6)
        #       Current uncertainty in the state estimate.
        #       Updated every predict/update cycle.
        #
        #   x → State Vector (6×1)
        #       Current best estimate of drone state.
        # Libraries: numpy

        # ── 7.2 initialize_state ─────────────────────────────────────────────
        # Set initial state vector from the first YOLO detection.
        # Convert bounding box [x1,y1,x2,y2] to
        # state format [x_center, y_center, width, height, 0, 0]
        # (velocity initialized to zero)
        # Libraries: numpy

        # ── 7.3 predict ──────────────────────────────────────────────────────
        # PREDICT STEP: project state forward one timestep
        # using the motion model (constant velocity).
        #
        #   x = F · x          (predicted state)
        #   P = F · P · Fᵀ + Q (predicted covariance)
        #
        # Returns predicted bounding box [x1, y1, x2, y2]
        # Libraries: numpy

        # ── 7.4 update ───────────────────────────────────────────────────────
        # UPDATE STEP: correct prediction using new YOLO measurement.
        #
        #   y = z - H · x          (innovation / residual)
        #   S = H · P · Hᵀ + R     (innovation covariance)
        #   K = P · Hᵀ · S⁻¹       (Kalman gain)
        #   x = x + K · y          (corrected state)
        #   P = (I - K · H) · P    (corrected covariance)
        #
        # Accepts measurement z = [x_center, y_center, width, height]
        # converted from YOLO's [x1, y1, x2, y2]
        # Libraries: numpy

        # ── 7.5 get_state ─────────────────────────────────────────────────────
        # Return current state as bounding box [x1, y1, x2, y2]
        # by converting from [x_center, y_center, width, height]
        # Libraries: numpy

        # ── 7.6 get_velocity ──────────────────────────────────────────────────
        # Return current velocity estimate [dx, dy] in pixels per frame
        # Extracted directly from state vector
        # Libraries: numpy

        # ── 7.7 get_uncertainty ───────────────────────────────────────────────
        # Return the diagonal of the covariance matrix P
        # as a measure of current position uncertainty.
        # High uncertainty = track may be unreliable
        # Libraries: numpy


# =============================================================================
# SECTION 8: DRONE TRACKER CLASS
# =============================================================================

    # class DroneTracker
    # Maintains persistent identities for all drones across frames.
    # Receives a list of raw detections each frame and returns
    # the same detections enriched with stable track IDs,
    # Kalman-predicted positions, velocity vectors, and track state.
    #
    # Track States:
    #   TENTATIVE  → newly created, not yet confirmed (< min_hits)
    #   CONFIRMED  → track has been matched consistently
    #   LOST       → unmatched this frame, kept alive by prediction
    #   DELETED    → exceeded max_disappeared, removed from registry
    #
    # Libraries: numpy, scipy, collections, time, logging, dataclasses

        # ── 8.1 __init__ ─────────────────────────────────────────────────────
        # Initialize track registry (OrderedDict keyed by track_id),
        # next_track_id counter, and tracker configuration parameters.
        # Libraries: collections (OrderedDict), time

        # ── 8.2 _compute_iou_matrix ──────────────────────────────────────────
        # Compute pairwise IoU (Intersection over Union) between
        # all predicted track bounding boxes and all new detections.
        # Returns a cost matrix of shape (num_tracks × num_detections).
        # Lower IoU = higher cost = less likely to be the same object.
        # Libraries: numpy

        # ── 8.3 _run_hungarian_assignment ────────────────────────────────────
        # Apply the Hungarian Algorithm to the IoU cost matrix
        # to find the globally optimal one-to-one assignment
        # between existing tracks and new detections.
        # Filter out assignments where IoU cost exceeds max_distance.
        # Returns: matched_pairs, unmatched_tracks, unmatched_detections
        # Libraries: scipy (linear_sum_assignment), numpy

        # ── 8.4 _create_new_track ─────────────────────────────────────────────
        # Create a new track entry for an unmatched detection:
        #   - Assign next available track_id
        #   - Initialize a new KalmanFilter (Section 7)
        #   - Set track state to TENTATIVE
        #   - Initialize position history deque
        #   - Record creation timestamp
        # Libraries: collections (deque), time, dataclasses

        # ── 8.5 _update_matched_track ────────────────────────────────────────
        # For a matched (track, detection) pair:
        #   - Run Kalman UPDATE step with new measurement
        #   - Increment hit counter
        #   - Promote state from TENTATIVE to CONFIRMED if min_hits reached
        #   - Reset disappeared counter to zero
        #   - Append new position to trajectory history
        #   - Update last_seen timestamp
        # Libraries: numpy, time

        # ── 8.6 _handle_lost_track ────────────────────────────────────────────
        # For a track with no matching detection this frame:
        #   - Run Kalman PREDICT step only (no update)
        #   - Set track state to LOST
        #   - Increment disappeared counter
        #   - Mark track for deletion if disappeared > max_disappeared
        # Libraries: numpy, time

        # ── 8.7 _purge_deleted_tracks ─────────────────────────────────────────
        # Remove all tracks marked for deletion from the registry.
        # Log the removal of each deleted track with its lifetime stats.
        # Libraries: collections, logging

        # ── 8.8 update ────────────────────────────────────────────────────────
        # Master update method called once per frame per camera.
        # Orchestrates the full tracking cycle:
        #   1. Run Kalman PREDICT on all active tracks
        #   2. Compute IoU cost matrix
        #   3. Run Hungarian assignment
        #   4. Update matched tracks
        #   5. Create new tracks for unmatched detections
        #   6. Handle lost tracks
        #   7. Purge deleted tracks
        #   8. Return list of TrackedDrone objects
        # Libraries: numpy, scipy, collections, time

        # ── 8.9 get_confirmed_tracks ──────────────────────────────────────────
        # Return only tracks in CONFIRMED state.
        # Used by the alert system and logger to avoid
        # acting on tentative/unreliable tracks.
        # Libraries: collections

        # ── 8.10 get_track_trajectories ───────────────────────────────────────
        # Return the historical position trail for all active tracks.
        # Each trajectory is a deque of (x_center, y_center) positions.
        # Used by the visualizer to draw movement trails.
        # Libraries: collections (deque)

        # ── 8.11 reset ────────────────────────────────────────────────────────
        # Clear all active tracks and reset track ID counter.
        # Called on system restart or when switching scenes.
        # Libraries: collections, logging


# =============================================================================
# SECTION 9: POSITION ESTIMATOR CLASS
# =============================================================================

    # class PositionEstimator
    # Converts 2D bounding box coordinates in the image frame
    # into real-world positional estimates:
    #   - Azimuth angle  (horizontal direction to drone in degrees)
    #   - Elevation angle (vertical angle to drone in degrees)
    #   - Estimated distance (in meters, based on known drone size)
    #   - Normalized screen position (0.0 to 1.0 in x and y)
    #
    # Note: Without stereo calibration, distance is an approximation.
    # With both cameras, stereo triangulation can improve accuracy.
    # Libraries: numpy, math, dataclasses, logging

        # ── 9.1 __init__ ─────────────────────────────────────────────────────
        # Initialize camera intrinsic parameters:
        # frame dimensions, horizontal and vertical FOV,
        # focal length estimate, and reference drone size.
        # Libraries: numpy, math

        # ── 9.2 _compute_focal_length ────────────────────────────────────────
        # Estimate focal length in pixels from FOV and frame width:
        #   focal_length = (frame_width / 2) / tan(FOV_horizontal / 2)
        # Libraries: math

        # ── 9.3 compute_azimuth_elevation ────────────────────────────────────
        # Convert bounding box center (cx, cy) to angular position:
        #   azimuth   = angle from image center in horizontal axis
        #   elevation = angle from image center in vertical axis
        # Returns: (azimuth_degrees, elevation_degrees)
        # Libraries: math, numpy

        # ── 9.4 estimate_distance ─────────────────────────────────────────────
        # Estimate distance to drone using the known drone size
        # and the apparent size of the bounding box in the image:
        #   distance = (real_size × focal_length) / pixel_size
        # Returns distance estimate in meters.
        # Libraries: math, numpy

        # ── 9.5 estimate_stereo_position ──────────────────────────────────────
        # If the same drone is detected in both cameras simultaneously,
        # use the horizontal disparity between the two detections
        # to triangulate a more accurate distance estimate.
        # Requires known baseline distance between the two cameras.
        # Returns improved distance estimate in meters.
        # Libraries: math, numpy

        # ── 9.6 compute_position ──────────────────────────────────────────────
        # Master function that takes a TrackedDrone object and
        # returns a DronePosition object containing:
        #   azimuth, elevation, estimated_distance,
        #   normalized_x, normalized_y, confidence_in_estimate
        # Libraries: numpy, math, dataclasses


# =============================================================================
# SECTION 10: DATA STRUCTURES
# =============================================================================

    # ── 10.1 RawDetection ────────────────────────────────────────────────────
    # Dataclass: output of the detector (Section 6) before tracking.
    # Fields: x1, y1, x2, y2, confidence, class_id, class_name,
    #         camera_id, frame_id, timestamp
    # Libraries: dataclasses, datetime

    # ── 10.2 TrackedDrone ────────────────────────────────────────────────────
    # Dataclass: output of the tracker (Section 8) after tracking.
    # Extends RawDetection with tracking information.
    # Fields: all RawDetection fields PLUS
    #         track_id, track_state, hits, disappeared,
    #         velocity_x, velocity_y, predicted_box,
    #         frames_tracked, first_seen, last_seen
    # Libraries: dataclasses, datetime

    # ── 10.3 DronePosition ───────────────────────────────────────────────────
    # Dataclass: output of the position estimator (Section 9).
    # Fields: track_id, azimuth_deg, elevation_deg,
    #         estimated_distance_m, normalized_x, normalized_y,
    #         stereo_distance_m (optional), confidence,
    #         timestamp
    # Libraries: dataclasses, datetime

    # ── 10.4 DetectionFrame ──────────────────────────────────────────────────
    # Dataclass: all information associated with one processed frame.
    # Fields: camera_id, frame_id, timestamp, raw_frame (numpy),
    #         annotated_frame (numpy), detections (list of RawDetection),
    #         tracked_drones (list of TrackedDrone),
    #         positions (list of DronePosition),
    #         inference_time_ms, total_processing_time_ms
    # Libraries: dataclasses, numpy, datetime

    # ── 10.5 SystemStatus ────────────────────────────────────────────────────
    # Dataclass: snapshot of the entire system state at one moment.
    # Fields: camera0_fps, camera1_fps, inference_fps,
    #         active_track_count, confirmed_track_count,
    #         gpu_usage_pct, cpu_usage_pct, ram_usage_pct,
    #         temperature_c, total_detections, uptime_seconds,
    #         alert_active, timestamp
    # Libraries: dataclasses, datetime


# =============================================================================
# SECTION 11: DETECTION VISUALIZER CLASS
# =============================================================================

    # class DetectionVisualizer
    # Responsible for all drawing, annotation, and frame composition.
    # Takes annotated TrackedDrone and DronePosition data and
    # renders them onto frames for display.
    # Libraries: cv2, numpy, time

        # ── 11.1 __init__ ────────────────────────────────────────────────────
        # Initialize color palette (per track state), font settings,
        # display layout dimensions, and trajectory buffer references.
        # Libraries: cv2, numpy

        # ── 11.2 draw_bounding_boxes ─────────────────────────────────────────
        # Draw bounding boxes for all TrackedDrone objects on a frame.
        # Color-code by track state:
        #   CONFIRMED  → green
        #   TENTATIVE  → yellow
        #   LOST       → red (Kalman predicted box, dashed)
        # Libraries: cv2, numpy

        # ── 11.3 draw_track_labels ───────────────────────────────────────────
        # Draw text label above each bounding box showing:
        # Track ID, confidence score, and track state
        # e.g. "ID:3 | 0.87 | CONFIRMED"
        # Libraries: cv2

        # ── 11.4 draw_velocity_vector ────────────────────────────────────────
        # Draw an arrow from the center of each bounding box
        # indicating the drone's velocity direction and magnitude.
        # Arrow length scaled to velocity magnitude.
        # Libraries: cv2, numpy, math

        # ── 11.5 draw_trajectory ─────────────────────────────────────────────
        # Draw the historical position trail for each active track
        # as a fading polyline showing where the drone has flown.
        # Older positions are drawn with lower opacity.
        # Libraries: cv2, numpy, collections

        # ── 11.6 draw_position_data ──────────────────────────────────────────
        # Draw position estimate overlay next to each confirmed track:
        # azimuth, elevation, and estimated distance
        # e.g. "Az: 12.3° | El: -5.1° | ~42m"
        # Libraries: cv2

        # ── 11.7 draw_camera_info ────────────────────────────────────────────
        # Draw camera ID, capture FPS, and frame counter
        # in the top corner of each camera's frame.
        # Libraries: cv2

        # ── 11.8 draw_hud_overlay ────────────────────────────────────────────
        # Draw the heads-up display panel in the bottom of the
        # combined display frame showing system-wide stats:
        # inference FPS, active tracks, GPU/CPU/RAM usage,
        # temperature, total detections, system uptime
        # Libraries: cv2, numpy

        # ── 11.9 draw_alert_banner ───────────────────────────────────────────
        # Draw a flashing red banner across the top of the display
        # when one or more confirmed drones are actively detected.
        # Flash rate controlled by system clock.
        # Libraries: cv2, numpy, time

        # ── 11.10 combine_camera_frames ──────────────────────────────────────
        # Combine both annotated camera frames side by side
        # into a single unified display frame.
        # Add a dividing line and camera labels between them.
        # Resize to target display resolution.
        # Libraries: cv2, numpy

        # ── 11.11 render_frame ────────────────────────────────────────────────
        # Master render function. Takes two DetectionFrame objects
        # (one per camera) and a SystemStatus object and produces
        # the final display frame by calling all draw methods in order:
        #   1. draw_bounding_boxes (both frames)
        #   2. draw_track_labels
        #   3. draw_velocity_vector
        #   4. draw_trajectory
        #   5. draw_position_data
        #   6. draw_camera_info
        #   7. combine_camera_frames
        #   8. draw_hud_overlay
        #   9. draw_alert_banner (if active)
        # Returns final combined display frame (numpy array)
        # Libraries: cv2, numpy


# =============================================================================
# SECTION 12: ALERT SYSTEM CLASS
# =============================================================================

    # class AlertSystem
    # Evaluates confirmed track data each frame and decides
    # whether to trigger a detection alert.
    # Implements cooldown logic to prevent alert flooding.
    # Libraries: time, logging, datetime, threading, collections

        # ── 12.1 __init__ ────────────────────────────────────────────────────
        # Initialize alert state, cooldown timer, alert history list,
        # and configurable thresholds (min_confidence, min_confirmed_tracks,
        # cooldown_seconds)
        # Libraries: time, threading, collections (deque)

        # ── 12.2 evaluate ────────────────────────────────────────────────────
        # Evaluate current confirmed tracks this frame:
        # Check if alert conditions are met (confirmed tracks present,
        # confidence above threshold, cooldown has expired).
        # Return True if alert should fire, False otherwise.
        # Libraries: time

        # ── 12.3 trigger_alert ───────────────────────────────────────────────
        # Fire alert: log event to detection logger,
        # record alert in history with timestamp and track count,
        # start cooldown timer, set alert_active flag.
        # Libraries: logging, datetime, time

        # ── 12.4 clear_alert ─────────────────────────────────────────────────
        # Clear the active alert flag when no drones are
        # detected for a sustained period.
        # Libraries: time

        # ── 12.5 get_status ──────────────────────────────────────────────────
        # Return current alert status:
        # alert_active, cooldown_remaining, total_alerts_fired,
        # last_alert_timestamp
        # Libraries: time, datetime

        # ── 12.6 get_history ─────────────────────────────────────────────────
        # Return full alert history list with timestamps,
        # track counts, and confidence scores at time of alert
        # Libraries: collections (deque), datetime


# =============================================================================
# SECTION 13: DETECTION LOGGER CLASS
# =============================================================================

    # class DetectionLogger
    # Persists all detection and tracking events to disk.
    # Writes structured logs in CSV and JSON formats
    # and optionally saves annotated frame images.
    # Libraries: csv, json, os, cv2, datetime, pathlib, logging, pandas

        # ── 13.1 __init__ ────────────────────────────────────────────────────
        # Create output directory structure, initialize CSV log file
        # with column headers, set up image save subdirectory,
        # open session metadata file.
        # Libraries: os, pathlib, csv, datetime

        # ── 13.2 log_detection ───────────────────────────────────────────────
        # Write one row to the CSV log for each TrackedDrone detected:
        # timestamp, camera_id, track_id, track_state, confidence,
        # x1, y1, x2, y2, velocity_x, velocity_y,
        # azimuth, elevation, estimated_distance
        # Libraries: csv, datetime

        # ── 13.3 log_position ────────────────────────────────────────────────
        # Write position estimate data to a separate position log CSV:
        # timestamp, track_id, azimuth_deg, elevation_deg,
        # estimated_distance_m, stereo_distance_m, confidence
        # Libraries: csv, datetime

        # ── 13.4 save_detection_image ────────────────────────────────────────
        # Save the annotated frame to disk when a confirmed detection occurs.
        # Filename includes timestamp and track IDs.
        # Only saves if save_images flag is enabled in config.
        # Libraries: cv2, os, datetime

        # ── 13.5 generate_session_report ─────────────────────────────────────
        # At end of session, read the CSV log and generate a
        # JSON summary report containing:
        # session duration, total detections, unique tracks,
        # average confidence, detections per camera,
        # average estimated distance, alert count
        # Libraries: pandas, json, csv, datetime


# =============================================================================
# SECTION 14: SYSTEM MONITOR CLASS
# =============================================================================

    # class SystemMonitor
    # Polls Jetson hardware metrics in a background thread
    # and makes them available to the pipeline and HUD.
    # Libraries: threading, subprocess, time, re, logging, collections

        # ── 14.1 __init__ ────────────────────────────────────────────────────
        # Initialize polling interval, stats buffer,
        # running flag, and background thread handle.
        # Libraries: threading, collections (deque)

        # ── 14.2 start ───────────────────────────────────────────────────────
        # Start background polling thread as a daemon thread.
        # Libraries: threading

        # ── 14.3 _monitor_loop ───────────────────────────────────────────────
        # Continuously poll hardware stats at configured interval.
        # Call _read_tegrastats() and store results in rolling buffer.
        # Libraries: time, subprocess

        # ── 14.4 _read_tegrastats ────────────────────────────────────────────
        # Run tegrastats command, parse its output using regex
        # to extract: GPU usage %, CPU usage %, RAM usage %,
        # core temperature °C, swap usage %
        # Libraries: subprocess, re

        # ── 14.5 get_stats ───────────────────────────────────────────────────
        # Thread-safe return of the most recent hardware stats snapshot.
        # Libraries: threading

        # ── 14.6 stop ────────────────────────────────────────────────────────
        # Set running flag to False and join monitoring thread.
        # Libraries: threading


# =============================================================================
# SECTION 15: DETECTION PIPELINE CLASS
# =============================================================================

    # class DetectionPipeline
    # The central coordinator of the entire system.
    # Owns all major components and orchestrates the
    # frame-by-frame processing loop across both cameras.
    # Libraries: threading, queue, cv2, numpy, time, logging

        # ── 15.1 __init__ ────────────────────────────────────────────────────
        # Instantiate all system components:
        #   - Two CSICamera instances (cam0, cam1)
        #   - One DroneDetector instance (shared between cameras)
        #   - Two DroneTracker instances (one per camera)
        #   - One PositionEstimator instance
        #   - One DetectionVisualizer instance
        #   - One AlertSystem instance
        #   - One DetectionLogger instance
        #   - One SystemMonitor instance
        # Initialize frame queues, processing threads, and state flags.
        # Libraries: threading, queue, collections

        # ── 15.2 _initialize_and_verify ──────────────────────────────────────
        # Run startup verification for all components:
        # verify cameras are live, model loads cleanly,
        # output directories are writable.
        # Raise with descriptive message on any failure.
        # Libraries: logging, os

        # ── 15.3 _process_camera_frame ───────────────────────────────────────
        # Core per-frame processing function for one camera.
        # Pipeline for a single frame:
        #   1. Read frame from CSICamera (Section 5)
        #   2. Run detection via DroneDetector (Section 6)
        #   3. Run tracking update via DroneTracker (Section 8)
        #   4. Estimate position via PositionEstimator (Section 9)
        #   5. Package results into DetectionFrame (Section 10.4)
        #   6. Put DetectionFrame onto display queue
        # Libraries: time, numpy, logging

        # ── 15.4 _camera_processing_loop ─────────────────────────────────────
        # Thread loop that continuously calls _process_camera_frame()
        # for one camera. One instance of this runs per camera.
        # Handles frame timing and measures end-to-end latency.
        # Libraries: threading, time, logging

        # ── 15.5 _cross_camera_association ───────────────────────────────────
        # Periodically compare active tracks from both cameras
        # to identify when both cameras see the same physical drone.
        # Pass matched pairs to PositionEstimator for stereo distance.
        # Libraries: numpy, scipy

        # ── 15.6 _display_loop ───────────────────────────────────────────────
        # Main thread display loop. Reads DetectionFrames from
        # both camera queues, calls visualizer to render final frame,
        # evaluates alert system, and calls cv2.imshow().
        # Must run on the main thread (OpenCV display requirement).
        # Libraries: cv2, queue, time

        # ── 15.7 _evaluate_alerts ─────────────────────────────────────────────
        # Collect confirmed tracks from both cameras,
        # pass to AlertSystem.evaluate(), trigger or clear alert.
        # Log detections and positions via DetectionLogger.
        # Libraries: logging, time

        # ── 15.8 get_system_status ───────────────────────────────────────────
        # Aggregate stats from all components into a SystemStatus object:
        # camera fps, inference fps, track counts,
        # hardware metrics, alert state, uptime
        # Libraries: time, datetime, dataclasses

        # ── 15.9 start ───────────────────────────────────────────────────────
        # Start all background threads in order:
        #   1. SystemMonitor
        #   2. CSICamera threads (both)
        #   3. Camera processing threads (both)
        # Then hand control to _display_loop() on main thread.
        # Libraries: threading, logging

        # ── 15.10 stop ───────────────────────────────────────────────────────
        # Graceful shutdown sequence:
        #   1. Set stop flag on all threads
        #   2. Stop camera processing loops
        #   3. Stop CSICamera threads and release captures
        #   4. Stop SystemMonitor
        #   5. Finalize DetectionLogger and generate session report
        #   6. Destroy all OpenCV windows
        #   7. Log shutdown summary
        # Libraries: threading, cv2, logging, time


# =============================================================================
# SECTION 16: ARGUMENT PARSER
# =============================================================================

    # ── 16.1 build_argument_parser ────────────────────────────────────────────
    # Define all command-line arguments for runtime configuration:
    #   --model       path to TensorRT .engine file
    #   --conf        confidence threshold (default: 0.4)
    #   --iou         IoU threshold for NMS (default: 0.5)
    #   --resolution  camera resolution e.g. 1280x720
    #   --fps         target capture framerate
    #   --no-display  run headless without cv2.imshow
    #   --save-images save detection frames to disk
    #   --log-level   DEBUG / INFO / WARNING
    #   --output-dir  directory for logs and saved images
    #   --max-tracks  maximum number of simultaneous tracks
    # Libraries: argparse

    # ── 16.2 parse_and_validate ───────────────────────────────────────────────
    # Parse arguments, validate ranges and file paths,
    # and override the relevant config dataclass fields.
    # Exit with helpful message if any argument is invalid.
    # Libraries: argparse, os, sys, logging


# =============================================================================
# SECTION 17: STARTUP CHECKS
# =============================================================================

    # ── 17.1 check_gpu ────────────────────────────────────────────────────────
    # Verify CUDA is available and GPU memory is sufficient.
    # Print GPU device name and available memory.
    # Libraries: torch, logging

    # ── 17.2 check_cameras ────────────────────────────────────────────────────
    # Verify both CSI cameras are physically connected and
    # returning valid frames before pipeline starts.
    # Libraries: cv2, subprocess, logging

    # ── 17.3 check_model_file ─────────────────────────────────────────────────
    # Verify TensorRT .engine file exists, is readable,
    # and can be loaded by Ultralytics without error.
    # Libraries: os, pathlib, ultralytics, logging

    # ── 17.4 check_output_directories ────────────────────────────────────────
    # Verify or create all required output directories:
    # logs/, detections/images/, reports/
    # Libraries: os, pathlib

    # ── 17.5 check_dependencies ───────────────────────────────────────────────
    # Verify all required Python packages are installed
    # and meet minimum version requirements.
    # Libraries: importlib, pkg_resources, logging

    # ── 17.6 run_all_startup_checks ──────────────────────────────────────────
    # Run all checks above in sequence.
    # Print a pass/fail summary table to console.
    # Exit with code 1 and descriptive error if any check fails.
    # Libraries: sys, logging


# =============================================================================
# SECTION 18: SHUTDOWN HANDLER
# =============================================================================

    # ── 18.1 register_signal_handlers ────────────────────────────────────────
    # Register OS-level signal handlers for SIGINT (Ctrl+C)
    # and SIGTERM so both trigger a clean shutdown
    # rather than abrupt process termination.
    # Libraries: signal, sys

    # ── 18.2 graceful_shutdown ────────────────────────────────────────────────
    # Shutdown callback function:
    #   1. Call DetectionPipeline.stop()
    #   2. Finalize all logs
    #   3. Print session summary (duration, total detections,
    #      unique tracks, alerts fired) to console
    #   4. Exit cleanly with code 0
    # Libraries: signal, logging, sys, time


# =============================================================================
# SECTION 19: MAIN ENTRY POINT
# =============================================================================

    # ── 19.1 main ─────────────────────────────────────────────────────────────
    # Top-level orchestration function:
    #   1. Call parse_and_validate() to handle CLI arguments
    #   2. Configure logging (level, format, output file)
    #   3. Print system banner (project name, config summary)
    #   4. Call run_all_startup_checks()
    #   5. Register signal handlers
    #   6. Instantiate DetectionPipeline
    #   7. Call pipeline.start() → hands control to display loop
    #   8. On exit: graceful_shutdown()
    # Libraries: argparse, logging, sys, time

    # ── 19.2 Entry Guard ──────────────────────────────────────────────────────
    # Standard Python entry point guard:
    #   if __name__ == "__main__":
    #       main()