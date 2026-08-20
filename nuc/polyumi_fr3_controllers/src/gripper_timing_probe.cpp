// Copyright (c) 2026 PolyUMI. MIT.
//
// What does a Franka Hand Move actually cost, with nothing in the way?
//
// docs/franka-inference-bringup.md asserts "~0.34 s of latency.gripper_exec is the hand itself",
// but every measurement behind that number was taken through franka_gripper, which the same doc
// records as worth 40-85 ms on its own (a detached std::thread per goal, and a state timer that
// calls the blocking readOnce() while holding the state mutex). So the split between firmware and
// middleware has never actually been observed. This measures it with no ROS, no action server and
// no executor: one franka::Gripper, one read thread, one command thread.
//
// Three numbers, because three different things are conflated in the 0.34 s:
//
//   A  rested Move       tcpSendRequest -> first width change, from a stationary hand.
//                        If this is ~0.34 s the floor really is firmware. If it collapses, it
//                        was franka_gripper and every downstream latency figure is too big.
//   B  abort cost        Issue a Move, supersede it partway, and time the SECOND Move's motion
//                        from its own send. fr3_gripper_bridge supersedes every 0.25 s, so if
//                        aborting re-pays A the hand can never leave the restart transient --
//                        which is exactly the 76%-stationary chirp result.
//   C  chained segments  Let a Move finish, immediately send the next. Move(width, speed) IS one
//                        constant-velocity segment, so if chaining is seamless a piecewise-linear
//                        trajectory can be executed as-is; if each link re-pays A it cannot.
//
// Run on the NUC, with franka_gripper NOT running -- only one process may hold the hand's
// connection (bring up with load_gripper:=false):
//
//     ros2 run polyumi_fr3_controllers gripper_timing_probe 192.168.51.20
//
// MOVES THE FINGERS. Nothing between them.

#include <franka/exception.h>
#include <franka/gripper.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

// Width change that counts as "the fingers moved". The encoder's resting noise is well under a
// tenth of a millimetre; this clears it without waiting for meaningful travel, since what is being
// timed is the ONSET, not the stroke.
constexpr double kOnsetThresholdM = 0.0005;

// The sweep the probe works over. Mid-stroke so neither end hits a hard stop, and wide enough that
// a Move at kSpeedMps runs for well over a second -- B has to be able to supersede a move that is
// still genuinely in flight.
constexpr double kLowWidthM = 0.02;
constexpr double kHighWidthM = 0.07;
constexpr double kSpeedMps = 0.03;

// How long to wait for an onset before declaring the hand did not move at all.
constexpr double kOnsetTimeoutS = 5.0;

// Settle window before each timed event. Longer than any plausible command->motion delay, so a
// measurement can never be contaminated by the previous rep still finishing.
constexpr double kSettleS = 1.5;

/// One (instant, width) sample off the gripper's own state stream.
struct Sample {
  double t;
  double width;
};

/**
 * Continuously readOnce()s the hand into a timestamped buffer.
 *
 * Its own thread for the same reason the Phase 1 node will use one: readOnce() blocks until a
 * fresh UDP datagram arrives, so anything sharing its thread stalls for most of each inter-datagram
 * gap. That is the franka_gripper defect this probe must not reproduce, or it would measure it.
 *
 * Samples are stamped when readOnce() RETURNS, which is the closest instant to the measurement
 * that libfranka exposes -- GripperState carries no measure timestamp (UMI reads the WSG's).
 */
class StateReader {
 public:
  StateReader(franka::Gripper& gripper, Clock::time_point epoch)
      : gripper_(gripper), epoch_(epoch), thread_([this] { loop(); }) {}

  ~StateReader() {
    running_ = false;
    if (thread_.joinable()) {
      thread_.join();
    }
  }

  StateReader(const StateReader&) = delete;
  StateReader& operator=(const StateReader&) = delete;

  /// Every sample taken so far.
  std::vector<Sample> samples() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_;
  }

  /// Most recent width, or nullopt if nothing has arrived yet.
  std::optional<double> width() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return samples_.empty() ? std::nullopt : std::optional<double>(samples_.back().width);
  }

  /**
   * Instant of the first sample after `since` that has moved `threshold` off `rest`.
   *
   * Keyed on the sample's own stamp rather than on when this loop noticed, so the polling rate
   * cannot inflate the answer.
   */
  std::optional<double> waitForOnset(double since, double rest, double threshold, double timeout_s) {
    const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(Seconds(timeout_s));
    while (Clock::now() < deadline) {
      for (const Sample& s : samples()) {
        if (s.t > since && std::fabs(s.width - rest) > threshold) {
          return s.t;
        }
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::nullopt;
  }

  /**
   * Instant the fingers REVERSE after `since`, i.e. turn back by `threshold` from their extreme.
   *
   * Section B supersedes a Move that is still travelling, so the hand is already moving when the
   * second command goes out. A plain onset test would fire on the first Move's motion within a
   * few ms and measure nothing. What marks the second Move taking effect is the turnaround, so
   * this tracks the running extreme since `since` and waits for the width to come back off it.
   */
  std::optional<double> waitForReversal(double since, double threshold, double timeout_s) {
    const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(Seconds(timeout_s));
    while (Clock::now() < deadline) {
      std::optional<double> low;
      std::optional<double> high;
      for (const Sample& s : samples()) {
        if (s.t <= since) {
          continue;
        }
        // Direction-agnostic: whichever extreme the hand walks away from by `threshold` first is
        // the reversal, so this works for a supersede in either direction.
        if (high.has_value() && *high - s.width > threshold) {
          return s.t;
        }
        if (low.has_value() && s.width - *low > threshold) {
          return s.t;
        }
        high = high.has_value() ? std::max(*high, s.width) : s.width;
        low = low.has_value() ? std::min(*low, s.width) : s.width;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return std::nullopt;
  }

  /// Block until the width stops changing, so a timed event cannot start on a coasting hand.
  bool waitUntilStill(double tol, double window_s, double timeout_s) {
    const auto deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(Seconds(timeout_s));
    while (Clock::now() < deadline) {
      const double cutoff = now() - window_s;
      double lo = 0.0;
      double hi = 0.0;
      std::size_t n = 0;
      for (const Sample& s : samples()) {
        if (s.t < cutoff) {
          continue;
        }
        lo = (n == 0) ? s.width : std::min(lo, s.width);
        hi = (n == 0) ? s.width : std::max(hi, s.width);
        ++n;
      }
      if (n >= 3 && hi - lo < tol) {
        return true;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return false;
  }

  /// Fraction of samples in [from, to] where the width was changing, and the longest stall.
  std::pair<double, double> motionSummary(double from, double to, double tol) const {
    std::vector<Sample> in;
    for (const Sample& s : samples()) {
      if (s.t >= from && s.t <= to) {
        in.push_back(s);
      }
    }
    if (in.size() < 2) {
      return {0.0, 0.0};
    }
    std::size_t moving = 0;
    double longest_stall = 0.0;
    double stall_start = in.front().t;
    for (std::size_t i = 1; i < in.size(); ++i) {
      if (std::fabs(in[i].width - in[i - 1].width) > tol) {
        ++moving;
        longest_stall = std::max(longest_stall, in[i].t - stall_start);
        stall_start = in[i].t;
      }
    }
    longest_stall = std::max(longest_stall, in.back().t - stall_start);
    return {static_cast<double>(moving) / static_cast<double>(in.size() - 1), longest_stall};
  }

  double now() const { return Seconds(Clock::now() - epoch_).count(); }

  /// Median interval between samples, i.e. the state rate this probe actually achieved.
  double medianIntervalS() const {
    const std::vector<Sample> s = samples();
    if (s.size() < 3) {
      return 0.0;
    }
    std::vector<double> gaps;
    gaps.reserve(s.size() - 1);
    for (std::size_t i = 1; i < s.size(); ++i) {
      gaps.push_back(s[i].t - s[i - 1].t);
    }
    std::nth_element(gaps.begin(), gaps.begin() + gaps.size() / 2, gaps.end());
    return gaps[gaps.size() / 2];
  }

 private:
  void loop() {
    while (running_) {
      try {
        const franka::GripperState state = gripper_.readOnce();
        const double t = now();
        std::lock_guard<std::mutex> lock(mutex_);
        samples_.push_back({t, state.width});
      } catch (const franka::Exception& e) {
        // A read that fails while a command is being superseded is expected, not fatal -- the
        // point of the probe is to run commands over the top of this stream.
        std::fprintf(stderr, "  [read] %s\n", e.what());
      }
    }
  }

  franka::Gripper& gripper_;
  Clock::time_point epoch_;
  mutable std::mutex mutex_;
  std::vector<Sample> samples_;
  std::atomic<bool> running_{true};
  std::thread thread_;
};

/// Park at `width` and wait for the hand to stop, so a timed event starts from a known rest.
bool settleAt(franka::Gripper& gripper, StateReader& reader, double width) {
  try {
    if (!gripper.move(width, kSpeedMps)) {
      std::fprintf(stderr,
                   "  settle move to %.0f mm REFUSED (move() returned false) -- is the hand homed?\n",
                   width * 1e3);
      return false;
    }
  } catch (const franka::Exception& e) {
    std::fprintf(stderr, "  settle move failed: %s\n", e.what());
    return false;
  }
  std::this_thread::sleep_for(std::chrono::duration_cast<Clock::duration>(Seconds(kSettleS)));
  return reader.width().has_value();
}

/**
 * Send a Move on its own thread and report when tcpSendRequest went out.
 *
 * Move() blocks until the motion ENDS (libfranka gripper.cpp: tcpSendRequest then
 * tcpBlockingReceiveResponse), so timing the call itself would measure the stroke, not the
 * command. It has to run detached from the timing. `aborted` reports whether this Move was the one
 * superseded -- that is the CommandException carrying "Command aborted!".
 */
struct MoveHandle {
  double sent_at;
  std::thread thread;
  std::shared_ptr<std::atomic<bool>> finished;
  std::shared_ptr<std::atomic<bool>> aborted;
  /// What move() RETURNED. Not redundant with `aborted`: executeCommand maps kUnsuccessful to a
  /// plain `false` and only kFail/kAborted throw, so a command the hand simply refused (an unhomed
  /// hand refuses every Move) comes back false with no exception at all. Ignoring this makes a
  /// rejected command indistinguishable from one that ran and moved nothing.
  std::shared_ptr<std::atomic<bool>> ok;
};

MoveHandle sendMove(franka::Gripper& gripper, StateReader& reader, double width, double speed) {
  auto finished = std::make_shared<std::atomic<bool>>(false);
  auto aborted = std::make_shared<std::atomic<bool>>(false);
  auto ok = std::make_shared<std::atomic<bool>>(false);
  const double sent_at = reader.now();
  std::thread t([&gripper, width, speed, finished, aborted, ok] {
    try {
      *ok = gripper.move(width, speed);
    } catch (const franka::Exception&) {
      *aborted = true;
    }
    *finished = true;
  });
  return {sent_at, std::move(t), finished, aborted, ok};
}

void report(const char* label, std::optional<double> onset, const MoveHandle& h) {
  if (onset.has_value()) {
    std::printf("  %-46s %7.1f ms\n", label, (*onset - h.sent_at) * 1e3);
  } else if (*h.aborted) {
    std::printf("  %-46s   NO MOTION (threw)\n", label);
  } else if (!*h.ok) {
    // move() returned false: the hand received the command and declined it. Almost always an
    // unhomed hand -- readOnce() works fine unhomed, so the connection looks healthy.
    std::printf("  %-46s   REFUSED (move() returned false)\n", label);
  } else {
    std::printf("  %-46s   NO MOTION (move() succeeded)\n", label);
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2 || argc > 3 || (argc == 3 && std::string(argv[2]) != "--home")) {
    std::fprintf(stderr,
                 "usage: gripper_timing_probe <robot_ip> [--home]\n"
                 "  MOVES THE FINGERS. franka_gripper must NOT be running (load_gripper:=false).\n"
                 "  --home runs Gripper::homing() first. An unhomed hand REFUSES every Move (they\n"
                 "  return false without throwing) while readOnce() keeps working, so the\n"
                 "  connection looks healthy and nothing moves.\n");
    return 2;
  }

  try {
    franka::Gripper gripper(argv[1]);

    // Read once before the reader thread starts, so this is uncontended, and print everything the
    // hand reports. max_width is the giveaway for an unhomed hand: it reads 0 until homing has
    // established the stroke, and it is not published on any topic, so this is the only place it
    // is visible at all.
    const franka::GripperState initial = gripper.readOnce();
    std::printf("\nConnected. width=%.4f m  max_width=%.4f m  grasped=%d  temp=%d C\n",
                initial.width, initial.max_width, static_cast<int>(initial.is_grasped),
                initial.temperature);

    if (argc == 3) {
      std::printf("Homing (fingers will open fully, then close) ...\n");
      if (!gripper.homing()) {
        std::fprintf(stderr, "homing() returned false. The hand will not accept Moves.\n");
        return 1;
      }
      const franka::GripperState homed = gripper.readOnce();
      std::printf("Homed. width=%.4f m  max_width=%.4f m\n", homed.width, homed.max_width);
    }

    const Clock::time_point epoch = Clock::now();
    StateReader reader(gripper, epoch);

    // Fail here rather than misattributing an empty state stream to the hand not moving.
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    if (!reader.width().has_value()) {
      std::fprintf(stderr, "No gripper state arrived. Is another process holding the connection?\n");
      return 1;
    }
    std::printf("State stream: %.1f ms median interval (%.1f Hz).\n",
                reader.medianIntervalS() * 1e3,
                reader.medianIntervalS() > 0 ? 1.0 / reader.medianIntervalS() : 0.0);
    std::printf("FINGERS WILL MOVE between %.0f and %.0f mm.\n\n", kLowWidthM * 1e3, kHighWidthM * 1e3);

    // ---- A: rested Move -------------------------------------------------------------------
    // The headline number. Everything else is measured against it.
    std::printf("A  rested Move: send -> first motion\n");
    std::vector<double> rested;
    for (int rep = 0; rep < 4; ++rep) {
      const double from = (rep % 2 == 0) ? kLowWidthM : kHighWidthM;
      const double to = (rep % 2 == 0) ? kHighWidthM : kLowWidthM;
      if (!settleAt(gripper, reader, from)) {
        return 1;
      }
      const double rest = *reader.width();
      MoveHandle h = sendMove(gripper, reader, to, kSpeedMps);
      const auto onset = reader.waitForOnset(h.sent_at, rest, kOnsetThresholdM, kOnsetTimeoutS);
      h.thread.join();
      report(rep % 2 == 0 ? "rep (opening)" : "rep (closing)", onset, h);
      if (onset.has_value()) {
        rested.push_back(*onset - h.sent_at);
      }
    }
    double rested_median = 0.0;
    if (!rested.empty()) {
      std::nth_element(rested.begin(), rested.begin() + rested.size() / 2, rested.end());
      rested_median = rested[rested.size() / 2];
      std::printf("   -> median %.1f ms\n", rested_median * 1e3);
    }

    // ---- B: abort cost --------------------------------------------------------------------
    // Does superseding re-pay A? fr3_gripper_bridge supersedes every 250 ms, so the 250 ms row is
    // the one that describes the shipped configuration.
    std::printf("\nB  superseded Move: second send -> its own first motion\n");
    for (const double delay_s : {0.05, 0.10, 0.25, 0.50}) {
      if (!settleAt(gripper, reader, kLowWidthM)) {
        return 1;
      }
      // First Move heads for the far end; it is still travelling when the second supersedes it.
      MoveHandle first = sendMove(gripper, reader, kHighWidthM, kSpeedMps);
      std::this_thread::sleep_for(std::chrono::duration_cast<Clock::duration>(Seconds(delay_s)));
      // Reverse, so the second Move announces itself as a turnaround. The hand is still
      // travelling from the first Move at this point, so nothing but a direction change can
      // distinguish the second taking effect from the first continuing.
      MoveHandle second = sendMove(gripper, reader, kLowWidthM, kSpeedMps);
      const auto onset = reader.waitForReversal(second.sent_at, kOnsetThresholdM, kOnsetTimeoutS);
      first.thread.join();
      second.thread.join();
      char label[64];
      std::snprintf(label, sizeof(label), "superseded after %3.0f ms%s", delay_s * 1e3,
                    *first.aborted ? " (first aborted)" : " (first NOT aborted)");
      report(label, onset, second);
    }
    if (rested_median > 0.0) {
      std::printf("   -> compare against A's %.1f ms: equal means an abort re-pays the full\n"
                  "      start-up, so superseding faster than that leaves the hand permanently\n"
                  "      in the restart transient.\n",
                  rested_median * 1e3);
    }

    // ---- C: chained segments --------------------------------------------------------------
    // Whether a piecewise-linear trajectory can be executed as consecutive Moves without the hand
    // stopping at every knot.
    std::printf("\nC  chained Moves: does a completed Move flow into the next?\n");
    if (!settleAt(gripper, reader, kLowWidthM)) {
      return 1;
    }
    const double step = (kHighWidthM - kLowWidthM) / 3.0;
    for (int seg = 1; seg <= 3; ++seg) {
      const double target = kLowWidthM + step * seg;
      // Wait for a genuine standstill first. Without this, `before` is sampled off a hand still
      // coasting from the previous segment and the detector fires on that residue -- which is what
      // produced a spurious 0.4 ms here on the first run.
      reader.waitUntilStill(kOnsetThresholdM, 0.3, 3.0);
      const double before = *reader.width();
      MoveHandle h = sendMove(gripper, reader, target, kSpeedMps);
      const auto onset = reader.waitForOnset(h.sent_at, before, kOnsetThresholdM, kOnsetTimeoutS);
      h.thread.join();  // Runs to completion -- no supersede anywhere in this section.
      char label[64];
      std::snprintf(label, sizeof(label), "segment %d -> %.0f mm", seg, target * 1e3);
      report(label, onset, h);
    }
    std::printf("   -> move() returns only when the motion ENDS, so each segment here starts from\n"
                "      rest and this is really A repeated. Section D is the case that matters.\n");

    // ---- D: extending a move that is already running ----------------------------------------
    // The one that decides the command policy. B showed a reversal costs only ~60 ms once the hand
    // is moving, against ~360 ms from rest -- so the question is whether a controller can re-issue
    // periodically WITHOUT ever letting the hand stop. That is exactly what a lookahead scheme
    // does: every period, command a point further along the trajectory.
    std::printf("\nD  extending in-flight: can periodic re-issue keep the hand moving?\n");
    for (const double period_s : {0.15, 0.25}) {
      if (!settleAt(gripper, reader, kLowWidthM)) {
        return 1;
      }
      reader.waitUntilStill(kOnsetThresholdM, 0.3, 3.0);
      const double rest = *reader.width();
      // Aim only part way, then keep pushing the target further out before it is reached, so no
      // Move ever runs out of travel -- the hand should never see a reason to stop.
      double target = kLowWidthM + 0.015;
      MoveHandle h = sendMove(gripper, reader, target, kSpeedMps);
      const auto onset = reader.waitForOnset(h.sent_at, rest, kOnsetThresholdM, kOnsetTimeoutS);
      std::vector<MoveHandle> issued;
      const double extend_from = reader.now();
      while (target < kHighWidthM - 0.01) {
        std::this_thread::sleep_for(std::chrono::duration_cast<Clock::duration>(Seconds(period_s)));
        target += kSpeedMps * period_s * 1.5;  // Stay ahead of where the hand can have reached.
        issued.push_back(sendMove(gripper, reader, std::min(target, kHighWidthM), kSpeedMps));
      }
      const double extend_to = reader.now();
      h.thread.join();
      for (MoveHandle& m : issued) {
        m.thread.join();
      }
      const auto [moving_frac, longest_stall] =
          reader.motionSummary(extend_from, extend_to, kOnsetThresholdM);
      std::printf("  re-issued every %3.0f ms (%zu moves): onset %.0f ms, then moving %3.0f%% of\n"
                  "      samples, longest stall %.0f ms\n",
                  period_s * 1e3, issued.size(),
                  onset.has_value() ? (*onset - h.sent_at) * 1e3 : -1.0, moving_frac * 100.0,
                  longest_stall * 1e3);
    }
    std::printf("   -> high moving%% with a short longest stall means a lookahead controller can\n"
                "      hold the hand in continuous motion, where corrections cost B's ~60 ms rather\n"
                "      than A's ~360 ms. A long stall means every re-issue re-pays the cold start.\n");

    settleAt(gripper, reader, (kLowWidthM + kHighWidthM) / 2.0);
    std::printf("\nState stream ended at %.1f ms median interval over %zu samples.\n\n",
                reader.medianIntervalS() * 1e3, reader.samples().size());
  } catch (const franka::Exception& e) {
    std::fprintf(stderr, "libfranka: %s\n", e.what());
    return 1;
  }
  return 0;
}
