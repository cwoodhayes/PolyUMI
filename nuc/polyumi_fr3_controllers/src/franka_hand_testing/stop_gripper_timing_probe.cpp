/// Franka Hand timing probe that pre-empts every Move with stop().
//
// Same free-running reader as csv_gripper_timing_probe, but every Move is cut short by
// gripper.stop(), and the next Move goes out as soon as stop() returns. The question is whether
// stop() is a usable pre-emption primitive.
//
// The Move runs at a DELIBERATELY SLOW speed so the stroke lasts ~5 s, and the stop offset is
// swept from before the fingers have started to most of the way through. At full speed the whole
// stroke is only ~1.2 s, which is too little range to tell "stop is queued behind the Move" from
// "stop was sent too late to matter" -- and only ~6 width refreshes wide, so a halt is hard to see.
// Slow travel puts ~2 mm between refreshes, making a mid-stroke halt unmistakable.
//
// Targets alternate, so the fingers oscillate around mid-stroke rather than walking into a stop.
//
// Writes samples and commands as two CSVs; analysis lives in Python. MOVES THE FINGERS.

#include <franka/exception.h>
#include <franka/gripper.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <future>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

auto GRIPPER_OPEN_M = 0.07;
auto GRIPPER_CLOSED_M = 0.02;

// The trial speed: 50 mm at 10 mm/s is a ~5 s stroke, giving the sweep below room to work.
auto GRIPPER_SPEED_M_P_S = 0.01;
// Setup moves have nothing to measure, so they run at the usual speed.
auto GRIPPER_SETUP_SPEED_M_P_S = 0.05;

// Seconds after the Move is sent at which stop() goes out. Spans from inside the ~0.17 s start-up
// (fingers not yet moving) to nearly the end of the stroke.
const double STOP_DELAYS_S[] = {0.05, 0.15, 0.3, 0.6, 1.0, 1.5, 2.5, 3.5, 4.5};
const int N_TRIALS = static_cast<int>(sizeof(STOP_DELAYS_S) / sizeof(STOP_DELAYS_S[0]));

// A clear gap between trials. Without it a Move can be issued in the same millisecond a previous
// stop() returns, which is the one condition under which a Move was seen to abort.
auto TRIAL_REST_S = 0.5;

/// Seconds since the probe started, so every stamp is one comparable double.
const Clock::time_point EPOCH = Clock::now();
double now() { return Seconds(Clock::now() - EPOCH).count(); }

/// One (instant, width) sample off the gripper's own state stream.
struct Sample {
    double t;
    // GripperState.time, which is NOT a clock: libfranka builds it as Duration(message_id), so it
    // is the hand's datagram counter reinterpreted as milliseconds (convertGripperState, libfranka
    // src/gripper.cpp). A gap in it is datagrams we never saw.
    uint64_t message_id;
    double width;
};

struct CommandRecord {
    const char* label;
    double target_width;
    double t_sent;
    double t_complete;
    const char* outcome;
};

// Only the reader thread touches these, and main reads them only after the join -- so the join is
// the synchronisation and no mutex is needed.
std::vector<Sample> samples;
std::atomic<bool> reading{true};

void read_loop(franka::Gripper& gripper) {
    while (reading) {
        try {
            auto state = gripper.readOnce();
            // Stamped on return, the closest instant to the measurement libfranka exposes.
            samples.push_back({now(), state.time.toMSec(), state.width});
        } catch (const franka::Exception& e) {
            // A read that fails while a command is in flight is expected, not fatal.
            std::fprintf(stderr, "  [read] %s\n", e.what());
        }
    }
}

/// Move on its own thread, so main is free to stop() it partway through.
std::future<CommandRecord> move_async(franka::Gripper& gripper, double width, double speed) {
    return std::async(std::launch::async, [&gripper, width, speed] {
        CommandRecord rec{"move", width, now(), 0.0, "ok"};
        try {
            // A false return is a command the hand DECLINED, distinct from one it aborted.
            if (!gripper.move(width, speed)) {
                rec.outcome = "refused";
            }
        } catch (const franka::CommandException&) {
            // "Command aborted!" -- what a successful pre-emption should look like from here.
            rec.outcome = "aborted";
        } catch (const franka::Exception& e) {
            rec.outcome = "failed";
            std::fprintf(stderr, "  move failed: %s\n", e.what());
        }
        rec.t_complete = now();
        return rec;
    });
}

CommandRecord stop(franka::Gripper& gripper) {
    CommandRecord rec{"stop", 0.0, now(), 0.0, "ok"};
    try {
        if (!gripper.stop()) {
            rec.outcome = "refused";
        }
    } catch (const franka::Exception& e) {
        rec.outcome = "failed";
        std::fprintf(stderr, "  stop failed: %s\n", e.what());
    }
    rec.t_complete = now();
    return rec;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("usage: ./stop_gripper_timing_probe <arm_ip> [out_prefix]\n");
        return -1;
    }
    const std::string prefix = (argc == 3) ? argv[2] : "stop_probe";

    franka::Gripper gripper(argv[1]);
    samples.reserve(20000);
    std::vector<CommandRecord> commands;

    // Started before anything is commanded, so every command lands inside a stream already running.
    std::thread reader(read_loop, std::ref(gripper));

    printf("Homing gripper...\n");
    if (!gripper.homing()) {
        printf("Homing failed\n");
        return -1;
    }

    printf("Moving to start position...\n");
    commands.push_back(move_async(gripper, GRIPPER_CLOSED_M, GRIPPER_SETUP_SPEED_M_P_S).get());
    commands.back().label = "start";

    for (int i = 0; i < N_TRIALS; i++) {
        auto pending = move_async(gripper, (i % 2 == 0) ? GRIPPER_OPEN_M : GRIPPER_CLOSED_M,
                                  GRIPPER_SPEED_M_P_S);
        std::this_thread::sleep_for(Seconds(STOP_DELAYS_S[i]));

        const CommandRecord stopped = stop(gripper);
        // How long the aborted Move takes to unblock behind stop() is the number this probe
        // exists for, so it is joined right here rather than left to finish in its own time.
        const CommandRecord moved = pending.get();
        commands.push_back(moved);
        commands.push_back(stopped);

        printf("stop at %5.0f ms: move %-8s after %7.1f ms | stop %-8s in %7.1f ms | "
               "stop returned %7.1f ms after move\n",
               STOP_DELAYS_S[i] * 1e3, moved.outcome, (moved.t_complete - moved.t_sent) * 1e3,
               stopped.outcome, (stopped.t_complete - stopped.t_sent) * 1e3,
               (stopped.t_complete - moved.t_complete) * 1e3);

        std::this_thread::sleep_for(Seconds(TRIAL_REST_S));
    }

    // Keep reading a moment past the last command so the final settle is in the log too.
    std::this_thread::sleep_for(Seconds(1.0));
    reading = false;
    reader.join();

    FILE* out = fopen((prefix + "_samples.csv").c_str(), "w");
    fprintf(out, "t,message_id,width\n");
    for (const Sample& s : samples) {
        fprintf(out, "%.6f,%llu,%.6f\n", s.t, static_cast<unsigned long long>(s.message_id),
                s.width);
    }
    fclose(out);

    out = fopen((prefix + "_commands.csv").c_str(), "w");
    fprintf(out, "label,target_width,t_sent,t_complete,outcome\n");
    for (const CommandRecord& c : commands) {
        fprintf(out, "%s,%.6f,%.6f,%.6f,%s\n", c.label, c.target_width, c.t_sent, c.t_complete,
                c.outcome);
    }
    fclose(out);

    printf("\nwrote %s_samples.csv (%zu samples) and %s_commands.csv (%zu commands)\n",
           prefix.c_str(), samples.size(), prefix.c_str(), commands.size());
    return 0;
}
