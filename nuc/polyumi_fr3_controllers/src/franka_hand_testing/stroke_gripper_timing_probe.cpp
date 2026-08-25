/// Franka Hand probe: sweep stroke LENGTH to pin down the acceleration.
//
// The width field refreshes at ~5 Hz, so the acceleration ramp (tens of ms) is invisible in the
// position trace and a_max cannot be fitted from it. move()'s blocking duration, however, is
// timed to the microsecond -- and short strokes are exactly where a_max decides the answer:
//
//   trapezoidal (long stroke):  t = dx/v + v/a
//   triangular  (short stroke): t = 2*sqrt(dx/a)          [never reaches the commanded speed]
//
// The crossover sits at dx = v^2/a, which at 100 mm/s is 3 mm if a = 3 m/s^2 but 50 mm if
// a = 0.2 m/s^2. So sweeping dx from 2 to 60 mm at high speed separates those hypotheses by
// hundreds of milliseconds. Two speeds, because the crossover moves with v^2 and a model that
// gets both right is much harder to fake than one fitted to a single speed.
//
// This also covers the short-move regime that every previous probe skipped: nothing so far has
// commanded less than 30 mm, and a policy issuing fine corrections will live below that.
//
// Writes samples and commands as two CSVs; analysis lives in Python. MOVES THE FINGERS.

#include <franka/exception.h>
#include <franka/gripper.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

// Every stroke starts here and returns here, so each length is measured in both directions from
// the same place and no reading is biased by where in the range it sat.
auto GRIPPER_BASE_M = 0.010;
auto GRIPPER_SETUP_SPEED_M_P_S = 0.05;

const double STROKES_M[] = {0.002, 0.003, 0.005, 0.008, 0.012, 0.020, 0.030, 0.045, 0.060};
const int N_STROKES = static_cast<int>(sizeof(STROKES_M) / sizeof(STROKES_M[0]));

// 100 mm/s is just under the measured xdot_max (~115), so the hand is speed-limited on long
// strokes and acceleration-limited on short ones. 30 mm/s moves the crossover down by 11x.
const double SPEEDS_M_P_S[] = {0.100, 0.030};
const int N_SPEEDS = static_cast<int>(sizeof(SPEEDS_M_P_S) / sizeof(SPEEDS_M_P_S[0]));

// Repeats per (stroke, speed, direction). Single moves carry ~50 ms of jitter and the effects
// being separated here are tens of ms, so averaging is not optional.
const int N_REPS = 3;

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
    double speed;
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

CommandRecord move(franka::Gripper& gripper, const char* label, double width, double speed) {
    CommandRecord rec{label, width, speed, now(), 0.0, "ok"};
    try {
        // A 2 mm move is small enough that a refusal is a real possibility; record it rather than
        // letting a declined command masquerade as a fast one.
        if (!gripper.move(width, speed)) {
            rec.outcome = "refused";
        }
    } catch (const franka::CommandException&) {
        rec.outcome = "aborted";
    } catch (const franka::Exception& e) {
        rec.outcome = "failed";
        std::fprintf(stderr, "  move failed: %s\n", e.what());
    }
    rec.t_complete = now();
    return rec;
}

int main(int argc, char** argv) {
    if (argc < 2 || argc > 3) {
        printf("usage: ./stroke_gripper_timing_probe <arm_ip> [out_prefix]\n");
        return -1;
    }
    const std::string prefix = (argc == 3) ? argv[2] : "stroke";

    franka::Gripper gripper(argv[1]);
    samples.reserve(20000);
    std::vector<CommandRecord> commands;

    printf("Homing gripper...\n");
    if (!gripper.homing()) {
        printf("Homing failed\n");
        return -1;
    }

    // Started only once homing has succeeded: a `std::thread` that is still joinable when a
    // function returns calls std::terminate() in its destructor, and homing (which can fail or
    // throw on an unreachable/faulted hand) is the one failure path before this that had no
    // thread to clean up.
    std::thread reader(read_loop, std::ref(gripper));

    printf("Moving to start position...\n");
    commands.push_back(move(gripper, "start", GRIPPER_BASE_M, GRIPPER_SETUP_SPEED_M_P_S));

    for (int si = 0; si < N_SPEEDS; si++) {
        const double v = SPEEDS_M_P_S[si];
        for (int di = 0; di < N_STROKES; di++) {
            const double stroke = STROKES_M[di];
            double sum_up = 0.0;
            double sum_down = 0.0;
            for (int rep = 0; rep < N_REPS; rep++) {
                const CommandRecord up = move(gripper, "move", GRIPPER_BASE_M + stroke, v);
                const CommandRecord down = move(gripper, "move", GRIPPER_BASE_M, v);
                commands.push_back(up);
                commands.push_back(down);
                sum_up += up.t_complete - up.t_sent;
                sum_down += down.t_complete - down.t_sent;
            }
            printf("%5.0f mm/s %5.1f mm: blocked up %7.1f ms, down %7.1f ms (mean of %d)\n",
                   v * 1e3, stroke * 1e3, sum_up / N_REPS * 1e3, sum_down / N_REPS * 1e3, N_REPS);
        }
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
    fprintf(out, "label,target_width,speed,t_sent,t_complete,outcome\n");
    for (const CommandRecord& c : commands) {
        fprintf(out, "%s,%.6f,%.6f,%.6f,%.6f,%s\n", c.label, c.target_width, c.speed, c.t_sent,
                c.t_complete, c.outcome);
    }
    fclose(out);

    printf("\nwrote %s_samples.csv (%zu samples) and %s_commands.csv (%zu commands)\n",
           prefix.c_str(), samples.size(), prefix.c_str(), commands.size());
    return 0;
}
